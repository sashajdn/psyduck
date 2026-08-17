use std::{error::Error, io::Write};

use opentelemetry::{KeyValue, metrics::MeterProvider};
use serde::{Serialize, Serializer};
use strum::IntoStaticStr;
use tracing::info;

use crate::{
    operation::{CaptureConfig, OperationMetrics, OperationTimer, TimingClock},
    prometheus_exporter,
};

pub const ATTRIBUTE_OPERATION: &str = "operation";
pub const ATTRIBUTE_BACKEND: &str = "backend";
pub const ATTRIBUTE_CLOCK: &str = "clock";
pub const ATTRIBUTE_DTYPE: &str = "dtype";
pub const ATTRIBUTE_M: &str = "m";
pub const ATTRIBUTE_N: &str = "n";
pub const ATTRIBUTE_K: &str = "k";
pub const CHECKSUM_RELATIVE_TOLERANCE: f64 = 1.0e-6;

#[derive(Clone, Copy, Debug, Eq, IntoStaticStr, PartialEq, Serialize)]
#[strum(serialize_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum OperationKind {
    Add,
    Matmul,
}

#[derive(Clone, Copy, Debug, Eq, IntoStaticStr, PartialEq, Serialize)]
#[strum(serialize_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkTarget {
    Host,
    Device,
}

#[derive(Clone, Copy, Debug, Eq, IntoStaticStr, PartialEq, Serialize)]
#[strum(serialize_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum DataType {
    F32,
    F64,
}

#[derive(Clone, Copy, Debug, Eq, IntoStaticStr, PartialEq, Serialize)]
pub enum FlopsConvention {
    #[strum(serialize = "2*M*N*K")]
    #[serde(rename = "2*M*N*K")]
    MatmulMultiplyAddAsTwo,
}

#[derive(Clone, Copy, Debug)]
pub struct OperationWork {
    floating_point_operations: u128,
    algorithmic_bytes: Option<u128>,
}

impl OperationWork {
    #[inline]
    pub const fn new(floating_point_operations: u128, algorithmic_bytes: Option<u128>) -> Self {
        Self {
            floating_point_operations,
            algorithmic_bytes,
        }
    }

    #[inline]
    pub const fn matmul(m: usize, n: usize, k: usize, element_bytes: usize) -> Self {
        let m = m as u128;
        let n = n as u128;
        let k = k as u128;
        let element_bytes = element_bytes as u128;

        Self::new(2 * m * n * k, Some(element_bytes * (m * k + k * n + m * n)))
    }

    #[inline]
    pub const fn floating_point_operations(self) -> u128 {
        self.floating_point_operations
    }

    #[inline]
    pub const fn algorithmic_bytes(self) -> Option<u128> {
        self.algorithmic_bytes
    }

    #[inline]
    pub fn arithmetic_intensity(self) -> Option<f64> {
        self.algorithmic_bytes
            .map(|bytes| self.floating_point_operations as f64 / bytes as f64)
    }

    #[inline]
    pub fn flops_per_second(self, elapsed_seconds: f64) -> f64 {
        self.floating_point_operations as f64 / elapsed_seconds
    }

    #[inline]
    pub fn bytes_per_second(self, elapsed_seconds: f64) -> Option<f64> {
        self.algorithmic_bytes
            .map(|bytes| bytes as f64 / elapsed_seconds)
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Distribution {
    pub min: f64,
    pub p01: f64,
    pub p05: f64,
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub p100: f64,
}

impl Distribution {
    pub fn from_samples(samples: impl IntoIterator<Item = f64>) -> Option<Self> {
        let mut samples = samples
            .into_iter()
            .filter(|sample| sample.is_finite())
            .collect::<Vec<_>>();
        if samples.is_empty() {
            return None;
        }

        samples.sort_by(f64::total_cmp);

        Some(Self {
            min: samples[0],
            p01: percentile(&samples, 0.01),
            p05: percentile(&samples, 0.05),
            p50: percentile(&samples, 0.50),
            p90: percentile(&samples, 0.90),
            p95: percentile(&samples, 0.95),
            p99: percentile(&samples, 0.99),
            p100: samples[samples.len() - 1],
        })
    }
}

fn percentile(sorted_samples: &[f64], percentile: f64) -> f64 {
    let rank = percentile * (sorted_samples.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let fraction = rank - lower as f64;

    sorted_samples[lower] + (sorted_samples[upper] - sorted_samples[lower]) * fraction
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct BenchmarkShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct HostMetadata {
    pub architecture: &'static str,
    pub cpu_model: Option<String>,
    pub hostname: Option<String>,
    pub logical_cpus: usize,
    pub operating_system: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GitMetadata {
    pub commit: Option<String>,
    pub dirty: Option<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BenchmarkEnvironment {
    pub git: GitMetadata,
    pub host: HostMetadata,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum BenchmarkSchemaVersion {
    V1 = 1,
}

impl Serialize for BenchmarkSchemaVersion {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u32(*self as u32)
    }
}

#[derive(Clone, Debug)]
pub struct BenchmarkConfig {
    pub operation: OperationKind,
    pub target: BenchmarkTarget,
    pub dtype: DataType,
    pub shape: BenchmarkShape,
    pub capture: CaptureConfig,
    pub work: OperationWork,
    pub flops_convention: FlopsConvention,
    pub generator: &'static str,
    pub seed: u64,
    pub expected_output_sum: Option<f64>,
    pub environment: BenchmarkEnvironment,
}

pub trait BenchmarkedOperation<B> {
    type Error;

    fn execute(&mut self, backend: &B) -> Result<(), Self::Error>;
    fn output_sum(&self, backend: &B) -> Result<f64, Self::Error>;
}

#[derive(Debug, thiserror::Error)]
pub enum BenchmarkError {
    #[error("benchmark dimension {name}={value} does not fit in an i64 metric attribute")]
    InvalidDimension { name: &'static str, value: usize },
    #[error("failed to initialize benchmark telemetry")]
    Telemetry(#[source] opentelemetry_sdk::error::OTelSdkError),
    #[error("benchmark operation failed")]
    Operation(#[source] Box<dyn Error + Send + Sync>),
    #[error("failed to read benchmark output")]
    Output(#[source] Box<dyn Error + Send + Sync>),
    #[error("benchmark produced no latency samples")]
    MissingLatencySamples,
    #[error("benchmark produced no non-zero latency samples")]
    MissingRateSamples,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct AggregateReport {
    pub elapsed_us: f64,
    pub flops_per_second: f64,
    pub operations_per_second: f64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct CaptureReport {
    pub operation_count: usize,
    pub sample_count: usize,
    pub sample_every: usize,
    pub warmup_operations: usize,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct WorkReport {
    pub algorithmic_bytes_per_operation: Option<u128>,
    pub arithmetic_intensity_flops_per_byte: Option<f64>,
    pub flops_per_operation: u128,
    pub flops_convention: FlopsConvention,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ChecksumReport {
    pub actual: f64,
    pub expected: f64,
    pub absolute_error: f64,
    pub tolerance: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct BenchmarkReport {
    pub schema_version: BenchmarkSchemaVersion,
    pub aggregate: AggregateReport,
    pub algorithmic_bytes_per_second: Option<Distribution>,
    pub capture: CaptureReport,
    pub clock: TimingClock,
    pub correctness: Option<ChecksumReport>,
    pub dtype: DataType,
    pub environment: BenchmarkEnvironment,
    pub flops_per_second: Distribution,
    pub generator: &'static str,
    pub latency_us: Distribution,
    pub operation: OperationKind,
    pub output_sum: f64,
    pub seed: u64,
    pub shape: BenchmarkShape,
    pub target: BenchmarkTarget,
    pub work: WorkReport,
}

impl BenchmarkReport {
    /// Writes this report as one newline-terminated JSON record.
    pub fn write_to<W: Write>(&self, mut writer: W) -> Result<(), BenchmarkReportWriteError> {
        serde_json::to_writer(&mut writer, self)?;
        writer.write_all(b"\n")?;
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BenchmarkReportWriteError {
    #[error("failed to serialize benchmark report")]
    Serialize(#[from] serde_json::Error),
    #[error("failed to write benchmark report")]
    Write(#[from] std::io::Error),
}

pub struct Benchmark;

impl Benchmark {
    pub fn run<B, O>(
        config: BenchmarkConfig,
        backend: &B,
        mut operation: O,
    ) -> Result<BenchmarkReport, BenchmarkError>
    where
        B: OperationTimer,
        O: BenchmarkedOperation<B>,
        O::Error: Error + Send + Sync + From<B::Error> + 'static,
    {
        // Setup metrics.
        let attributes = Self::benchmark_attributes::<B>(&config)?;
        let (prometheus_registry, meter_provider) =
            prometheus_exporter::meter_provider().map_err(BenchmarkError::Telemetry)?;
        let histograms = OperationHistograms::new(&meter_provider.meter("psyduck.benchmark"));

        let operation_name: &'static str = config.operation.into();
        let target_name: &'static str = config.target.into();
        let dtype_name: &'static str = config.dtype.into();
        let clock_name: &'static str = B::CLOCK.into();

        info!(
            operation = operation_name,
            backend = target_name,
            clock = clock_name,
            dtype = dtype_name,
            m = config.shape.m,
            n = config.shape.n,
            k = config.shape.k,
            warmup_operations = config.capture.warmup_operations(),
            operation_count = config.capture.operation_count(),
            sample_every = config.capture.sample_every(),
            expected_output_sum = ?config.expected_output_sum,
            hostname = ?config.environment.host.hostname,
            git_commit = ?config.environment.git.commit,
            "starting benchmark run"
        );

        // Execute the run.
        let metrics = OperationMetrics::capture(backend, config.capture, |backend| {
            operation.execute(backend)
        })
        .map_err(|error| BenchmarkError::Operation(Box::new(error)))?;

        let latency_us = Distribution::from_samples(
            metrics
                .iter()
                .map(|metric| metric.elapsed().as_secs_f64() * 1_000_000.0),
        )
        .ok_or(BenchmarkError::MissingLatencySamples)?;

        let flops_per_second = Distribution::from_samples(metrics.iter().filter_map(|metric| {
            let seconds = metric.elapsed().as_secs_f64();
            (seconds > 0.0).then(|| config.work.flops_per_second(seconds))
        }))
        .ok_or(BenchmarkError::MissingRateSamples)?;

        let algorithmic_bytes_per_second =
            Distribution::from_samples(metrics.iter().filter_map(|metric| {
                let seconds = metric.elapsed().as_secs_f64();
                (seconds > 0.0)
                    .then(|| config.work.bytes_per_second(seconds))
                    .flatten()
            }));

        histograms.record(&metrics, config.work, &attributes);
        debug_assert!(!prometheus_registry.gather().is_empty());

        let total_seconds = metrics.total_elapsed().as_secs_f64();

        // Output inspection happens after capture so device downloads and
        // checksum calculation are never included in operation latency.
        let output_sum = operation
            .output_sum(backend)
            .map_err(|error| BenchmarkError::Output(Box::new(error)))?;
        let correctness = config.expected_output_sum.map(|expected| {
            let absolute_error = (output_sum - expected).abs();
            let tolerance = expected.abs().max(1.0) * CHECKSUM_RELATIVE_TOLERANCE;
            ChecksumReport {
                actual: output_sum,
                expected,
                absolute_error,
                tolerance,
                passed: absolute_error <= tolerance,
            }
        });

        let report = BenchmarkReport {
            schema_version: BenchmarkSchemaVersion::V1,
            aggregate: AggregateReport {
                elapsed_us: total_seconds * 1_000_000.0,
                flops_per_second: config.work.floating_point_operations() as f64
                    * metrics.operation_count() as f64
                    / total_seconds,
                operations_per_second: metrics.operation_count() as f64 / total_seconds,
            },
            algorithmic_bytes_per_second,
            capture: CaptureReport {
                operation_count: metrics.operation_count(),
                sample_count: metrics.sample_count(),
                sample_every: metrics.sample_every(),
                warmup_operations: metrics.warmup_operations(),
            },
            clock: metrics.clock(),
            correctness,
            dtype: config.dtype,
            environment: config.environment,
            flops_per_second,
            generator: config.generator,
            latency_us,
            operation: config.operation,
            output_sum,
            seed: config.seed,
            shape: config.shape,
            target: config.target,
            work: WorkReport {
                algorithmic_bytes_per_operation: config.work.algorithmic_bytes(),
                arithmetic_intensity_flops_per_byte: config.work.arithmetic_intensity(),
                flops_per_operation: config.work.floating_point_operations(),
                flops_convention: config.flops_convention,
            },
        };

        Ok(report)
    }

    fn benchmark_attributes<B: OperationTimer>(
        config: &BenchmarkConfig,
    ) -> Result<[KeyValue; 7], BenchmarkError> {
        let operation: &'static str = config.operation.into();
        let backend: &'static str = config.target.into();
        let clock: &'static str = B::CLOCK.into();
        let dtype: &'static str = config.dtype.into();

        Ok([
            KeyValue::new(ATTRIBUTE_OPERATION, operation),
            KeyValue::new(ATTRIBUTE_BACKEND, backend),
            KeyValue::new(ATTRIBUTE_CLOCK, clock),
            KeyValue::new(ATTRIBUTE_DTYPE, dtype),
            KeyValue::new(
                ATTRIBUTE_M,
                Self::metric_dimension(ATTRIBUTE_M, config.shape.m)?,
            ),
            KeyValue::new(
                ATTRIBUTE_N,
                Self::metric_dimension(ATTRIBUTE_N, config.shape.n)?,
            ),
            KeyValue::new(
                ATTRIBUTE_K,
                Self::metric_dimension(ATTRIBUTE_K, config.shape.k)?,
            ),
        ])
    }

    fn metric_dimension(name: &'static str, value: usize) -> Result<i64, BenchmarkError> {
        i64::try_from(value).map_err(|_| BenchmarkError::InvalidDimension { name, value })
    }
}

struct OperationHistograms {
    duration_microseconds: opentelemetry::metrics::Histogram<f64>,
    flops_per_second: opentelemetry::metrics::Histogram<f64>,
    algorithmic_bytes_per_second: opentelemetry::metrics::Histogram<f64>,
    run_duration_microseconds: opentelemetry::metrics::Histogram<f64>,
    aggregate_flops_per_second: opentelemetry::metrics::Histogram<f64>,
    aggregate_operations_per_second: opentelemetry::metrics::Histogram<f64>,
    runs: opentelemetry::metrics::Counter<u64>,
    operations: opentelemetry::metrics::Counter<u64>,
    samples: opentelemetry::metrics::Counter<u64>,
}

impl OperationHistograms {
    fn new(meter: &opentelemetry::metrics::Meter) -> Self {
        Self {
            duration_microseconds: meter
                .f64_histogram("psyduck_operation_duration_microseconds")
                .with_unit("us")
                .with_boundaries(duration_boundaries())
                .build(),
            flops_per_second: meter
                .f64_histogram("psyduck_operation_flops_per_second")
                .with_unit("FLOP/s")
                .with_boundaries(rate_boundaries())
                .build(),
            algorithmic_bytes_per_second: meter
                .f64_histogram("psyduck_operation_algorithmic_bytes_per_second")
                .with_unit("By/s")
                .with_boundaries(rate_boundaries())
                .build(),
            run_duration_microseconds: meter
                .f64_histogram("psyduck_benchmark_run_duration_microseconds")
                .with_unit("us")
                .with_boundaries(duration_boundaries())
                .build(),
            aggregate_flops_per_second: meter
                .f64_histogram("psyduck_benchmark_aggregate_flops_per_second")
                .with_unit("FLOP/s")
                .with_boundaries(rate_boundaries())
                .build(),
            aggregate_operations_per_second: meter
                .f64_histogram("psyduck_benchmark_aggregate_operations_per_second")
                .with_unit("{operation}/s")
                .with_boundaries(rate_boundaries())
                .build(),
            runs: meter
                .u64_counter("psyduck_benchmark_runs")
                .with_unit("{run}")
                .build(),
            operations: meter
                .u64_counter("psyduck_benchmark_operations")
                .with_unit("{operation}")
                .build(),
            samples: meter
                .u64_counter("psyduck_benchmark_samples")
                .with_unit("{sample}")
                .build(),
        }
    }

    fn record<T>(
        &self,
        metrics: &OperationMetrics<T>,
        work: OperationWork,
        attributes: &[KeyValue],
    ) {
        for metric in metrics {
            let elapsed_seconds = metric.elapsed().as_secs_f64();
            self.duration_microseconds
                .record(elapsed_seconds * 1_000_000.0, attributes);

            if elapsed_seconds > 0.0 {
                self.flops_per_second
                    .record(work.flops_per_second(elapsed_seconds), attributes);
                if let Some(bytes_per_second) = work.bytes_per_second(elapsed_seconds) {
                    self.algorithmic_bytes_per_second
                        .record(bytes_per_second, attributes);
                }
            }
        }

        let total_seconds = metrics.total_elapsed().as_secs_f64();
        self.run_duration_microseconds
            .record(total_seconds * 1_000_000.0, attributes);
        if total_seconds > 0.0 {
            self.aggregate_flops_per_second.record(
                work.floating_point_operations() as f64 * metrics.operation_count() as f64
                    / total_seconds,
                attributes,
            );
            self.aggregate_operations_per_second
                .record(metrics.operation_count() as f64 / total_seconds, attributes);
        }
        self.runs.add(1, attributes);
        self.operations
            .add(metrics.operation_count() as u64, attributes);
        self.samples.add(metrics.sample_count() as u64, attributes);
    }
}

fn duration_boundaries() -> Vec<f64> {
    vec![
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        25.0,
        50.0,
        100.0,
        250.0,
        500.0,
        1_000.0,
        2_500.0,
        5_000.0,
        10_000.0,
        25_000.0,
        50_000.0,
        100_000.0,
        250_000.0,
        500_000.0,
        1_000_000.0,
        2_500_000.0,
        5_000_000.0,
        10_000_000.0,
        25_000_000.0,
        50_000_000.0,
        100_000_000.0,
    ]
}

fn rate_boundaries() -> Vec<f64> {
    vec![
        1.0,
        10.0,
        100.0,
        1_000.0,
        10_000.0,
        100_000.0,
        1_000_000.0,
        10_000_000.0,
        100_000_000.0,
        1_000_000_000.0,
        10_000_000_000.0,
        100_000_000_000.0,
        1_000_000_000_000.0,
        10_000_000_000_000.0,
        100_000_000_000_000.0,
    ]
}

#[cfg(test)]
mod tests {
    use std::{
        cell::Cell, convert::Infallible, mem::size_of, num::NonZeroUsize, rc::Rc, time::Duration,
    };

    use crate::operation::{CaptureConfig, OperationTimer, TimingClock};

    use super::{
        Benchmark, BenchmarkConfig, BenchmarkEnvironment, BenchmarkSchemaVersion, BenchmarkShape,
        BenchmarkTarget, BenchmarkedOperation, DataType, Distribution, FlopsConvention,
        GitMetadata, HostMetadata, OperationKind, OperationWork,
    };

    struct FakeTimer {
        marker: Cell<u64>,
    }

    impl OperationTimer for FakeTimer {
        type Error = Infallible;
        type Marker = u64;

        const CLOCK: TimingClock = TimingClock::HostWall;

        fn mark(&self) -> Result<Self::Marker, Self::Error> {
            let marker = self.marker.get();
            self.marker.set(marker + 1);
            Ok(marker)
        }

        fn elapsed(&self, start: Self::Marker, end: Self::Marker) -> Result<Duration, Self::Error> {
            Ok(Duration::from_micros(end - start))
        }

        fn synchronize(&self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    struct CountingOperation {
        executions: Rc<Cell<usize>>,
    }

    impl BenchmarkedOperation<FakeTimer> for CountingOperation {
        type Error = Infallible;

        fn execute(&mut self, _backend: &FakeTimer) -> Result<(), Self::Error> {
            self.executions.set(self.executions.get() + 1);
            Ok(())
        }

        fn output_sum(&self, _backend: &FakeTimer) -> Result<f64, Self::Error> {
            Ok(42.0)
        }
    }

    fn config() -> BenchmarkConfig {
        BenchmarkConfig {
            operation: OperationKind::Matmul,
            target: BenchmarkTarget::Host,
            dtype: DataType::F32,
            shape: BenchmarkShape { m: 4, n: 4, k: 4 },
            capture: CaptureConfig::new(
                1,
                NonZeroUsize::new(3).unwrap(),
                NonZeroUsize::new(1).unwrap(),
            ),
            work: OperationWork::matmul(4, 4, 4, size_of::<f32>()),
            flops_convention: FlopsConvention::MatmulMultiplyAddAsTwo,
            generator: "deterministic",
            seed: 7,
            expected_output_sum: Some(42.0),
            environment: BenchmarkEnvironment {
                git: GitMetadata {
                    commit: Some("abc123".to_owned()),
                    dirty: Some(false),
                },
                host: HostMetadata {
                    architecture: "test-arch",
                    cpu_model: Some("test-cpu".to_owned()),
                    hostname: Some("test-host".to_owned()),
                    logical_cpus: 8,
                    operating_system: "test-os",
                },
            },
        }
    }

    #[test]
    fn calculates_interpolated_percentiles() {
        let distribution = Distribution::from_samples([1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("non-empty samples should produce a distribution");

        assert_eq!(distribution.min, 1.0);
        assert_eq!(distribution.p50, 3.0);
        assert_eq!(distribution.p100, 5.0);
    }

    #[test]
    fn returns_a_typed_report() {
        let executions = Rc::new(Cell::new(0));
        let report = Benchmark::run(
            config(),
            &FakeTimer {
                marker: Cell::new(0),
            },
            CountingOperation {
                executions: executions.clone(),
            },
        )
        .unwrap();

        assert_eq!(executions.get(), 4);
        assert_eq!(report.schema_version, BenchmarkSchemaVersion::V1);
        assert_eq!(serde_json::to_value(&report).unwrap()["schema_version"], 1);
        assert_eq!(report.operation, OperationKind::Matmul);
        assert_eq!(report.environment.git.commit.as_deref(), Some("abc123"));
        assert_eq!(report.environment.host.logical_cpus, 8);
        assert_eq!(report.capture.sample_count, 3);
        assert_eq!(report.latency_us.min, 1.0);
        assert_eq!(report.latency_us.p100, 1.0);
        assert_eq!(report.output_sum, 42.0);
        assert!(report.correctness.unwrap().passed);

        let mut encoded = Vec::new();
        report.write_to(&mut encoded).unwrap();
        assert!(encoded.ends_with(b"\n"));
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&encoded).unwrap()["output_sum"],
            42.0
        );
    }
}
