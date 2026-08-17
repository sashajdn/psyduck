use std::{num::NonZeroUsize, slice, time::Duration};

use serde::Serialize;
use strum::IntoStaticStr;

pub const DEFAULT_WARMUP_OPERATIONS: usize = 3;

#[derive(Clone, Copy, Debug, Eq, IntoStaticStr, PartialEq, Serialize)]
#[strum(serialize_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum TimingClock {
    HostWall,
    CudaStream,
}

pub trait OperationTimer {
    type Error;
    type Marker;

    const CLOCK: TimingClock;

    fn mark(&self) -> Result<Self::Marker, Self::Error>;

    fn elapsed(&self, start: Self::Marker, end: Self::Marker) -> Result<Duration, Self::Error>;

    fn synchronize(&self) -> Result<(), Self::Error>;
}

#[derive(Clone, Copy, Debug)]
pub struct CaptureConfig {
    warmup_operations: usize,
    operation_count: NonZeroUsize,
    sample_every: NonZeroUsize,
}

impl CaptureConfig {
    #[inline]
    pub const fn new(
        warmup_operations: usize,
        operation_count: NonZeroUsize,
        sample_every: NonZeroUsize,
    ) -> Self {
        Self {
            warmup_operations,
            operation_count,
            sample_every,
        }
    }

    #[inline]
    pub const fn warmup_operations(self) -> usize {
        self.warmup_operations
    }

    #[inline]
    pub const fn operation_count(self) -> usize {
        self.operation_count.get()
    }

    #[inline]
    pub const fn sample_every(self) -> usize {
        self.sample_every.get()
    }
}

#[derive(Debug)]
pub struct OperationMetric<T> {
    result: T,
    elapsed: Duration,
}

impl<T> OperationMetric<T> {
    #[inline]
    pub fn result(&self) -> &T {
        &self.result
    }

    #[inline]
    pub const fn elapsed(&self) -> Duration {
        self.elapsed
    }

    #[inline]
    pub fn into_result(self) -> T {
        self.result
    }
}

#[derive(Debug)]
pub struct OperationMetrics<T> {
    metrics: Vec<OperationMetric<T>>,
    total_elapsed: Duration,
    config: CaptureConfig,
    clock: TimingClock,
}

impl<T> OperationMetrics<T> {
    pub fn capture<B, E, Operation>(
        backend: &B,
        config: CaptureConfig,
        mut operation: Operation,
    ) -> Result<Self, E>
    where
        B: OperationTimer,
        Operation: FnMut(&B) -> Result<T, E>,
        E: From<B::Error>,
    {
        for _ in 0..config.warmup_operations() {
            drop(operation(backend)?);
        }
        backend.synchronize()?;

        let mut metrics =
            Vec::with_capacity(config.operation_count().div_ceil(config.sample_every()));
        let total_start = std::time::Instant::now();

        for index in 0..config.operation_count() {
            if index.is_multiple_of(config.sample_every()) {
                let start = backend.mark()?;
                let result = operation(backend)?;
                let end = backend.mark()?;
                let elapsed = backend.elapsed(start, end)?;
                metrics.push(OperationMetric { result, elapsed });
            } else {
                drop(operation(backend)?);
            }
        }

        backend.synchronize()?;
        let total_elapsed = total_start.elapsed();

        Ok(Self {
            metrics,
            total_elapsed,
            config,
            clock: B::CLOCK,
        })
    }

    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, OperationMetric<T>> {
        self.metrics.iter()
    }

    #[inline]
    pub fn sample_count(&self) -> usize {
        self.metrics.len()
    }

    #[inline]
    pub const fn operation_count(&self) -> usize {
        self.config.operation_count()
    }

    #[inline]
    pub const fn warmup_operations(&self) -> usize {
        self.config.warmup_operations()
    }

    #[inline]
    pub const fn sample_every(&self) -> usize {
        self.config.sample_every()
    }

    #[inline]
    pub const fn total_elapsed(&self) -> Duration {
        self.total_elapsed
    }

    #[inline]
    pub const fn clock(&self) -> TimingClock {
        self.clock
    }
}

impl<'a, T> IntoIterator for &'a OperationMetrics<T> {
    type Item = &'a OperationMetric<T>;
    type IntoIter = slice::Iter<'a, OperationMetric<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, convert::Infallible, num::NonZeroUsize, time::Duration};

    use super::{CaptureConfig, OperationMetrics, OperationTimer, TimingClock};

    struct FakeTimer {
        marker: Cell<u64>,
        synchronizations: Cell<usize>,
    }

    impl OperationTimer for FakeTimer {
        type Error = Infallible;
        type Marker = u64;

        const CLOCK: TimingClock = TimingClock::HostWall;

        fn mark(&self) -> Result<Self::Marker, Self::Error> {
            let marker = self.marker.get();
            self.marker.set(marker + 10);
            Ok(marker)
        }

        fn elapsed(&self, start: Self::Marker, end: Self::Marker) -> Result<Duration, Self::Error> {
            Ok(Duration::from_micros(end - start))
        }

        fn synchronize(&self) -> Result<(), Self::Error> {
            self.synchronizations.set(self.synchronizations.get() + 1);
            Ok(())
        }
    }

    #[test]
    fn warms_up_and_samples_at_the_configured_rate() {
        let timer = FakeTimer {
            marker: Cell::new(0),
            synchronizations: Cell::new(0),
        };
        let operations = Cell::new(0);
        let metrics = OperationMetrics::capture(
            &timer,
            CaptureConfig::new(
                2,
                NonZeroUsize::new(5).unwrap(),
                NonZeroUsize::new(2).unwrap(),
            ),
            |_| {
                operations.set(operations.get() + 1);
                Ok::<_, Infallible>(())
            },
        )
        .unwrap();

        assert_eq!(operations.get(), 7);
        assert_eq!(metrics.operation_count(), 5);
        assert_eq!(metrics.sample_count(), 3);
        assert!(
            metrics
                .iter()
                .all(|metric| metric.elapsed() == Duration::from_micros(10))
        );
        assert_eq!(timer.synchronizations.get(), 2);
    }
}
