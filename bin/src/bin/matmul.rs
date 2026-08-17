use std::{error::Error, io, mem::size_of, num::NonZeroUsize, path::PathBuf};

use clap::Parser;
use cudarc::driver::CudaContext;
use generator::canonical_matmul_checksum;
use instrument::{
    benchmark::{
        Benchmark, BenchmarkConfig, BenchmarkReport, BenchmarkShape, BenchmarkedOperation,
        DataType, FlopsConvention, OperationKind, OperationWork,
    },
    operation::{CaptureConfig, DEFAULT_WARMUP_OPERATIONS, OperationTimer},
};
use model::{
    cuda::CudaModelBackend,
    host::HostModelBackend,
    model::{ModelBackend, ModelError},
};
use psyduck::benchmark::{
    args::{GeneratorArgs, MatrixSize, Target},
    benchmark_environment, write_report_file,
};
use tensor::Shape;

#[derive(Debug, Parser)]
#[command(about = "Run a matrix multiplication")]
struct MatmulArgs {
    /// Execution target.
    #[arg(long, value_enum, default_value_t = Target::Host)]
    target: Target,

    /// Rows in A and C.
    #[arg(long, value_enum, default_value_t = MatrixSize::N512)]
    m: MatrixSize,

    /// Columns in B and C. Defaults to M.
    #[arg(long, value_enum)]
    n: Option<MatrixSize>,

    /// Columns in A and rows in B. Defaults to M.
    #[arg(long, value_enum)]
    k: Option<MatrixSize>,

    /// Directory in which to write a JSON report.
    #[arg(long)]
    report_dir: Option<PathBuf>,

    /// Number of untimed operations performed before capture.
    #[arg(long, default_value_t = DEFAULT_WARMUP_OPERATIONS)]
    warmup: usize,

    /// Total number of measured operations.
    #[arg(long, default_value_t = NonZeroUsize::new(1).expect("one is non-zero"))]
    operations: NonZeroUsize,

    /// Capture one timing sample for every N operations.
    #[arg(long, default_value_t = NonZeroUsize::new(1).expect("one is non-zero"))]
    sample_every: NonZeroUsize,

    /// Input generation mode. Defaults to the canonical deterministic seed.
    #[command(subcommand)]
    generator: Option<GeneratorArgs>,
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .init();

    run(MatmulArgs::parse()).expect("matmul benchmark should succeed");
}

fn run(args: MatmulArgs) -> Result<BenchmarkReport, Box<dyn Error>> {
    let report_dir = args.report_dir.clone();

    let report = match args.target {
        Target::Host => run_with_backend(args, HostModelBackend::<f32>::new())?,
        Target::Device => {
            let context = CudaContext::new(0)?;
            run_with_backend(args, CudaModelBackend::<f32>::new(context)?)?
        }
    };

    report.write_to(io::stdout().lock())?;
    if let Some(report_dir) = report_dir.as_deref() {
        write_report_file(&report, report_dir)?;
    }

    if report.correctness.is_some_and(|checksum| !checksum.passed) {
        return Err(
            std::io::Error::other("matmul output checksum did not match canonical value").into(),
        );
    }

    Ok(report)
}

fn run_with_backend<B>(args: MatmulArgs, backend: B) -> Result<BenchmarkReport, Box<dyn Error>>
where
    B: ModelBackend<f32> + OperationTimer<Error = ModelError>,
{
    // Generate the input matrices A and B.
    let m = args.m.get();
    let n = args.n.unwrap_or(args.m).get();
    let k = args.k.unwrap_or(args.m).get();

    let generator = GeneratorArgs::into_generator(args.generator);
    let expected_output_sum = (generator.is_canonical() && m == n && n == k)
        .then(|| canonical_matmul_checksum(m))
        .flatten();

    let generated = generator
        .matrices::<f32>(m, n, k)
        .expect("matrix inputs should be generated");

    // Transfer inputs and allocate C before timing begins.
    let a = backend.upload(&generated.a)?;
    let b = backend.upload(&generated.b)?;
    let mut output = backend.alloc(Shape::new([m, n]))?;

    // Run the benchmark.
    Benchmark::run(
        BenchmarkConfig {
            operation: OperationKind::Matmul,
            target: args.target.into(),
            dtype: DataType::F32,
            shape: BenchmarkShape { m, n, k },
            capture: CaptureConfig::new(args.warmup, args.operations, args.sample_every),
            work: OperationWork::matmul(m, n, k, size_of::<f32>()),
            flops_convention: FlopsConvention::MatmulMultiplyAddAsTwo,
            generator: generator.name(),
            seed: generated.seed,
            expected_output_sum,
            environment: benchmark_environment(),
        },
        &backend,
        MatmulOperation {
            a: &a,
            b: &b,
            output: &mut output,
        },
    )
    .map_err(Into::into)
}

struct MatmulOperation<'a, B: ModelBackend<f32>> {
    a: &'a B::Tensor<2>,
    b: &'a B::Tensor<2>,
    output: &'a mut B::Tensor<2>,
}

impl<B> BenchmarkedOperation<B> for MatmulOperation<'_, B>
where
    B: ModelBackend<f32>,
{
    type Error = ModelError;

    fn execute(&mut self, backend: &B) -> Result<(), Self::Error> {
        // TODO: Add a backend zero/fill operation and reset C before every
        // iteration when benchmarking matmul kernels that accumulate into the
        // existing target instead of overwriting it.
        backend.try_matmul(self.a, self.b, self.output)
    }

    fn output_sum(&self, backend: &B) -> Result<f64, Self::Error> {
        Ok(backend
            .download(self.output)?
            .as_slice()
            .iter()
            .copied()
            .map(f64::from)
            .sum())
    }
}
