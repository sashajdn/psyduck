use std::{error::Error, fs, mem::size_of, num::NonZeroUsize, path::PathBuf, process::Command};

use clap::{Parser, Subcommand, ValueEnum};
use generator::{Generator, Seed, canonical_matmul_checksum};
use instrument::{
    benchmark::{
        Benchmark, BenchmarkConfig, BenchmarkEnvironment, BenchmarkReport, BenchmarkShape,
        BenchmarkTarget, BenchmarkedOperation, DataType, GitMetadata, HostMetadata, OperationKind,
        OperationWork,
    },
    operation::{CaptureConfig, DEFAULT_WARMUP_OPERATIONS},
};
use model::{
    host::HostModelBackend,
    model::{ModelBackend, ModelError},
};
use tensor::{HostTensor, Shape};

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

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Target {
    Host,
    Device,
}

impl From<Target> for BenchmarkTarget {
    fn from(target: Target) -> Self {
        match target {
            Target::Host => Self::Host,
            Target::Device => Self::Device,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MatrixSize {
    #[value(name = "4")]
    N4,
    #[value(name = "8")]
    N8,
    #[value(name = "16")]
    N16,
    #[value(name = "32")]
    N32,
    #[value(name = "64")]
    N64,
    #[value(name = "128")]
    N128,
    #[value(name = "256")]
    N256,
    #[value(name = "512")]
    N512,
    #[value(name = "1024")]
    N1024,
    #[value(name = "2048")]
    N2048,
    #[value(name = "4096")]
    N4096,
}

impl MatrixSize {
    const fn get(self) -> usize {
        match self {
            Self::N4 => 4,
            Self::N8 => 8,
            Self::N16 => 16,
            Self::N32 => 32,
            Self::N64 => 64,
            Self::N128 => 128,
            Self::N256 => 256,
            Self::N512 => 512,
            Self::N1024 => 1024,
            Self::N2048 => 2048,
            Self::N4096 => 4096,
        }
    }
}

#[derive(Debug, Subcommand)]
enum GeneratorArgs {
    /// Generate reproducible inputs, optionally overriding the canonical seed.
    Deterministic {
        #[arg(long)]
        seed: Option<Seed>,
    },
    /// Choose a random seed, which is printed so the run can be reproduced.
    NonDeterministic,
}

impl GeneratorArgs {
    fn into_generator(args: Option<Self>) -> Generator {
        match args {
            None | Some(GeneratorArgs::Deterministic { seed: None }) => Generator::default(),
            Some(GeneratorArgs::Deterministic { seed: Some(seed) }) => {
                Generator::Deterministic(Some(seed))
            }
            Some(GeneratorArgs::NonDeterministic) => Generator::NonDeterministic,
        }
    }
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
    if matches!(args.target, Target::Device) {
        panic!("device matmul is not implemented yet");
    }

    let report_dir = args.report_dir;

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

    // Setup the backand & allocate C.
    let backend = HostModelBackend::<f32>::new();
    let mut output = backend
        .alloc(Shape::new([m, n]))
        .expect("output tensor should be allocated");

    let report = Benchmark::run(
        BenchmarkConfig {
            operation: OperationKind::Matmul,
            target: args.target.into(),
            dtype: DataType::F32,
            shape: BenchmarkShape { m, n, k },
            capture: CaptureConfig::new(args.warmup, args.operations, args.sample_every),
            work: OperationWork::matmul(m, n, k, size_of::<f32>()),
            flops_convention: "2*M*N*K",
            generator: generator.name(),
            seed: generated.seed,
            expected_output_sum,
            environment: benchmark_environment(),
        },
        &backend,
        MatmulOperation {
            a: &generated.a,
            b: &generated.b,
            output: &mut output,
        },
    )?;

    let encoded = serde_json::to_string(&report)?;
    println!("{encoded}");

    if let Some(report_dir) = report_dir {
        fs::create_dir_all(&report_dir)?;
        let operation: &'static str = report.operation.into();
        let target: &'static str = report.target.into();
        let report_path = report_dir.join(format!(
            "{operation}-{target}-m{}-n{}-k{}.json",
            report.shape.m, report.shape.n, report.shape.k
        ));
        fs::write(&report_path, format!("{encoded}\n"))?;
        tracing::info!(report_path = %report_path.display(), "benchmark report written");
    }

    if report.correctness.is_some_and(|checksum| !checksum.passed) {
        return Err(
            std::io::Error::other("matmul output checksum did not match canonical value").into(),
        );
    }

    Ok(report)
}

struct MatmulOperation<'a> {
    a: &'a HostTensor<f32, 2>,
    b: &'a HostTensor<f32, 2>,
    output: &'a mut HostTensor<f32, 2>,
}

impl BenchmarkedOperation<HostModelBackend<f32>> for MatmulOperation<'_> {
    type Error = ModelError;

    fn execute(&mut self, backend: &HostModelBackend<f32>) -> Result<(), Self::Error> {
        backend.try_matmul(self.a, self.b, self.output)
    }

    fn output_sum(&self) -> f64 {
        self.output.as_slice().iter().copied().map(f64::from).sum()
    }
}

fn benchmark_environment() -> BenchmarkEnvironment {
    BenchmarkEnvironment {
        git: GitMetadata {
            commit: environment_value("PSYDUCK_GIT_COMMIT").or_else(git_commit),
            dirty: environment_value("PSYDUCK_GIT_DIRTY")
                .and_then(|value| value.parse().ok())
                .or_else(git_dirty),
        },
        host: HostMetadata {
            architecture: std::env::consts::ARCH,
            cpu_model: cpu_model(),
            hostname: environment_value("HOSTNAME").or_else(hostname),
            logical_cpus: std::thread::available_parallelism().map_or(1, NonZeroUsize::get),
            operating_system: std::env::consts::OS,
        },
    }
}

fn environment_value(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn git_commit() -> Option<String> {
    command_output_in_repository("git", &["rev-parse", "HEAD"])
}

fn git_dirty() -> Option<bool> {
    command_output_in_repository("git", &["status", "--porcelain"]).map(|status| !status.is_empty())
}

fn hostname() -> Option<String> {
    command_output("hostname", &[])
}

#[cfg(target_os = "macos")]
fn cpu_model() -> Option<String> {
    command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
        .or_else(|| command_output("sysctl", &["-n", "hw.model"]))
}

#[cfg(target_os = "linux")]
fn cpu_model() -> Option<String> {
    fs::read_to_string("/proc/cpuinfo")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("model name\t:"))
        .map(str::trim)
        .map(str::to_owned)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn cpu_model() -> Option<String> {
    None
}

fn command_output_in_repository(program: &str, arguments: &[&str]) -> Option<String> {
    let output = Command::new(program)
        .args(arguments)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    successful_output(output)
}

fn command_output(program: &str, arguments: &[&str]) -> Option<String> {
    successful_output(Command::new(program).args(arguments).output().ok()?)
}

fn successful_output(output: std::process::Output) -> Option<String> {
    output
        .status
        .success()
        .then(|| String::from_utf8(output.stdout).ok())
        .flatten()
        .map(|value| value.trim().to_owned())
}
