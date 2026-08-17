use clap::{Subcommand, ValueEnum};
use generator::{Generator, Seed};
use instrument::benchmark::BenchmarkTarget;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Target {
    /// Run the benchmark on the host CPU.
    Host,
    /// Run the benchmark on the device GPU.
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
pub enum MatrixSize {
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
    pub const fn get(self) -> usize {
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
pub enum GeneratorArgs {
    /// Generate reproducible inputs, optionally overriding the canonical seed.
    Deterministic {
        #[arg(long)]
        seed: Option<Seed>,
    },
    /// Choose a random seed, which is printed so the run can be reproduced.
    NonDeterministic,
}

impl GeneratorArgs {
    pub fn into_generator(args: Option<Self>) -> Generator {
        match args {
            None | Some(Self::Deterministic { seed: None }) => Generator::default(),
            Some(Self::Deterministic { seed: Some(seed) }) => Generator::Deterministic(Some(seed)),
            Some(Self::NonDeterministic) => Generator::NonDeterministic,
        }
    }
}
