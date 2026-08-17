use std::{
    error::Error,
    fs::{self, File},
    io::BufWriter,
    num::NonZeroUsize,
    path::Path,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use instrument::benchmark::{BenchmarkEnvironment, BenchmarkReport, GitMetadata, HostMetadata};

pub mod args;

pub fn write_report_file(
    report: &BenchmarkReport,
    report_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(report_dir)?;
    let operation: &'static str = report.operation.into();
    let target: &'static str = report.target.into();
    let commit = report
        .environment
        .git
        .commit
        .as_deref()
        .map(short_commit)
        .unwrap_or("unknown");
    let timestamp_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let report_path = report_dir.join(format!(
        "{operation}-{target}-m{}-n{}-k{}-{commit}-{timestamp_ms}.json",
        report.shape.m, report.shape.n, report.shape.k
    ));

    report.write_to(BufWriter::new(File::create(&report_path)?))?;
    tracing::info!(report_path = %report_path.display(), "benchmark report written");

    Ok(())
}

pub fn benchmark_environment() -> BenchmarkEnvironment {
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

fn short_commit(commit: &str) -> &str {
    commit.get(..12).unwrap_or(commit)
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
