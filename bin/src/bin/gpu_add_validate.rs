use std::{error::Error, num::NonZeroUsize, process::ExitCode, time::Duration};

use cudarc::driver::CudaContext;
use instrument::operation::{CaptureConfig, OperationMetrics};
use model::{cuda::CudaModelBackend, host::HostModelBackend, model::ModelBackend};
use serde_json::json;
use tensor::{HostTensor, Shape};
use tracing::warn;

const SHAPE: Shape<2> = Shape::new([2, 2]);
struct ValidationResult {
    expected: Vec<f32>,
    actual: Vec<f32>,
    gpu_elapsed: Duration,
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .init();

    match run() {
        Ok(result) if result.actual == result.expected => {
            println!(
                "{}",
                json!({
                    "actual": &result.actual,
                    "expected": &result.expected,
                    "gpu_elapsed_us": result.gpu_elapsed.as_secs_f64() * 1_000_000.0,
                    "status": "passed",
                })
            );
            warn!(
                expected = ?result.expected,
                actual = ?result.actual,
                gpu_elapsed_us = result.gpu_elapsed.as_secs_f64() * 1_000_000.0,
                "GPU add validation succeeded"
            );
            ExitCode::SUCCESS
        }
        Ok(result) => {
            println!(
                "{}",
                json!({
                    "actual": &result.actual,
                    "expected": &result.expected,
                    "gpu_elapsed_us": result.gpu_elapsed.as_secs_f64() * 1_000_000.0,
                    "status": "failed",
                })
            );
            warn!(
                expected = ?result.expected,
                actual = ?result.actual,
                gpu_elapsed_us = result.gpu_elapsed.as_secs_f64() * 1_000_000.0,
                "GPU add validation failed: output mismatch"
            );
            ExitCode::FAILURE
        }
        Err(error) => {
            println!(
                "{}",
                json!({ "error": error.to_string(), "status": "error" })
            );
            warn!(error = %error, "GPU add validation failed");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ValidationResult, Box<dyn Error>> {
    let a = matrix([1.0, 2.0, 3.0, 4.0])?;
    let b = matrix([5.0, 6.0, 7.0, 8.0])?;

    let naive_backend = HostModelBackend::<f32>::new();
    let mut naive_target = naive_backend.alloc(SHAPE)?;
    naive_backend.try_add(&a, &b, &mut naive_target)?;

    let context = CudaContext::new(0)?;
    let cuda_backend = CudaModelBackend::<f32>::new(context)?;
    let cuda_a = cuda_backend.upload(&a)?;
    let cuda_b = cuda_backend.upload(&b)?;
    let mut cuda_c = cuda_backend.alloc(SHAPE)?;

    let metrics = OperationMetrics::capture(
        &cuda_backend,
        CaptureConfig::new(
            0,
            NonZeroUsize::new(1).expect("one operation is non-zero"),
            NonZeroUsize::new(1).expect("one sample is non-zero"),
        ),
        |backend| backend.try_add(&cuda_a, &cuda_b, &mut cuda_c),
    )?;

    let gpu_elapsed = metrics
        .iter()
        .next()
        .expect("capture records one operation metric")
        .elapsed();

    let result = cuda_backend.download(&cuda_c)?;

    Ok(ValidationResult {
        expected: naive_target.as_slice().to_vec(),
        actual: result.as_slice().to_vec(),
        gpu_elapsed,
    })
}

fn matrix(values: [f32; 4]) -> Result<HostTensor<f32, 2>, Box<dyn Error>> {
    Ok(HostTensor::from_vec(values.to_vec(), SHAPE)?)
}
