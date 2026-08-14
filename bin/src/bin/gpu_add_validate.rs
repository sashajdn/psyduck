use std::{error::Error, process::ExitCode, time::Duration};

use cudarc::driver::CudaContext;
use instrument::cuda::CudaKernelMetrics;
use model::{cuda::CudaModelBackend, model::ModelBackend, naive::NaiveModelBackend};
use serde_json::json;
use tensor::{NaiveTensor, Shape};
use tracing::warn;

const SHAPE: Shape<2> = Shape::new([2, 2]);
const ADD_THREADS_PER_BLOCK: u32 = 256;

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

    let naive_backend = NaiveModelBackend::<f32>::new();
    let mut naive_target = naive_backend.alloc(SHAPE)?;
    naive_backend.try_add::<ADD_THREADS_PER_BLOCK>(&a, &b, &mut naive_target)?;

    let context = CudaContext::new(0)?;
    let cuda_backend = CudaModelBackend::<f32>::new(context)?;
    let cuda_a = cuda_backend.upload_htod(&a)?;
    let cuda_b = cuda_backend.upload_htod(&b)?;
    let mut cuda_c = cuda_backend.alloc(SHAPE)?;

    let metrics = CudaKernelMetrics::instrument(&cuda_backend, |backend| {
        backend.try_add::<ADD_THREADS_PER_BLOCK>(&cuda_a, &cuda_b, &mut cuda_c)
    })?;

    let gpu_elapsed = metrics
        .iter()
        .next()
        .expect("capture records one kernel metric")
        .elapsed();

    let result = cuda_backend.download_dtoh(&cuda_c)?;

    Ok(ValidationResult {
        expected: naive_target.as_slice().to_vec(),
        actual: result.as_slice().to_vec(),
        gpu_elapsed,
    })
}

fn matrix(values: [f32; 4]) -> Result<NaiveTensor<f32, 2>, Box<dyn Error>> {
    Ok(NaiveTensor::from_vec(values.to_vec(), SHAPE)?)
}
