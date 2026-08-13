use std::{error::Error, process::ExitCode, time::Duration};

use cudarc::driver::CudaContext;
use instrument::cuda::CudaKernelMetrics;
use model::{cuda::CudaModelBackend, model::ModelBackend, naive::NaiveModelBackend};
use tensor::{NaiveTensor, Shape};
use tracing::warn;

const SHAPE: Shape<2> = Shape::new([2, 2]);

struct Validation {
    expected: Vec<f32>,
    actual: Vec<f32>,
    gpu_elapsed: Duration,
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    match run() {
        Ok(validation) if validation.actual == validation.expected => {
            warn!(
                expected = ?validation.expected,
                actual = ?validation.actual,
                gpu_elapsed_us = validation.gpu_elapsed.as_secs_f64() * 1_000_000.0,
                "GPU add validation succeeded"
            );
            ExitCode::SUCCESS
        }
        Ok(validation) => {
            warn!(
                expected = ?validation.expected,
                actual = ?validation.actual,
                gpu_elapsed_us = validation.gpu_elapsed.as_secs_f64() * 1_000_000.0,
                "GPU add validation failed: output mismatch"
            );
            ExitCode::FAILURE
        }
        Err(error) => {
            warn!(error = %error, "GPU add validation failed");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<Validation, Box<dyn Error>> {
    let a = matrix([1.0, 2.0, 3.0, 4.0])?;
    let b = matrix([5.0, 6.0, 7.0, 8.0])?;

    let naive_backend = NaiveModelBackend::<f32>::new();
    let mut naive_target = naive_backend.alloc(SHAPE)?;
    naive_backend.try_add(&a, &b, &mut naive_target)?;

    let context = CudaContext::new(0)?;
    let cuda_backend = CudaModelBackend::<f32>::new(context)?;
    let cuda_a = cuda_backend.upload(&a)?;
    let cuda_b = cuda_backend.upload(&b)?;
    let mut cuda_target = cuda_backend.alloc(SHAPE)?;

    let metrics = CudaKernelMetrics::instrument(&cuda_backend, |backend| {
        backend.try_add(&cuda_a, &cuda_b, &mut cuda_target)
    })?;

    let gpu_elapsed = metrics
        .iter()
        .next()
        .expect("capture records one kernel metric")
        .elapsed();

    let actual = cuda_backend.download(&cuda_target)?;

    Ok(Validation {
        expected: naive_target.as_slice().to_vec(),
        actual: actual.as_slice().to_vec(),
        gpu_elapsed,
    })
}

fn matrix(values: [f32; 4]) -> Result<NaiveTensor<f32, 2>, Box<dyn Error>> {
    Ok(NaiveTensor::from_vec(values.to_vec(), SHAPE)?)
}
