use std::{error::Error, process::ExitCode};

use cudarc::driver::CudaContext;
use model::{cuda::CudaModelBackend, model::ModelBackend, naive::NaiveModelBackend};
use tensor::{NaiveTensor, Shape};
use tracing::warn;

const SHAPE: Shape<2> = Shape::new([2, 2]);

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    match run() {
        Ok((expected, actual)) if actual == expected => {
            warn!(?expected, ?actual, "GPU add MVP succeeded");
            ExitCode::SUCCESS
        }
        Ok((expected, actual)) => {
            warn!(?expected, ?actual, "GPU add MVP failed: output mismatch");
            ExitCode::FAILURE
        }
        Err(error) => {
            warn!(error = %error, "GPU add MVP failed");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
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

    cuda_backend.try_add(&cuda_a, &cuda_b, &mut cuda_target)?;

    let actual = cuda_backend.download(&cuda_target)?;

    Ok((naive_target.as_slice().to_vec(), actual.as_slice().to_vec()))
}

fn matrix(values: [f32; 4]) -> Result<NaiveTensor<f32, 2>, Box<dyn Error>> {
    Ok(NaiveTensor::from_vec(values.to_vec(), SHAPE)?)
}
