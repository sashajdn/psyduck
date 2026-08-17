mod cuda;
mod host;
mod shape;
mod tensor;

pub use cuda::{CudaBuffer, CudaTensor};
pub use host::HostTensor;
pub use shape::Shape;
pub use tensor::{ElementCountMismatch, MatrixTensor, QuantizedFp, ShapeMismatch, Tensor};

#[derive(thiserror::Error, Debug)]
pub enum MatrixError {
    #[error("Matrix index out of bounds: {0:?}")]
    OutOfBounds(Shape<2>),
}
