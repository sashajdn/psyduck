mod shape;
mod tensor;

pub use shape::Shape;
pub use tensor::{MatrixTensor, NaiveTensor, QuantizedFp, ShapeMismatch, Tensor};
