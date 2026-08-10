use tensor::{NaiveTensor, QuantizedFp, Shape, Tensor};

#[derive(thiserror::Error, Debug)]
pub enum ModelError {
    #[error(transparent)]
    Cuda(#[from] cudarc::driver::DriverError),
    #[error(transparent)]
    ElementCountMismatch(#[from] tensor::ElementCountMismatch),
    #[error(transparent)]
    ShapeMismatch(#[from] tensor::ShapeMismatch),
}

pub trait ModelBackend<F: QuantizedFp> {
    type Tensor<const R: usize>: Tensor<F, R>;

    fn upload<const R: usize>(
        &self,
        source: &NaiveTensor<F, R>,
    ) -> Result<Self::Tensor<R>, ModelError>;
    fn download<const R: usize>(
        &self,
        source: &Self::Tensor<R>,
    ) -> Result<NaiveTensor<F, R>, ModelError>;
    fn alloc<const R: usize>(&self, shape: Shape<R>) -> Result<Self::Tensor<R>, ModelError>;

    // Primitives.
    fn try_matmul(
        &self,
        a: &Self::Tensor<2>,
        b: &Self::Tensor<2>,
        target: &mut Self::Tensor<2>,
    ) -> Result<(), ModelError>;

    fn try_add(
        &self,
        a: &Self::Tensor<2>,
        b: &Self::Tensor<2>,
        target: &mut Self::Tensor<2>,
    ) -> Result<(), ModelError>;
}
