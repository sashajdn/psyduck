use tensor::{HostTensor, QuantizedFp, Shape, Tensor};

#[derive(thiserror::Error, Debug)]
pub enum ModelError {
    #[error(transparent)]
    Cuda(#[from] cudarc::driver::DriverError),
    #[error(transparent)]
    ElementCountMismatch(#[from] tensor::ElementCountMismatch),
    #[error(transparent)]
    ShapeMismatch(#[from] tensor::ShapeMismatch),
    #[error(transparent)]
    KernelError(#[from] kernel::KernelError),
    #[error("tensor is too large: {elements} elements")]
    TensorTooLarge { elements: usize },
    #[error("CUDA block dimensions must be non-zero, got ({x}, {y}, {z})")]
    InvalidCudaBlockDimensions { x: u32, y: u32, z: u32 },
    #[error(transparent)]
    MatrixError(#[from] tensor::MatrixError),
}

pub trait ModelBackend<F: QuantizedFp> {
    type Tensor<const R: usize>: Tensor<F, R>;

    /// Uploads a tensor from the host to the device.
    fn upload<const R: usize>(
        &self,
        source: &HostTensor<F, R>,
    ) -> Result<Self::Tensor<R>, ModelError>;

    /// Downloads a tensor from the device to the host.
    fn download<const R: usize>(
        &self,
        source: &Self::Tensor<R>,
    ) -> Result<HostTensor<F, R>, ModelError>;

    /// Allocates a tensor on the device with the given shape.
    fn alloc<const R: usize>(&self, shape: Shape<R>) -> Result<Self::Tensor<R>, ModelError>;

    /// Performs matrix multiplication of two rank-2 tensors and stores the result in the target tensor.
    fn try_matmul(
        &self,
        a: &Self::Tensor<2>,
        b: &Self::Tensor<2>,
        target: &mut Self::Tensor<2>,
    ) -> Result<(), ModelError>;

    /// Performs element-wise addition of two rank-2 tensors and stores the result in the target tensor.
    fn try_add(
        &self,
        a: &Self::Tensor<2>,
        b: &Self::Tensor<2>,
        target: &mut Self::Tensor<2>,
    ) -> Result<(), ModelError>;
}
