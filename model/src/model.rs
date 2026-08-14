use tensor::{NaiveTensor, QuantizedFp, Shape, Tensor};

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
}

pub trait ModelBackend<F: QuantizedFp> {
    type Tensor<const R: usize>: Tensor<F, R>;

    /// Uploads a tensor from the host to the device.
    fn upload_htod<const R: usize>(
        &self,
        source: &NaiveTensor<F, R>,
    ) -> Result<Self::Tensor<R>, ModelError>;

    /// Downloads a tensor from the device to the host.
    fn download_dtoh<const R: usize>(
        &self,
        source: &Self::Tensor<R>,
    ) -> Result<NaiveTensor<F, R>, ModelError>;

    /// Allocates a tensor on the device with the given shape.
    fn alloc<const R: usize>(&self, shape: Shape<R>) -> Result<Self::Tensor<R>, ModelError>;

    /// Performs matrix multiplication of two rank-2 tensors and stores the result in the target tensor.
    fn try_matmul<const BLOCK_X: u32, const BLOCK_Y: u32, const THREADS_PER_BLOCK: u32>(
        &self,
        a: &Self::Tensor<2>,
        b: &Self::Tensor<2>,
        target: &mut Self::Tensor<2>,
    ) -> Result<(), ModelError>;

    /// Performs element-wise addition of two rank-2 tensors and stores the result in the target tensor.
    fn try_add<const THREADS_PER_BLOCK: u32>(
        &self,
        a: &Self::Tensor<2>,
        b: &Self::Tensor<2>,
        target: &mut Self::Tensor<2>,
    ) -> Result<(), ModelError>;
}
