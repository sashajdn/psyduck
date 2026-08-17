use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaEvent, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
    sys::CUevent_flags,
};
use instrument::operation::{OperationTimer, TimingClock};
use kernel::{CudaKernels, Kernel};
use tensor::{CudaTensor, HostTensor, MatrixTensor, QuantizedFp, Tensor};

use crate::model::{ModelBackend, ModelError};

#[derive(Debug, Clone)]
pub struct CudaModelBackend<F: DeviceRepr> {
    stream: Arc<CudaStream>,
    kernels: CudaKernels,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: DeviceRepr + QuantizedFp> CudaModelBackend<F> {
    pub fn new(context: Arc<CudaContext>) -> Result<Self, ModelError> {
        let kernels = CudaKernels::load(&context, vec![kernel::Kernel::AddF32])?;
        let stream = context.default_stream();

        Ok(Self {
            stream,
            kernels,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<F: DeviceRepr> OperationTimer for CudaModelBackend<F> {
    type Error = ModelError;
    type Marker = CudaEvent;

    const CLOCK: TimingClock = TimingClock::CudaStream;

    fn mark(&self) -> Result<Self::Marker, Self::Error> {
        Ok(self
            .stream
            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?)
    }

    fn elapsed(
        &self,
        start: Self::Marker,
        end: Self::Marker,
    ) -> Result<std::time::Duration, Self::Error> {
        let milliseconds = start.elapsed_ms(&end)?;
        Ok(std::time::Duration::from_secs_f64(
            f64::from(milliseconds) / 1_000.0,
        ))
    }

    fn synchronize(&self) -> Result<(), Self::Error> {
        Ok(self.stream.synchronize()?)
    }
}

impl<F> ModelBackend<F> for CudaModelBackend<F>
where
    F: DeviceRepr + ValidAsZeroBits + QuantizedFp,
{
    type Tensor<const R: usize> = CudaTensor<F, R>;

    /// Uploads a tensor from the host to the device.
    fn upload<const R: usize>(
        &self,
        source: &HostTensor<F, R>,
    ) -> Result<Self::Tensor<R>, ModelError> {
        let buffer = self.stream.clone_htod(source.as_slice())?;
        Ok(CudaTensor::from_cuda_slice(buffer, source.shape().clone())?)
    }

    // Downloads a tensor from the device to the host.
    fn download<const R: usize>(
        &self,
        source: &Self::Tensor<R>,
    ) -> Result<HostTensor<F, R>, ModelError> {
        let values = self.stream.clone_dtoh(source.as_cuda_slice())?;
        Ok(HostTensor::from_vec(values, source.shape().clone())?)
    }

    fn alloc<const R: usize>(
        &self,
        shape: tensor::Shape<R>,
    ) -> Result<Self::Tensor<R>, ModelError> {
        let buffer = self.stream.alloc_zeros(shape.numel())?;
        Ok(CudaTensor::from_cuda_slice(buffer, shape)?)
    }

    fn try_matmul(
        &self,
        _a: &Self::Tensor<2>,
        _b: &Self::Tensor<2>,
        _target: &mut Self::Tensor<2>,
    ) -> Result<(), crate::model::ModelError> {
        unimplemented!("try_matmul unimplemented to CudaBackend")
    }

    fn try_add(
        &self,
        a: &Self::Tensor<2>,
        b: &Self::Tensor<2>,
        target: &mut Self::Tensor<2>,
    ) -> Result<(), crate::model::ModelError> {
        // Validate the tensor shapes.
        a.validate_add_shape_with(b)?;
        a.validate_add_shape_with(target)?;

        // Get the preloaded kernel.
        let kernel = self.kernels.get(Kernel::AddF32)?;
        let count =
            u32::try_from(a.as_cuda_slice().len()).map_err(|_| ModelError::TensorTooLarge {
                elements: a.as_cuda_slice().len(),
            })?;

        // Build launch config.
        let config = elementwise_launch_config::<256>(count)?;

        // SAFETY:
        // - `kernel` refers to `add_f32`.
        // - Arguments match the CUDA ABI and declaration order.
        // - All buffers contain at least `count` f32 elements.
        // - `target` is exclusively borrowed for the write.
        // - The kernel checks `index < count`.
        unsafe {
            self.stream
                .launch_builder(kernel)
                .arg(a.as_cuda_slice())
                .arg(b.as_cuda_slice())
                .arg(target.as_cuda_slice_mut())
                .arg(&count)
                .launch(config)?;
        }

        Ok(())
    }
}

#[inline]
fn elementwise_launch_config<const THREADS_PER_BLOCK: u32>(
    element_count: u32,
) -> Result<LaunchConfig, ModelError> {
    if THREADS_PER_BLOCK == 0 {
        return Err(ModelError::InvalidCudaBlockDimensions { x: 0, y: 1, z: 1 });
    }

    Ok(LaunchConfig {
        grid_dim: (element_count.div_ceil(THREADS_PER_BLOCK), 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::elementwise_launch_config;

    #[test]
    fn elementwise_launch_config_covers_every_element() {
        let config = elementwise_launch_config::<256>(1_025).unwrap();

        assert_eq!(config.grid_dim, (5, 1, 1));
        assert_eq!(config.block_dim, (256, 1, 1));
        assert_eq!(config.shared_mem_bytes, 0);
    }

    #[test]
    fn elementwise_launch_config_rejects_empty_blocks() {
        assert!(elementwise_launch_config::<0>(1).is_err());
    }
}
