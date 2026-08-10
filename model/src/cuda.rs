use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, DeviceRepr, ValidAsZeroBits};
use tensor::{CudaTensor, NaiveTensor, QuantizedFp, Tensor};

use crate::model::{ModelBackend, ModelError};

#[derive(Debug, Clone)]
pub struct CudaModelBackend<F: DeviceRepr> {
    stream: Arc<CudaStream>,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: DeviceRepr + QuantizedFp> CudaModelBackend<F> {
    pub fn new(context: Arc<CudaContext>) -> Self {
        let stream = context.default_stream();

        Self {
            stream,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F> ModelBackend<F> for CudaModelBackend<F>
where
    F: DeviceRepr + ValidAsZeroBits + QuantizedFp,
{
    type Tensor<const R: usize> = CudaTensor<F, R>;

    fn upload<const R: usize>(
        &self,
        source: &NaiveTensor<F, R>,
    ) -> Result<Self::Tensor<R>, ModelError> {
        let buffer = self.stream.clone_htod(source.as_slice())?;
        Ok(CudaTensor::from_cuda_slice(buffer, source.shape().clone())?)
    }

    fn download<const R: usize>(
        &self,
        source: &Self::Tensor<R>,
    ) -> Result<NaiveTensor<F, R>, ModelError> {
        let values = self.stream.clone_dtoh(source.as_cuda_slice())?;
        Ok(NaiveTensor::from_vec(values, source.shape().clone())?)
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
        todo!()
    }

    fn try_add(
        &self,
        _a: &Self::Tensor<2>,
        _b: &Self::Tensor<2>,
        _target: &mut Self::Tensor<2>,
    ) -> Result<(), crate::model::ModelError> {
        todo!()
    }
}
