use cudarc::driver::{CudaSlice, DeviceRepr};

use crate::{ElementCountMismatch, Shape, Tensor};

pub struct CudaBuffer<F: DeviceRepr> {
    buffer: CudaSlice<F>,
}

impl<F: DeviceRepr> CudaBuffer<F> {
    #[inline]
    pub fn from_cuda_slice(buffer: CudaSlice<F>) -> Self {
        Self { buffer }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    #[inline]
    pub fn as_slice(&self) -> &CudaSlice<F> {
        &self.buffer
    }

    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut CudaSlice<F> {
        &mut self.buffer
    }
}

pub struct CudaTensor<F: DeviceRepr, const R: usize> {
    buffer: CudaBuffer<F>,
    shape: Shape<R>,
}

impl<F: DeviceRepr, const R: usize> CudaTensor<F, R> {
    #[inline]
    pub fn from_cuda_slice(
        buffer: CudaSlice<F>,
        shape: Shape<R>,
    ) -> Result<Self, ElementCountMismatch> {
        let expected = shape.numel();
        let actual = buffer.len();

        if actual != expected {
            return Err(ElementCountMismatch { expected, actual });
        }

        Ok(Self {
            buffer: CudaBuffer::from_cuda_slice(buffer),
            shape,
        })
    }

    #[inline]
    pub fn as_cuda_slice(&self) -> &CudaSlice<F> {
        self.buffer.as_slice()
    }

    #[inline]
    pub fn as_cuda_slice_mut(&mut self) -> &mut CudaSlice<F> {
        self.buffer.as_slice_mut()
    }
}

impl<F: DeviceRepr> CudaTensor<F, 2> {
    #[inline]
    pub fn rows(&self) -> usize {
        self.shape().rows()
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.shape().cols()
    }
}

impl<F: DeviceRepr, const R: usize> Tensor<F, R> for CudaTensor<F, R> {
    #[inline]
    fn shape(&self) -> &Shape<R> {
        &self.shape
    }
}
