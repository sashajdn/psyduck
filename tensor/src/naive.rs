use crate::{ElementCountMismatch, QuantizedFp, Shape, Tensor};

#[derive(Clone)]
pub struct NaiveTensor<F, const R: usize> {
    inner: Vec<F>,
    shape: Shape<R>,
}

impl<F: QuantizedFp, const R: usize> NaiveTensor<F, R> {
    #[inline]
    pub fn from_vec(inner: Vec<F>, shape: Shape<R>) -> Result<Self, ElementCountMismatch> {
        let expected = shape.numel();
        let actual = inner.len();

        if actual != expected {
            return Err(ElementCountMismatch { expected, actual });
        }

        Ok(Self { inner, shape })
    }

    #[inline]
    pub fn zeros(shape: Shape<R>) -> Self {
        let inner = vec![F::zero(); shape.numel()];
        Self { inner, shape }
    }

    #[inline]
    pub fn as_slice(&self) -> &[F] {
        &self.inner
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [F] {
        &mut self.inner
    }
}

impl<F: QuantizedFp, const R: usize> Tensor<F, R> for NaiveTensor<F, R> {
    #[inline]
    fn shape(&self) -> &Shape<R> {
        &self.shape
    }
}
