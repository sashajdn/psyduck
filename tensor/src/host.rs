use crate::{ElementCountMismatch, MatrixError, QuantizedFp, Shape, Tensor};

#[derive(Clone)]
pub struct HostTensor<F, const R: usize> {
    inner: Vec<F>,
    shape: Shape<R>,
}

impl<F: QuantizedFp, const R: usize> HostTensor<F, R> {
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

impl<F: QuantizedFp, const R: usize> Tensor<F, R> for HostTensor<F, R> {
    #[inline]
    fn shape(&self) -> &Shape<R> {
        &self.shape
    }
}

impl<F: QuantizedFp> HostTensor<F, 2> {
    #[inline]
    pub fn rows(&self) -> usize {
        self.shape().rows()
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.shape().cols()
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> Result<F, MatrixError> {
        let shape = self.shape().clone();
        let cols = shape.cols();

        x.checked_mul(cols)
            .and_then(|offset| offset.checked_add(y))
            .filter(|_| y < cols)
            .and_then(|index| self.inner.get(index))
            .cloned()
            .ok_or(MatrixError::OutOfBounds(shape))
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: F) -> Result<(), MatrixError> {
        let shape = self.shape().clone();
        let cols = shape.cols();

        let xy = x
            .checked_mul(cols)
            .and_then(|offset| offset.checked_add(y))
            .filter(|_| y < cols)
            .and_then(|index| self.inner.get_mut(index))
            .ok_or(MatrixError::OutOfBounds(shape))?;

        *xy = v;
        Ok(())
    }

    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize) -> Result<&mut F, MatrixError> {
        let shape = self.shape().clone();
        let cols = shape.cols();

        x.checked_mul(cols)
            .and_then(|offset| offset.checked_add(y))
            .filter(|_| y < cols)
            .and_then(|index| self.inner.get_mut(index))
            .ok_or(MatrixError::OutOfBounds(shape))
    }
}
