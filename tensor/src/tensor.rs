use std::ops::{Add, Mul, Sub};

use crate::Shape;

pub trait Tensor<F, const R: usize> {
    fn zeros(shape: Shape<R>) -> Self;
    fn shape(&self) -> &Shape<R>;
}

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
#[error("tensor shape mismatch: lhs {lhs:?}, rhs {rhs:?}")]
pub struct ShapeMismatch {
    pub lhs: Vec<usize>,
    pub rhs: Vec<usize>,
}

pub trait MatrixTensor<F>: Tensor<F, 2> {
    #[inline]
    fn validate_matmul_shape<T: Tensor<F, 2>>(&self, rhs: &T) -> Result<(), ShapeMismatch> {
        (self.shape().cols() == rhs.shape().rows())
            .then_some(())
            .ok_or_else(|| ShapeMismatch {
                lhs: self.shape().dims().to_vec(),
                rhs: rhs.shape().dims().to_vec(),
            })
    }

    #[inline]
    fn validate_add_shape<T: Tensor<F, 2>>(&self, rhs: &T) -> Result<(), ShapeMismatch> {
        (self.shape().dims() == rhs.shape().dims())
            .then_some(())
            .ok_or_else(|| ShapeMismatch {
                lhs: self.shape().dims().to_vec(),
                rhs: rhs.shape().dims().to_vec(),
            })
    }
}

impl<F, T: Tensor<F, 2>> MatrixTensor<F> for T {}

pub struct NaiveTensor<F, const R: usize> {
    inner: Vec<F>,
    shape: Shape<R>,
}

impl<F: QuantizedFp, const R: usize> NaiveTensor<F, R> {
    #[allow(dead_code)]
    #[inline]
    pub fn as_slice(&self) -> &[F] {
        &self.inner
    }

    #[allow(dead_code)]
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [F] {
        &mut self.inner
    }
}

impl<F: QuantizedFp, const R: usize> Tensor<F, R> for NaiveTensor<F, R> {
    #[inline]
    fn zeros(shape: Shape<R>) -> Self {
        let inner = (0..shape.numel()).map(|_| F::zero()).collect();
        Self { inner, shape }
    }

    #[inline]
    fn shape(&self) -> &Shape<R> {
        &self.shape
    }
}

pub trait QuantizedFp: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> {
    fn zero() -> Self;
}

impl QuantizedFp for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
}

impl QuantizedFp for f32 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_matmul_shapes() {
        let lhs = NaiveTensor::<f32, 2>::zeros(Shape::new([2, 3]));
        let compatible_rhs = NaiveTensor::<f32, 2>::zeros(Shape::new([3, 4]));
        let incompatible_rhs = NaiveTensor::<f32, 2>::zeros(Shape::new([4, 3]));

        assert_eq!(lhs.validate_matmul_shape(&compatible_rhs), Ok(()));
        assert_eq!(
            lhs.validate_matmul_shape(&incompatible_rhs),
            Err(ShapeMismatch {
                lhs: vec![2, 3],
                rhs: vec![4, 3],
            })
        );
    }
}
