use std::{
    fmt,
    ops::{Add, Mul, Sub},
};

use crate::Shape;

pub trait Tensor<F, const R: usize> {
    fn shape(&self) -> &Shape<R>;
}

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
#[error("tensor element count mismatch: expected {expected}, actual {actual}")]
pub struct ElementCountMismatch {
    pub expected: usize,
    pub actual: usize,
}

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
#[error("tensor shape mismatch: lhs {lhs:?}, rhs {rhs:?}")]
pub struct ShapeMismatch {
    pub lhs: Vec<usize>,
    pub rhs: Vec<usize>,
}

pub trait MatrixTensor<F>: Tensor<F, 2> {
    #[inline]
    fn validate_matmul_shape_with<T: Tensor<F, 2>>(&self, rhs: &T) -> Result<(), ShapeMismatch> {
        (self.shape().cols() == rhs.shape().rows())
            .then_some(())
            .ok_or_else(|| ShapeMismatch {
                lhs: self.shape().dims().to_vec(),
                rhs: rhs.shape().dims().to_vec(),
            })
    }

    #[inline]
    fn validate_matmul_target_with<Rhs, Target>(
        &self,
        rhs: &Rhs,
        target: &Target,
    ) -> Result<(), ShapeMismatch>
    where
        Rhs: Tensor<F, 2>,
        Target: Tensor<F, 2>,
    {
        self.validate_matmul_shape_with(rhs)?;

        let expected = [self.shape().rows(), rhs.shape().cols()];
        (target.shape().dims() == &expected)
            .then_some(())
            .ok_or_else(|| ShapeMismatch {
                lhs: expected.to_vec(),
                rhs: target.shape().dims().to_vec(),
            })
    }

    #[inline]
    fn validate_add_shape_with<T: Tensor<F, 2>>(&self, rhs: &T) -> Result<(), ShapeMismatch> {
        (self.shape().dims() == rhs.shape().dims())
            .then_some(())
            .ok_or_else(|| ShapeMismatch {
                lhs: self.shape().dims().to_vec(),
                rhs: rhs.shape().dims().to_vec(),
            })
    }
}

impl<F, T: Tensor<F, 2>> MatrixTensor<F> for T {}

pub trait QuantizedFp:
    fmt::Debug + Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
{
    fn from_f32(value: f32) -> Self;
    fn zero() -> Self;
}

impl QuantizedFp for f64 {
    #[inline]
    fn from_f32(value: f32) -> Self {
        Self::from(value)
    }

    #[inline]
    fn zero() -> Self {
        0.0
    }
}

impl QuantizedFp for f32 {
    #[inline]
    fn from_f32(value: f32) -> Self {
        value
    }

    #[inline]
    fn zero() -> Self {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HostTensor;

    #[test]
    fn validates_matmul_shapes() {
        let lhs = HostTensor::<f32, 2>::zeros(Shape::new([2, 3]));
        let compatible_rhs = HostTensor::<f32, 2>::zeros(Shape::new([3, 4]));
        let incompatible_rhs = HostTensor::<f32, 2>::zeros(Shape::new([4, 3]));

        assert_eq!(lhs.validate_matmul_shape_with(&compatible_rhs), Ok(()));
        assert_eq!(
            lhs.validate_matmul_shape_with(&incompatible_rhs),
            Err(ShapeMismatch {
                lhs: vec![2, 3],
                rhs: vec![4, 3],
            })
        );
    }

    #[test]
    fn validates_matmul_target_shape() {
        let lhs = HostTensor::<f32, 2>::zeros(Shape::new([2, 3]));
        let rhs = HostTensor::<f32, 2>::zeros(Shape::new([3, 4]));
        let compatible_target = HostTensor::<f32, 2>::zeros(Shape::new([2, 4]));
        let incompatible_target = HostTensor::<f32, 2>::zeros(Shape::new([4, 2]));

        assert_eq!(
            lhs.validate_matmul_target_with(&rhs, &compatible_target),
            Ok(())
        );
        assert_eq!(
            lhs.validate_matmul_target_with(&rhs, &incompatible_target),
            Err(ShapeMismatch {
                lhs: vec![2, 4],
                rhs: vec![4, 2],
            })
        );
    }
}
