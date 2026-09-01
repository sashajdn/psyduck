use std::time::{Duration, Instant};

use instrument::operation::{OperationTimer, TimingClock};
use tensor::{HostTensor, MatrixTensor, Shape};

use crate::model::{ModelBackend, ModelError};

use self::simd::{ACCUMULATORS, LANES, SimdDotProduct};

pub mod simd;
pub mod stride;

pub struct HostModelBackend<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F> HostModelBackend<F> {
    pub const fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F> Default for HostModelBackend<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> OperationTimer for HostModelBackend<F> {
    type Error = ModelError;
    type Marker = Instant;

    const CLOCK: TimingClock = TimingClock::HostWall;

    #[inline]
    fn mark(&self) -> Result<Self::Marker, Self::Error> {
        Ok(Instant::now())
    }

    #[inline]
    fn elapsed(&self, start: Self::Marker, end: Self::Marker) -> Result<Duration, Self::Error> {
        Ok(end.duration_since(start))
    }

    #[inline]
    fn synchronize(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl<F: SimdDotProduct<LANES, ACCUMULATORS>> ModelBackend<F> for HostModelBackend<F> {
    type Tensor<const R: usize> = HostTensor<F, R>;

    #[inline]
    fn upload<const R: usize>(
        &self,
        source: &HostTensor<F, R>,
    ) -> Result<Self::Tensor<R>, ModelError> {
        Ok(source.clone())
    }

    #[inline]
    fn download<const R: usize>(
        &self,
        source: &Self::Tensor<R>,
    ) -> Result<HostTensor<F, R>, ModelError> {
        Ok(source.clone())
    }

    #[inline]
    fn alloc<const R: usize>(&self, shape: Shape<R>) -> Result<Self::Tensor<R>, ModelError> {
        Ok(HostTensor::zeros(shape))
    }

    fn try_matmul(
        &self,
        a: &Self::Tensor<2>,
        b: &Self::Tensor<2>,
        c: &mut Self::Tensor<2>,
    ) -> Result<(), ModelError> {
        a.validate_matmul_target_with(b, c)?;

        // Transpose `b` for memory locality.
        //
        // The k-stride over `b` is now contiguous given the
        // underlying is a Vec<F>.
        let mut b_transposed = b.clone();
        b_transposed.transpose()?;

        let k = a.columns();
        for i in 0..a.rows() {
            // Collect i-th row of `A`.
            let a_row = &a.as_slice()[i * k..(i + 1) * k];

            for j in 0..b_transposed.rows() {
                // Collect j-th (transposed) row of `B`.
                let b_row = &b_transposed.as_slice()[j * k..(j + 1) * k];

                // Compute the dot product of the i-th row of `A` and the j-th row of `B`
                // with SIMD acceleration across `ACCUMULATORS` SIMD registers.
                c.set(i, j, F::simd_dot_product(a_row, b_row))?;
            }
        }

        Ok(())
    }

    #[inline]
    fn try_add(
        &self,
        a: &Self::Tensor<2>,
        b: &Self::Tensor<2>,
        target: &mut Self::Tensor<2>,
    ) -> Result<(), ModelError> {
        a.validate_add_shape_with(b)?;
        a.validate_add_shape_with(target)?;

        for ((a, b), output) in a
            .as_slice()
            .iter()
            .zip(b.as_slice())
            .zip(target.as_mut_slice())
        {
            *output = *a + *b;
        }

        Ok(())
    }

    #[inline(always)]
    fn transpose(target: &mut Self::Tensor<2>) -> Result<(), ModelError> {
        target.transpose().map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use tensor::{HostTensor, Shape};

    use super::{HostModelBackend, ModelBackend, SimdDotProduct};

    fn square_matrix(values: &[f32; 4]) -> HostTensor<f32, 2> {
        let mut tensor = HostTensor::zeros(Shape::new([2, 2]));
        tensor.as_mut_slice().copy_from_slice(values);
        tensor
    }

    fn rectangular_matrix(values: &[f32; 6]) -> HostTensor<f32, 2> {
        let mut tensor = HostTensor::zeros(Shape::new([2, 3]));
        tensor.as_mut_slice().copy_from_slice(values);
        tensor
    }

    #[test]
    fn correctly_adds_two_matrices() {
        let backend = HostModelBackend::<f32>::new();
        let a = square_matrix(&[1.0, 2.0, 3.0, 4.0]);
        let b = square_matrix(&[5.0, 6.0, 7.0, 8.0]);
        let mut target = HostTensor::zeros(Shape::new([2, 2]));

        backend
            .try_add(&a, &b, &mut target)
            .expect("2x2 matrices should be addable");

        assert_eq!(target.as_slice(), &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn correctly_multiplies_two_matrices() {
        let backend = HostModelBackend::<f32>::new();
        let a = square_matrix(&[1.0, 2.0, 3.0, 4.0]);
        let b = square_matrix(&[5.0, 6.0, 7.0, 8.0]);
        let mut target = HostTensor::zeros(Shape::new([2, 2]));

        backend
            .try_matmul(&a, &b, &mut target)
            .expect("2x2 matrices should be multipliable");

        assert_eq!(target.as_slice(), &[19.0, 22.0, 43.0, 50.0]);

        backend
            .try_matmul(&a, &b, &mut target)
            .expect("repeated matmul should overwrite its target");

        assert_eq!(target.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn correctly_accumulates_full_simd_chunks_and_the_remainder() {
        let backend = HostModelBackend::<f32>::new();
        let mut a = HostTensor::zeros(Shape::new([1, 10]));
        let mut b = HostTensor::zeros(Shape::new([10, 1]));
        let mut target = HostTensor::zeros(Shape::new([1, 1]));

        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        a.as_mut_slice().copy_from_slice(&values);
        b.as_mut_slice().copy_from_slice(&values);

        backend
            .try_matmul(&a, &b, &mut target)
            .expect("a full SIMD chunk and its remainder should be multipliable");

        assert_eq!(target.as_slice(), &[385.0]);
    }

    #[test]
    fn supports_a_generic_simd_lane_count() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let result = <f32 as SimdDotProduct<4, 8>>::simd_dot_product(&values, &values);

        assert_eq!(result, 91.0);
    }

    #[test]
    fn correctly_transpose_rank_2_matrix_inplace() {
        // Validate square transpose.
        let mut square = square_matrix(&[1.0, 2.0, 3.0, 4.0]);
        HostModelBackend::<f32>::transpose(&mut square).expect("2x2 matrix should be transposable");
        assert_eq!(square.as_slice(), &[1.0, 3.0, 2.0, 4.0]);

        // Validate rectangular transpose.
        let mut rectangular = rectangular_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        HostModelBackend::<f32>::transpose(&mut rectangular)
            .expect("2x3 matrix should be transposable");
        assert_eq!((rectangular.rows(), rectangular.columns()), (3, 2));
        assert_eq!(rectangular.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        HostModelBackend::<f32>::transpose(&mut rectangular)
            .expect("3x2 matrix should be transposable");
        assert_eq!((rectangular.rows(), rectangular.columns()), (2, 3));
        assert_eq!(rectangular.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
