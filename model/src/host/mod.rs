use std::time::{Duration, Instant};

use instrument::operation::{OperationTimer, TimingClock};
use tensor::{HostTensor, MatrixTensor, QuantizedFp, Shape};

use crate::model::{ModelBackend, ModelError};

use std::simd::{f32x8, num::SimdFloat};

pub mod stride;

trait HostSimdDotProduct: QuantizedFp {
    const LANES: usize;

    fn single_simd_dot_product(a: &[Self], b: &[Self]) -> Self;
}

impl HostSimdDotProduct for f32 {
    // The target machine (AVX2) has 256-bit-wide registers, which can hold
    // eight f32 values at 32 bits each.
    const LANES: usize = 8;

    fn single_simd_dot_product(a: &[Self], b: &[Self]) -> Self {
        // Asset that lengths of the two vectors are identical over K.
        assert_eq!(a.len(), b.len());

        // Setup SIMD acculumator & set each land to zero.
        let mut accumulator = f32x8::splat(0.0);

        // Split vectors in `SIMD_LANE` chunks.
        let (a_chunks, a_remainder) = a.as_chunks::<{ Self::LANES }>();
        let (b_chunks, b_remainder) = b.as_chunks::<{ Self::LANES }>();

        // Iterate over all chunks exactly & accumulate.
        for (a_chunk, b_chunk) in a_chunks.iter().zip(b_chunks.iter()) {
            let a_simd = f32x8::from_slice(a_chunk);
            let b_simd = f32x8::from_slice(b_chunk);
            accumulator += a_simd * b_simd;
        }

        // Reduce the SIMD accumulator to a single value.
        let vector_sum = accumulator.reduce_sum();

        // Handle the remainder of the vectors that don't fit into a full SIMD chunk.
        vector_sum
            + a_remainder
                .iter()
                .zip(b_remainder.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f32>()
    }
}

const SIMD_LANES: usize = <f32 as HostSimdDotProduct>::LANES;

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

impl<F: HostSimdDotProduct> ModelBackend<F> for HostModelBackend<F> {
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
        // The k-stride over `b` is now contiguous given the
        // underlying is a Vec<F>.
        //
        // As we read cache-lines from memory, the following b-stride values are likely
        // to already be in the cache, given we've packed the cacheline - reducing the likihood of a cache miss &
        // more expensive cache read from a higher level of the memory hierarchy.
        //
        // Given a transpose operation is of order (K * M) per O(K * M * N) reads in the naive case. We are increasing the computation
        // but reducing memory reads from caches further away from compute. Hence, memory locality improves.
        //
        // We should expect slightly worse performance for low N but performance to increase relative to N,
        // as we grow pass the bound at which a given `b` K-stride fits into resident registers or memory.
        //
        // Before:
        //        B{2}
        // | 0 1 [2] |
        // | 3 4 [5] |
        // | 6 7 [8] |
        //
        // As read from Vec: [0, 1, [2], 3, 4, [5], 6, 7, [8]]
        //
        // After:
        //
        // | 0 3 6 |
        // | 1 4 7 |
        // | [2] [5] [8] |
        //
        // As read from Vec: [0, 3, 6, 1, 4, 7, [2], [5], [8]]
        //
        // Now we can see that the stride *is* contiguous in memory.
        // This makes little differnce for N < 16.
        // 16 given a f32 is 4 bytes, a cacheline is 64 bytes.
        // So we can fit 16 f32 values in a cacheline.
        //
        // If we there can pack the entire cacheline from contiguous memory only *with*
        // values we want at a given point in time, we should reduce cache misses
        // quite substantially.
        let mut b_t = b.clone();
        b_t.transpose()?;

        let k_size = a.cols();
        for i in 0..a.rows() {
            let a_row = &a.as_slice()[i * k_size..(i + 1) * k_size];

            for j in 0..b_t.rows() {
                let b_row = &b_t.as_slice()[j * k_size..(j + 1) * k_size];

                let cij = F::single_simd_dot_product(a_row, b_row);
                c.set(i, j, cij);
            }
        }

        Ok(())
    }

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

    #[inline]
    fn transpose(target: &mut Self::Tensor<2>) -> Result<(), ModelError> {
        target.transpose().map_err(Into::into)
    }
}

impl<F: QuantizedFp> HostModelBackend<F> {
    #[inline(always)]
    #[allow(unused)]
    fn binary_sum_over(stride: &mut [F; SIMD_LANES]) -> F {
        let mut width = SIMD_LANES;
        while width > 1 {
            let half = width / 2;

            for lane in 0..half {
                stride[lane] += stride[lane + half];
            }

            width = half;
        }

        stride[0]
    }
}

#[cfg(test)]
mod tests {
    use tensor::{HostTensor, Shape};

    use super::{HostModelBackend, ModelBackend};

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
    fn correctly_transpose_rank_2_matrix_inplace() {
        // Validate square transpose.
        let mut square = square_matrix(&[1.0, 2.0, 3.0, 4.0]);
        HostModelBackend::<f32>::transpose(&mut square).expect("2x2 matrix should be transposable");
        assert_eq!(square.as_slice(), &[1.0, 3.0, 2.0, 4.0]);

        // Validate rectangular transpose.
        let mut rectangular = rectangular_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        HostModelBackend::<f32>::transpose(&mut rectangular)
            .expect("2x3 matrix should be transposable");
        assert_eq!((rectangular.rows(), rectangular.cols()), (3, 2));
        assert_eq!(rectangular.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        HostModelBackend::<f32>::transpose(&mut rectangular)
            .expect("3x2 matrix should be transposable");
        assert_eq!((rectangular.rows(), rectangular.cols()), (2, 3));
        assert_eq!(rectangular.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn correct_binary_sum_over() {
        let mut stride = &mut [1.0, 2.0, 3.0, 4.0];
        let sum = HostModelBackend::<f32>::binary_sum_over(&mut stride);
        assert_eq!(sum, 10.0);
    }
}
