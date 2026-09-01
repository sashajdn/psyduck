use std::simd::{Simd, StdFloat, num::SimdFloat};

use tensor::QuantizedFp;

pub(crate) const LANES: usize = 8;
pub(crate) const ACCUMULATORS: usize = 8;

pub(crate) trait SimdVector<T, const LANES: usize>: Copy
where
    T: QuantizedFp,
{
    fn splat(initial: T) -> Self;
    fn load(lanes: &[T; LANES]) -> Self;
    fn multiply_add(self, multiplier: Self, accumulator: Self) -> Self;
    fn add(self, rhs: Self) -> Self;
    fn sum(self) -> T;
}

impl<const LANES: usize> SimdVector<f32, LANES> for Simd<f32, LANES> {
    #[inline(always)]
    fn splat(initial: f32) -> Self {
        Simd::splat(initial)
    }

    #[inline(always)]
    fn load(lanes: &[f32; LANES]) -> Self {
        Simd::from_array(*lanes)
    }

    #[inline(always)]
    fn multiply_add(self, multiplier: Self, accumulator: Self) -> Self {
        StdFloat::mul_add(self, multiplier, accumulator)
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn sum(self) -> f32 {
        SimdFloat::reduce_sum(self)
    }
}

pub(crate) trait SimdDotProduct<const LANES: usize, const ACCUMULATORS: usize>:
    QuantizedFp + Sized
{
    type Vector: SimdVector<Self, LANES>;

    #[inline(always)]
    fn simd_accumulate(a: &[Self; LANES], b: &[Self; LANES], accumulator: &mut Self::Vector) {
        let a_simd = Self::Vector::load(a);
        let b_simd = Self::Vector::load(b);
        *accumulator = a_simd.multiply_add(b_simd, *accumulator);
    }

    fn simd_dot_product(a: &[Self], b: &[Self]) -> Self {
        assert_eq!(
            a.len(),
            b.len(),
            "dot-product inputs must have equal lengths"
        );
        assert!(
            ACCUMULATORS > 0,
            "at least one SIMD accumulator is required"
        );

        // Construct set of SIMD accumulators to round robin.
        let mut accumulators = [Self::Vector::splat(Self::zero()); ACCUMULATORS];

        // Split the input slices into chunks of LANES elements.
        let (a_chunks, a_remainder) = a.as_chunks::<LANES>();
        let (b_chunks, b_remainder) = b.as_chunks::<LANES>();

        // Accumulate the dot product of each chunk into a selected SIMD accumulator.
        // NOTE: its not a guarantee that the LLVM compiler will unroll this loop.
        for (chunk_index, (a_chunk, b_chunk)) in a_chunks.iter().zip(b_chunks.iter()).enumerate() {
            let accumulator = &mut accumulators[chunk_index % ACCUMULATORS];
            Self::simd_accumulate(a_chunk, b_chunk, accumulator);
        }

        // Handle any remaining elements that don't fit into a full SIMD chunk.
        if !a_remainder.is_empty() {
            let mut a_tail = [Self::zero(); LANES];
            let mut b_tail = [Self::zero(); LANES];
            a_tail[..a_remainder.len()].copy_from_slice(a_remainder);
            b_tail[..b_remainder.len()].copy_from_slice(b_remainder);

            let accumulator = &mut accumulators[a_chunks.len() % ACCUMULATORS];
            Self::simd_accumulate(&a_tail, &b_tail, accumulator);
        }

        // Reduce the SIMD accumulators down to a single value using pairwise addition.
        let mut width = ACCUMULATORS;
        while width > 1 {
            let pairs = width / 2;

            for pair in 0..pairs {
                let left = accumulators[pair * 2];
                let right = accumulators[pair * 2 + 1];
                accumulators[pair] = left.add(right);
            }

            if width % 2 == 1 {
                accumulators[pairs] = accumulators[width - 1];
                width = pairs + 1;
            } else {
                width = pairs;
            }
        }

        // Reduce the accumulator to a single scalar value and return it.
        accumulators[0].sum()
    }
}

impl<const LANES: usize, const ACCUMULATORS: usize> SimdDotProduct<LANES, ACCUMULATORS> for f32 {
    type Vector = Simd<f32, LANES>;
}
