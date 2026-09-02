use std::simd::{Simd, StdFloat, num::SimdFloat};

use tensor::QuantizedFp;

mod macros;

pub(crate) const LANES: usize = 8;
pub(crate) const ACCUMULATORS: usize = 8;

pub trait SimdVector<T, const LANES: usize>: Copy
where
    T: QuantizedFp,
{
    #[allow(dead_code)]
    fn splat(initial: T) -> Self;
    fn from_array(lanes: &[T; LANES]) -> Self;
    fn mul_add(self, multiplier: Self, accumulator: Self) -> Self;
    fn add(self, rhs: Self) -> Self;
    fn sum(self) -> T;
}

impl<const LANES: usize> SimdVector<f32, LANES> for Simd<f32, LANES> {
    #[inline(always)]
    fn splat(initial: f32) -> Self {
        Simd::splat(initial)
    }

    #[inline(always)]
    fn from_array(lanes: &[f32; LANES]) -> Self {
        Simd::from_array(*lanes)
    }

    #[inline(always)]
    fn mul_add(self, multiplier: Self, accumulator: Self) -> Self {
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

pub trait SimdDotProduct<const LANES: usize, const ACCUMULATORS: usize>:
    QuantizedFp + Sized
{
    type Vector: SimdVector<Self, LANES>;

    #[inline(always)]
    fn simd_accumulate(a: &[Self; LANES], b: &[Self; LANES], accumulator: &mut Self::Vector) {
        let a_simd = Self::Vector::from_array(a);
        let b_simd = Self::Vector::from_array(b);
        *accumulator = a_simd.mul_add(b_simd, *accumulator);
    }

    fn binary_sum(accumulators: [Self::Vector; ACCUMULATORS]) -> Self::Vector;

    fn simd_dot_product(a: &[Self], b: &[Self]) -> Self;
}

// Eight accumulators are the measured optimum on the target Xeon D-1531.
impl<const LANES: usize> SimdDotProduct<LANES, 8> for f32 {
    type Vector = Simd<f32, LANES>;

    #[inline(always)]
    fn binary_sum(accumulators: [Self::Vector; 8]) -> Self::Vector {
        let [
            accumulator_0,
            accumulator_1,
            accumulator_2,
            accumulator_3,
            accumulator_4,
            accumulator_5,
            accumulator_6,
            accumulator_7,
        ] = accumulators;

        let pair_01 = accumulator_0.add(accumulator_1);
        let pair_23 = accumulator_2.add(accumulator_3);
        let pair_45 = accumulator_4.add(accumulator_5);
        let pair_67 = accumulator_6.add(accumulator_7);

        let half_0 = pair_01.add(pair_23);
        let half_1 = pair_45.add(pair_67);

        half_0.add(half_1)
    }

    #[inline]
    fn simd_dot_product(a: &[Self], b: &[Self]) -> Self {
        const {
            assert!(LANES > 0, "LANES must be greater than zero");
        }

        assert_eq!(
            a.len(),
            b.len(),
            "dot-product inputs must have equal lengths"
        );

        // Hand unrolled eight accumulators to avoid register pressure and allow the compiler to optimize the hot loop.
        //
        // This forces the LLVM compiler to unroll per iteration. Peaking at assembly showed any
        // attempt to make an abstraction resulted in a rolled loop.
        let mut accumulator_0 = Self::Vector::splat(Self::zero());
        let mut accumulator_1 = Self::Vector::splat(Self::zero());
        let mut accumulator_2 = Self::Vector::splat(Self::zero());
        let mut accumulator_3 = Self::Vector::splat(Self::zero());
        let mut accumulator_4 = Self::Vector::splat(Self::zero());
        let mut accumulator_5 = Self::Vector::splat(Self::zero());
        let mut accumulator_6 = Self::Vector::splat(Self::zero());
        let mut accumulator_7 = Self::Vector::splat(Self::zero());

        let (a_chunks, a_scalar_remainder) = a.as_chunks::<LANES>();
        let (b_chunks, b_scalar_remainder) = b.as_chunks::<LANES>();
        let (a_groups, a_chunk_remainder) = a_chunks.as_chunks::<8>();
        let (b_groups, b_chunk_remainder) = b_chunks.as_chunks::<8>();

        for (a_group, b_group) in a_groups.iter().zip(b_groups) {
            Self::simd_accumulate(&a_group[0], &b_group[0], &mut accumulator_0);
            Self::simd_accumulate(&a_group[1], &b_group[1], &mut accumulator_1);
            Self::simd_accumulate(&a_group[2], &b_group[2], &mut accumulator_2);
            Self::simd_accumulate(&a_group[3], &b_group[3], &mut accumulator_3);
            Self::simd_accumulate(&a_group[4], &b_group[4], &mut accumulator_4);
            Self::simd_accumulate(&a_group[5], &b_group[5], &mut accumulator_5);
            Self::simd_accumulate(&a_group[6], &b_group[6], &mut accumulator_6);
            Self::simd_accumulate(&a_group[7], &b_group[7], &mut accumulator_7);
        }

        let mut remainder_accumulator = Self::Vector::splat(Self::zero());
        for (a_chunk, b_chunk) in a_chunk_remainder.iter().zip(b_chunk_remainder) {
            Self::simd_accumulate(a_chunk, b_chunk, &mut remainder_accumulator);
        }

        if !a_scalar_remainder.is_empty() {
            let mut padded_a = [Self::zero(); LANES];
            let mut padded_b = [Self::zero(); LANES];
            padded_a[..a_scalar_remainder.len()].copy_from_slice(a_scalar_remainder);
            padded_b[..b_scalar_remainder.len()].copy_from_slice(b_scalar_remainder);

            Self::simd_accumulate(&padded_a, &padded_b, &mut remainder_accumulator);
        }

        // Sum across the independent accumulators.
        let accumulator = Self::binary_sum([
            accumulator_0,
            accumulator_1,
            accumulator_2,
            accumulator_3,
            accumulator_4,
            accumulator_5,
            accumulator_6,
            accumulator_7,
        ]);

        // Before finally reducing once.
        remainder_accumulator.add(accumulator).sum()
    }
}
