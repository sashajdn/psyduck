use std::simd::{Simd, StdFloat, num::SimdFloat};

mod macros;
pub mod row;
pub mod tile;

use tensor::QuantizedFp;

pub use row::SimdDotProduct;
pub use tile::SimdMicroKernel;

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

    #[inline(always)]
    fn fma_accumulate(self, multiplier: Self, accumulator: &mut Self) {
        *accumulator = self.mul_add(multiplier, *accumulator);
    }
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
