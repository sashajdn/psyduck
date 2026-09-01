#![allow(unused_macros)]

macro_rules! accumulator_count {
    (@unit $accumulator:ident) => {
        ()
    };
    ($($accumulator:ident),+ $(,)?) => {
        [$(accumulator_count!(@unit $accumulator)),+].len()
    };
}

macro_rules! unrolled_simd_dot_product {
    (
        $accumulators:literal;
        $a:expr,
        $b:expr;
        $($index:literal => $accumulator:ident),+ $(,)?
    ) => {{
        const {
            assert!(LANES > 0, "LANES must be greater than zero");
            assert!($accumulators > 0, "ACCUMULATORS must be greater than zero");
            assert!(
                $accumulators == accumulator_count!($($accumulator),+),
                "ACCUMULATORS must match the unrolled accumulator count",
            );
        }

        let a = $a;
        let b = $b;
        assert_eq!(
            a.len(),
            b.len(),
            "dot-product inputs must have equal lengths"
        );

        // Expand one independent accumulator for every unrolled accumulator in the hot loop.
        $(
            let mut $accumulator =
                <Self::Vector as SimdVector<Self, LANES>>::splat(Self::zero());
        )+

        let (a_chunks, a_scalar_remainder) = a.as_chunks::<LANES>();
        let (b_chunks, b_scalar_remainder) = b.as_chunks::<LANES>();
        let (a_groups, a_chunk_remainder) = a_chunks.as_chunks::<$accumulators>();
        let (b_groups, b_chunk_remainder) = b_chunks.as_chunks::<$accumulators>();

        // Handle complete SIMD chunks that fill the unrolled accumulator groups.
        for (a_group, b_group) in a_groups.iter().zip(b_groups.iter()) {
            $(
                <Self as SimdDotProduct<LANES, $accumulators>>::simd_accumulate(
                    &a_group[$index],
                    &b_group[$index],
                    &mut $accumulator,
                );
            )+
        }

        // Handle complete SIMD chunks that do not fill another accumulator group.
        let mut remainder_accumulator =
            <Self::Vector as SimdVector<Self, LANES>>::splat(Self::zero());
        for (a_chunk, b_chunk) in a_chunk_remainder.iter().zip(b_chunk_remainder.iter()) {
            <Self as SimdDotProduct<LANES, $accumulators>>::simd_accumulate(
                a_chunk,
                b_chunk,
                &mut remainder_accumulator,
            );
        }

        // Zero-pad the final partial SIMD chunk.
        if !a_scalar_remainder.is_empty() {
            let mut padded_a = [Self::zero(); LANES];
            let mut padded_b = [Self::zero(); LANES];

            padded_a[..a_scalar_remainder.len()].copy_from_slice(a_scalar_remainder);
            padded_b[..b_scalar_remainder.len()].copy_from_slice(b_scalar_remainder);

            <Self as SimdDotProduct<LANES, $accumulators>>::simd_accumulate(
                &padded_a,
                &padded_b,
                &mut remainder_accumulator,
            );
        }

        // Combine the independent vectors after the hot loop, then reduce once.
        $(remainder_accumulator = remainder_accumulator.add($accumulator);)+
        remainder_accumulator.sum()
    }};
}

macro_rules! impl_simd_dot_product {
    ($accumulators:literal; $($index:literal => $accumulator:ident),+ $(,)?) => {
        impl<const LANES: usize> SimdDotProduct<LANES, $accumulators> for f32 {
            type Vector = Simd<f32, LANES>;

            #[inline]
            fn simd_dot_product(a: &[Self], b: &[Self]) -> Self {
                unrolled_simd_dot_product!(
                    $accumulators;
                    a,
                    b;
                    $($index => $accumulator),+
                )
            }
        }
    };
}
