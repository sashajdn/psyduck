use std::simd::Simd;

use tensor::{HostTensor, MatrixError, QuantizedFp};

use super::super::modifier::OutputElementWiseModifier;
use super::{ACCUMULATORS, LANES, SimdVector};

const KM: usize = 2;
const KN: usize = 4;
const KK: usize = LANES;

pub trait SimdMicroKernel<const LANES: usize, const ACCUMULATORS: usize>:
    QuantizedFp + Sized
{
    type Vector: SimdVector<Self, LANES>;

    fn matmul_microkernel<Mod>(
        a: &HostTensor<Self, 2>,
        b: &HostTensor<Self, 2>,
        c: &mut HostTensor<Self, 2>,
    ) -> Result<(), MatrixError>
    where
        Mod: OutputElementWiseModifier<Self>;

    #[inline(always)]
    fn validate_tile_sizes(m: usize, n: usize, k: usize) -> Result<(), MatrixError> {
        if m.is_multiple_of(KM) && n.is_multiple_of(KN) && k.is_multiple_of(KK) {
            Ok(())
        } else {
            Err(MatrixError::InvalidTileSize((KM, KN, KK), (m, n, k)))
        }
    }
}

impl SimdMicroKernel<LANES, ACCUMULATORS> for f32 {
    type Vector = Simd<f32, LANES>;

    fn matmul_microkernel<Mod>(
        a: &HostTensor<Self, 2>,
        b: &HostTensor<Self, 2>,
        c: &mut HostTensor<Self, 2>,
    ) -> Result<(), MatrixError>
    where
        Mod: OutputElementWiseModifier<Self>,
    {
        let m = a.rows();
        let n = b.rows();
        let k = a.columns();

        debug_assert!(m.is_multiple_of(KM));
        debug_assert!(n.is_multiple_of(KN));
        debug_assert!(k.is_multiple_of(KK));

        for i in (0..m).step_by(KM) {
            // Collect the a rows & chunk per SIMD.
            let (a0, _remainder) = a.as_slice()[i * k..(i + 1) * k].as_chunks::<LANES>();
            let (a1, _remainder) = a.as_slice()[(i + 1) * k..(i + 2) * k].as_chunks::<LANES>();

            for j in (0..n).step_by(KN) {
                let (b0, _remainder) = b.as_slice()[j * k..(j + 1) * k].as_chunks::<LANES>();
                let (b1, _remainder) = b.as_slice()[(j + 1) * k..(j + 2) * k].as_chunks::<LANES>();
                let (b2, _remainder) = b.as_slice()[(j + 2) * k..(j + 3) * k].as_chunks::<LANES>();
                let (b3, _remainder) = b.as_slice()[(j + 3) * k..(j + 4) * k].as_chunks::<LANES>();

                // c0 accumulators.
                let mut c00 = Self::Vector::splat(Self::zero());
                let mut c01 = Self::Vector::splat(Self::zero());
                let mut c02 = Self::Vector::splat(Self::zero());
                let mut c03 = Self::Vector::splat(Self::zero());

                // c1 accumulators.
                let mut c10 = Self::Vector::splat(Self::zero());
                let mut c11 = Self::Vector::splat(Self::zero());
                let mut c12 = Self::Vector::splat(Self::zero());
                let mut c13 = Self::Vector::splat(Self::zero());

                for chunk in 0..a0.len() {
                    // SIMD load a rows.
                    let a0_simd = Self::Vector::from_array(a0[chunk]);
                    let a1_simd = Self::Vector::from_array(a1[chunk]);

                    // SIMD load b rows.
                    let b0_simd = Self::Vector::from_array(b0[chunk]);
                    let b1_simd = Self::Vector::from_array(b1[chunk]);
                    let b2_simd = Self::Vector::from_array(b2[chunk]);
                    let b3_simd = Self::Vector::from_array(b3[chunk]);

                    // SIMD FMA for c accumulators across a0.
                    a0_simd.fma_accumulate(b0_simd, &mut c00);
                    a0_simd.fma_accumulate(b1_simd, &mut c01);
                    a0_simd.fma_accumulate(b2_simd, &mut c02);
                    a0_simd.fma_accumulate(b3_simd, &mut c03);

                    // SIMD FMA for c accumulators across a1.
                    a1_simd.fma_accumulate(b0_simd, &mut c10);
                    a1_simd.fma_accumulate(b1_simd, &mut c11);
                    a1_simd.fma_accumulate(b2_simd, &mut c12);
                    a1_simd.fma_accumulate(b3_simd, &mut c13);
                }

                // Horizontally reduce the accumulators and write each C element
                // according to the requested output mode.
                c.with_mut(i, j, |current| Mod::apply(current, c00.sum()))?;
                c.with_mut(i, j + 1, |current| Mod::apply(current, c01.sum()))?;
                c.with_mut(i, j + 2, |current| Mod::apply(current, c02.sum()))?;
                c.with_mut(i, j + 3, |current| Mod::apply(current, c03.sum()))?;

                c.with_mut(i + 1, j, |current| Mod::apply(current, c10.sum()))?;
                c.with_mut(i + 1, j + 1, |current| Mod::apply(current, c11.sum()))?;
                c.with_mut(i + 1, j + 2, |current| Mod::apply(current, c12.sum()))?;
                c.with_mut(i + 1, j + 3, |current| Mod::apply(current, c13.sum()))?;
            }
        }

        Ok(())
    }
}
