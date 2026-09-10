use std::time::{Duration, Instant};

use instrument::operation::{OperationTimer, TimingClock};
use tensor::{HostTensor, MatrixTensor, Shape, Tensor};

use crate::model::{ModelBackend, ModelError};

use self::{
    modifier::{Accumulate, OutputElementWiseModifier, Overwrite},
    simd::{ACCUMULATORS, LANES, SimdDotProduct, tile::SimdMicroKernel},
};

pub(crate) mod modifier;
pub mod simd;
pub mod stride;

pub struct HostModelBackend<F> {
    _phantom: std::marker::PhantomData<F>,
}

/// PreparedTiledMatmul holds the transposed version of matrix `B` and local tile
/// buffers for matrices `A`, `B`, and `C`.
struct PreparedTiledMatmul<F> {
    /// The transposed version of matrix `B` to ensure locality during tiled matrix multiplication.
    b_transposed: HostTensor<F, 2>,
    /// Local tile buffers for matrices `A`.
    a_tile: HostTensor<F, 2>,
    /// Local tile buffers for matrices `B`.
    b_tile: HostTensor<F, 2>,
    /// Local tile buffers for matrices `C`.
    c_tile: HostTensor<F, 2>,
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

impl<F> ModelBackend<F> for HostModelBackend<F>
where
    F: SimdDotProduct<LANES, ACCUMULATORS> + SimdMicroKernel<LANES, ACCUMULATORS>,
{
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

    fn try_matmul<const TM: usize, const TN: usize, const TK: usize>(
        &self,
        a: &Self::Tensor<2>,
        b: &Self::Tensor<2>,
        c: &mut Self::Tensor<2>,
    ) -> Result<(), ModelError> {
        // Validation.
        a.validate_matmul_target_with(b, c)?;

        // Dispatch once before entering either tiled loop nest.
        if F::validate_tile_sizes(TM, TN, TK).is_ok() {
            self.try_matmul_microkernel_tiled::<TM, TN, TK>(a, b, c)
        } else {
            self.try_matmul_dot_product_tiled::<TM, TN, TK>(a, b, c)
        }
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

impl<F> HostModelBackend<F>
where
    F: SimdDotProduct<LANES, ACCUMULATORS> + SimdMicroKernel<LANES, ACCUMULATORS>,
{
    /// Tiled matrix multiplication using a SIMD accelerated microkernel for the inner loop.
    fn try_matmul_microkernel_tiled<const TM: usize, const TN: usize, const TK: usize>(
        &self,
        a: &HostTensor<F, 2>,
        b: &HostTensor<F, 2>,
        c: &mut HostTensor<F, 2>,
    ) -> Result<(), ModelError> {
        // Prepare matrices for correctness & locality.
        let PreparedTiledMatmul {
            b_transposed,
            mut a_tile,
            mut b_tile,
            mut c_tile,
        } = self.prepare_tiled_matmul::<TM, TN, TK>(a, b, c)?;

        for i in (0..a.rows()).step_by(TM) {
            for j in (0..b_transposed.rows()).step_by(TN) {
                // Copy the first tile of `A` and `B` into the local tile buffers.
                a.copy_tile([i, 0], &mut a_tile)?;
                b_transposed.copy_tile([j, 0], &mut b_tile)?;

                // Compute the first tile of `C` using the microkernel.
                // Overwrites on the first iteration.
                F::matmul_microkernel::<Overwrite>(&a_tile, &b_tile, &mut c_tile)?;

                // Accumulate on the following iterations.
                for kk in (TK..a.columns()).step_by(TK) {
                    a.copy_tile([i, kk], &mut a_tile)?;
                    b_transposed.copy_tile([j, kk], &mut b_tile)?;
                    F::matmul_microkernel::<Accumulate>(&a_tile, &b_tile, &mut c_tile)?;
                }

                // Write back the computed tile to the output matrix `C`.
                c.write_tile([i, j], &c_tile)?;
            }
        }

        Ok(())
    }

    /// Tiled matrix multiplication using SIMD accelerated dot product for the inner loop.
    fn try_matmul_dot_product_tiled<const TM: usize, const TN: usize, const TK: usize>(
        &self,
        a: &HostTensor<F, 2>,
        b: &HostTensor<F, 2>,
        c: &mut HostTensor<F, 2>,
    ) -> Result<(), ModelError> {
        // Prepare matrices for correctness & locality.
        let PreparedTiledMatmul {
            b_transposed,
            mut a_tile,
            mut b_tile,
            mut c_tile,
        } = self.prepare_tiled_matmul::<TM, TN, TK>(a, b, c)?;

        for i in (0..a.rows()).step_by(TM) {
            for j in (0..b_transposed.rows()).step_by(TN) {
                // Copy the first tile of `A` and `B` into the local tile buffers.
                a.copy_tile([i, 0], &mut a_tile)?;
                b_transposed.copy_tile([j, 0], &mut b_tile)?;

                // Compute the dot product of the first tile of `A` and `B` into the local tile buffer for `C`.
                // Overwrite on the first iteration.
                Self::try_matmul_dot_product::<Overwrite>(&a_tile, &b_tile, &mut c_tile)?;

                // Accumulate on the following iterations.
                for kk in (TK..a.columns()).step_by(TK) {
                    a.copy_tile([i, kk], &mut a_tile)?;
                    b_transposed.copy_tile([j, kk], &mut b_tile)?;
                    Self::try_matmul_dot_product::<Accumulate>(&a_tile, &b_tile, &mut c_tile)?;
                }

                // Write back the computed tile to the output matrix `C`.
                c.write_tile([i, j], &c_tile)?;
            }
        }

        Ok(())
    }

    /// Compute the dot product of two tiles of matrices `A` and `B` into a tile of matrix `C`,
    /// with SIMD acceleration.
    fn try_matmul_dot_product<Mod: OutputElementWiseModifier<F>>(
        a_tile: &HostTensor<F, 2>,
        b_transposed: &HostTensor<F, 2>,
        c_tile: &mut HostTensor<F, 2>,
    ) -> Result<(), ModelError> {
        let k = a_tile.columns();
        for i in 0..a_tile.rows() {
            // Collect i-th row of `A`.
            let a_row = &a_tile.as_slice()[i * k..(i + 1) * k];

            for j in 0..b_transposed.rows() {
                // Collect j-th (transposed) row of `B`.
                let b_row = &b_transposed.as_slice()[j * k..(j + 1) * k];

                // Compute the dot product of the i-th row of `A` and the j-th row of `B`
                // with SIMD acceleration across `ACCUMULATORS` SIMD registers.
                c_tile.with_mut(i, j, |current_value| {
                    Mod::apply(current_value, F::simd_dot_product(a_row, b_row));
                })?;
            }
        }

        Ok(())
    }

    /// Prepare the matrices for tiled matrix multiplication by validating tile shapes,
    /// Ensure locality by transposing `B`, and allocating local tile buffers for `A`, `B`, and `C`.
    fn prepare_tiled_matmul<const TM: usize, const TN: usize, const TK: usize>(
        &self,
        a: &HostTensor<F, 2>,
        b: &HostTensor<F, 2>,
        c: &HostTensor<F, 2>,
    ) -> Result<PreparedTiledMatmul<F>, ModelError> {
        let a_tile_shape = Shape::new([TM, TK]);
        let mut b_tile_shape = Shape::new([TK, TN]);
        let c_tile_shape = Shape::new([TM, TN]);

        Self::validate_tile_shapes(&a_tile_shape, &b_tile_shape, &c_tile_shape)?;
        Self::validate_tile_fits(a.shape(), &a_tile_shape)?;
        Self::validate_tile_fits(b.shape(), &b_tile_shape)?;
        Self::validate_tile_fits(c.shape(), &c_tile_shape)?;

        let mut b_transposed = b.clone();
        b_transposed.transpose()?;
        b_tile_shape.transpose();

        let a_tile = self.alloc(a_tile_shape)?;
        let b_tile = self.alloc(b_tile_shape)?;
        let c_tile = self.alloc(c_tile_shape)?;

        Ok(PreparedTiledMatmul {
            b_transposed,
            a_tile,
            b_tile,
            c_tile,
        })
    }

    /// Validate that the tile shapes for matrices `A`, `B`, and `C` are compatible for matrix multiplication.
    fn validate_tile_shapes(a: &Shape<2>, b: &Shape<2>, c: &Shape<2>) -> Result<(), ModelError> {
        if a.columns() != b.rows() {
            return Err(tensor::ShapeMismatchError {
                lhs: a.dims().to_vec(),
                rhs: b.dims().to_vec(),
            }
            .into());
        }

        if a.rows() != c.rows() {
            return Err(tensor::ShapeMismatchError {
                lhs: a.dims().to_vec(),
                rhs: c.dims().to_vec(),
            }
            .into());
        }

        if b.columns() != c.columns() {
            return Err(tensor::ShapeMismatchError {
                lhs: b.dims().to_vec(),
                rhs: c.dims().to_vec(),
            }
            .into());
        }

        Ok(())
    }

    /// Validate that the tile shape fits evenly into the matrix shape for both rows and columns.
    fn validate_tile_fits(matrix: &Shape<2>, tile: &Shape<2>) -> Result<(), ModelError> {
        let tile_rows = tile.rows();
        let tile_columns = tile.columns();
        let fits = tile_rows != 0
            && tile_columns != 0
            && matrix.rows().is_multiple_of(tile_rows)
            && matrix.columns().is_multiple_of(tile_columns);

        fits.then_some(())
            .ok_or_else(|| ModelError::InvalidTileShape {
                matrix: matrix.dims().to_vec(),
                tile: tile.dims().to_vec(),
            })
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
            .try_matmul::<1, 1, 1>(&a, &b, &mut target)
            .expect("2x2 matrices should be multipliable");

        assert_eq!(target.as_slice(), &[19.0, 22.0, 43.0, 50.0]);

        backend
            .try_matmul::<1, 1, 1>(&a, &b, &mut target)
            .expect("repeated matmul should overwrite its target");

        assert_eq!(target.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn tile_sizes_one_and_two_are_functionally_identical() {
        const SIZE: usize = 16;
        const ELEMENTS: usize = SIZE * SIZE;

        let backend = HostModelBackend::<f32>::new();
        let a = HostTensor::from_vec(
            (0..ELEMENTS)
                .map(|index| ((index * 17 + 3) % 29) as f32 / 16.0 - 0.875)
                .collect(),
            Shape::new([SIZE, SIZE]),
        )
        .expect("A should contain exactly SIZE squared elements");
        let b = HostTensor::from_vec(
            (0..ELEMENTS)
                .map(|index| ((index * 11 + 7) % 31) as f32 / 16.0 - 0.9375)
                .collect(),
            Shape::new([SIZE, SIZE]),
        )
        .expect("B should contain exactly SIZE squared elements");
        let mut tile_one = HostTensor::zeros(Shape::new([SIZE, SIZE]));
        let mut tile_two = HostTensor::zeros(Shape::new([SIZE, SIZE]));

        backend
            .try_matmul::<1, 1, 1>(&a, &b, &mut tile_one)
            .expect("1x1x1 tiled matmul should succeed");
        backend
            .try_matmul::<2, 2, 2>(&a, &b, &mut tile_two)
            .expect("2x2x2 tiled matmul should succeed");

        for (index, (&one, &two)) in tile_one
            .as_slice()
            .iter()
            .zip(tile_two.as_slice())
            .enumerate()
        {
            let tolerance = 1.0e-6 * one.abs().max(two.abs()).max(1.0);
            assert!(
                (one - two).abs() <= tolerance,
                "tile outputs differ at element {index}: {one} != {two}"
            );
        }
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
            .try_matmul::<1, 1, 1>(&a, &b, &mut target)
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

    #[test]
    fn correctly_ensure_valid_square_tile_shapes() {
        let a_tile_shape = Shape::new([2, 2]);
        let b_tile_shape = Shape::new([2, 2]);
        let c_tile_shape = Shape::new([2, 2]);

        HostModelBackend::<f32>::validate_tile_shapes(&a_tile_shape, &b_tile_shape, &c_tile_shape)
            .expect("valid square tile shapes should be valid");
    }

    #[test]
    fn correctly_guard_against_invalid_square_tile_shapes() {
        let a_tile_shape = Shape::new([2, 2]);
        let b_tile_shape = Shape::new([2, 2]);
        let c_tile_shape = Shape::new([3, 3]);

        HostModelBackend::<f32>::validate_tile_shapes(&a_tile_shape, &b_tile_shape, &c_tile_shape)
            .expect_err("invalid square tile shapes should be invalid");
    }

    #[test]
    fn correctly_ensure_valid_rectangular_tile_shapes() {
        let a_tile_shape = Shape::new([2, 3]);
        let b_tile_shape = Shape::new([3, 2]);
        let c_tile_shape = Shape::new([2, 2]);

        HostModelBackend::<f32>::validate_tile_shapes(&a_tile_shape, &b_tile_shape, &c_tile_shape)
            .expect("valid rectangular tile shapes should be valid");
    }

    #[test]
    fn correctly_guard_against_invalid_rectangular_tile_shapes() {
        let a_tile_shape = Shape::new([3, 2]);
        let b_tile_shape = Shape::new([3, 2]);
        let c_tile_shape = Shape::new([2, 2]);

        HostModelBackend::<f32>::validate_tile_shapes(&a_tile_shape, &b_tile_shape, &c_tile_shape)
            .expect_err("invalid rectangular tile shapes should be invalid");
    }

    #[test]
    fn correctly_ensures_tile_fits_matrix() {
        let matrix_shape = Shape::new([128, 256]);
        let tile_shape = Shape::new([32, 64]);

        HostModelBackend::<f32>::validate_tile_fits(&matrix_shape, &tile_shape)
            .expect("tile dimensions should evenly divide matrix dimensions");
    }

    #[test]
    fn correctly_rejects_tile_that_does_not_fit_matrix() {
        let matrix_shape = Shape::new([128, 256]);

        HostModelBackend::<f32>::validate_tile_fits(&matrix_shape, &Shape::new([32, 48]))
            .expect_err("tile dimensions should evenly divide matrix dimensions");

        HostModelBackend::<f32>::validate_tile_fits(&matrix_shape, &Shape::new([0, 64]))
            .expect_err("tile dimensions must be non-zero");
    }
}
