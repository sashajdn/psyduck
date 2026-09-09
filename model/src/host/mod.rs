use std::{
    ops::Deref,
    time::{Duration, Instant},
};

use instrument::operation::{OperationTimer, TimingClock};
use tensor::{HostTensor, MatrixTensor, Shape};

use crate::model::{ModelBackend, ModelError};

use self::{
    modifier::{Accumulate, OutputElementWiseModifier},
    simd::{ACCUMULATORS, LANES, SimdDotProduct},
};

pub(crate) mod modifier;
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

    fn try_matmul<const TM: usize, const TN: usize, const TK: usize>(
        &self,
        a: &Self::Tensor<2>,
        b: &Self::Tensor<2>,
        c: &mut Self::Tensor<2>,
    ) -> Result<(), ModelError> {
        // Validation.
        a.validate_matmul_target_with(b, c)?;

        // Construct tile shapes for the inner matmul kernel and validate.
        let a_tile_shape = Shape::new([TM, TK]);
        let b_tile_shape = Shape::new([TN, TK]);
        let c_tile_shape = Shape::new([TM, TN]);

        Self::validate_tile_shapes(&a_tile_shape, &b_tile_shape, &c_tile_shape)?;

        // Transpose `b` for memory locality.
        //
        // The tiled k-stride over `b` is now contiguous given the
        // underlying is a Vec<F> for all tiles.
        let mut b_transposed = b.clone();
        b_transposed.transpose()?;

        // Collect dimensionality.
        let m = a.rows();
        let n = b_transposed.rows();
        let k = a.columns();

        // Perform the matrix multiplication with tiling & a SIMD accelerated
        // inner dot product over tiles.
        for i in (0..m).step_by(TM) {
            for j in (0..n).step_by(TN) {
                // Build tile of C{m, n} for C outputs.
                let mut c_tile = self.alloc(c_tile_shape.clone())?;

                for kk in (0..k).step_by(TK) {
                    // Construct resident tiles.
                    let a_tile = a.copy_tile([i, kk], a_tile_shape.clone())?;
                    let b_tile = b_transposed.copy_tile([j, kk], b_tile_shape.clone())?;

                    // Perform the inner matmul kernel with SIMD acceleration over constructed tiles.
                    self.try_matmul_kernel_inner::<Accumulate>(
                        &a_tile,
                        MaybeTransposedMatrix::transposed(&b_tile),
                        &mut c_tile,
                    )?;
                }

                // Write c_tile back to output.
                c.write_tile([i, j], &c_tile)?;
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

enum MaybeTransposedMatrix<'a, F: SimdDotProduct<LANES, ACCUMULATORS>> {
    NotTransposed(&'a mut HostTensor<F, 2>),
    Transposed(&'a HostTensor<F, 2>),
}

impl<F: SimdDotProduct<LANES, ACCUMULATORS>> Deref for MaybeTransposedMatrix<'_, F> {
    type Target = HostTensor<F, 2>;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::NotTransposed(matrix) => matrix,
            Self::Transposed(matrix) => matrix,
        }
    }
}

impl<'a, F: SimdDotProduct<LANES, ACCUMULATORS>> MaybeTransposedMatrix<'a, F> {
    #[inline(always)]
    fn transposed(matrix: &'a HostTensor<F, 2>) -> Self {
        Self::Transposed(matrix)
    }

    #[inline(always)]
    fn not_transposed(matrix: &'a mut HostTensor<F, 2>) -> Self {
        Self::NotTransposed(matrix)
    }
}

impl<F: SimdDotProduct<LANES, ACCUMULATORS>> HostModelBackend<F> {
    fn try_matmul_kernel_inner<Mod: OutputElementWiseModifier<F>>(
        &self,
        a_tile: &HostTensor<F, 2>,
        b_tile: MaybeTransposedMatrix<F>,
        c_tile: &mut HostTensor<F, 2>,
    ) -> Result<(), ModelError> {
        let b_transposed = match b_tile {
            MaybeTransposedMatrix::NotTransposed(b) => {
                b.transpose()?;
                b
            }
            MaybeTransposedMatrix::Transposed(b_t) => b_t,
        };

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

    fn validate_tile_shapes(a: &Shape<2>, b: &Shape<2>, c: &Shape<2>) -> Result<(), ModelError> {
        if a.rows() != b.columns() {
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
}
