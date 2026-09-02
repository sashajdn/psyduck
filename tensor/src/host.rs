use crate::{ElementCountMismatch, MatrixError, QuantizedFp, Shape, Tensor};

#[derive(Clone)]
pub struct HostTensor<F, const R: usize> {
    inner: Vec<F>,
    shape: Shape<R>,
}

impl<F: QuantizedFp, const R: usize> HostTensor<F, R> {
    #[inline]
    pub fn from_vec(inner: Vec<F>, shape: Shape<R>) -> Result<Self, ElementCountMismatch> {
        let expected = shape.numel();
        let actual = inner.len();

        if actual != expected {
            return Err(ElementCountMismatch { expected, actual });
        }

        Ok(Self { inner, shape })
    }

    #[inline]
    pub fn zeros(shape: Shape<R>) -> Self {
        let inner = vec![F::zero(); shape.numel()];
        Self { inner, shape }
    }

    #[inline]
    pub fn as_slice(&self) -> &[F] {
        &self.inner
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [F] {
        &mut self.inner
    }
}

impl<F: QuantizedFp, const R: usize> Tensor<F, R> for HostTensor<F, R> {
    #[inline]
    fn shape(&self) -> &Shape<R> {
        &self.shape
    }
}

impl<F: QuantizedFp> HostTensor<F, 2> {
    pub fn copy_tile(&self, origin: [usize; 2], shape: Shape<2>) -> Result<Self, MatrixError> {
        let [row_start, column_start] = origin;
        let rows = shape.rows();
        let columns = shape.cols();

        row_start
            .checked_add(rows)
            .filter(|&row_end| row_end <= self.rows())
            .ok_or_else(|| MatrixError::OutOfBounds(self.shape().clone()))?;

        column_start
            .checked_add(columns)
            .filter(|&column_end| column_end <= self.columns())
            .ok_or_else(|| MatrixError::OutOfBounds(self.shape().clone()))?;

        let mut tile = Self::zeros(shape);
        for tile_row in 0..rows {
            let source_start = (row_start + tile_row) * self.columns() + column_start;
            let target_start = tile_row * columns;

            tile.as_mut_slice()[target_start..target_start + columns]
                .copy_from_slice(&self.as_slice()[source_start..source_start + columns]);
        }

        Ok(tile)
    }

    pub fn write_tile(&mut self, origin: [usize; 2], tile: &Self) -> Result<(), MatrixError> {
        let [row_start, column_start] = origin;
        let rows = tile.rows();
        let columns = tile.columns();

        row_start
            .checked_add(rows)
            .filter(|&row_end| row_end <= self.rows())
            .ok_or_else(|| MatrixError::OutOfBounds(self.shape().clone()))?;

        column_start
            .checked_add(columns)
            .filter(|&column_end| column_end <= self.columns())
            .ok_or_else(|| MatrixError::OutOfBounds(self.shape().clone()))?;

        for tile_row in 0..rows {
            let source_start = tile_row * columns;
            let target_start = (row_start + tile_row) * self.columns() + column_start;
            self.as_mut_slice()[target_start..target_start + columns]
                .copy_from_slice(&tile.as_slice()[source_start..source_start + columns]);
        }

        Ok(())
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.shape().rows()
    }

    #[inline]
    pub fn columns(&self) -> usize {
        self.shape().cols()
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> Result<F, MatrixError> {
        let cols = self.columns();

        x.checked_mul(cols)
            .and_then(|offset| offset.checked_add(y))
            .filter(|_| y < cols)
            .and_then(|index| self.inner.get(index))
            .cloned()
            .ok_or(MatrixError::OutOfBounds(self.shape().clone()))
    }

    #[inline(always)]
    pub fn set(&mut self, x: usize, y: usize, new_value: F) -> Result<(), MatrixError> {
        self.with_mut(x, y, |value| *value = new_value)
    }

    #[inline(always)]
    pub fn assign_add(&mut self, x: usize, y: usize, value: F) -> Result<(), MatrixError> {
        self.with_mut(x, y, |current_value| *current_value += value)
    }

    #[inline]
    pub fn swap(&mut self, x1: usize, y1: usize, x2: usize, y2: usize) -> Result<(), MatrixError> {
        let cols = self.columns();
        let index1 = x1
            .checked_mul(cols)
            .and_then(|offset| offset.checked_add(y1))
            .filter(|_| y1 < cols)
            .filter(|index| *index < self.inner.len())
            .ok_or(MatrixError::OutOfBounds(self.shape.clone()))?;

        let index2 = x2
            .checked_mul(cols)
            .and_then(|offset| offset.checked_add(y2))
            .filter(|_| y2 < cols)
            .filter(|index| *index < self.inner.len())
            .ok_or(MatrixError::OutOfBounds(self.shape.clone()))?;

        self.inner.swap(index1, index2);
        Ok(())
    }

    /// Transposes this row-major matrix in place.
    pub fn transpose(&mut self) -> Result<(), MatrixError> {
        let rows = self.rows();
        let cols = self.columns();
        let mut visited = vec![false; self.inner.len()];

        for start in 0..self.inner.len() {
            if visited[start] {
                continue;
            }

            let mut current = start;
            loop {
                visited[current] = true;

                let row = current / cols;
                let column = current % cols;
                let next = column * rows + row;

                if next == start {
                    break;
                }

                self.swap(start / cols, start % cols, next / cols, next % cols)?;
                current = next;
            }
        }

        self.shape = Shape::new([cols, rows]);
        Ok(())
    }

    #[inline]
    pub fn with_mut(
        &mut self,
        x: usize,
        y: usize,
        f: impl FnOnce(&mut F),
    ) -> Result<(), MatrixError> {
        let shape = self.shape().clone();
        let columns = shape.cols();

        let value = x
            .checked_mul(columns)
            .and_then(|offset| offset.checked_add(y))
            .filter(|_| y < columns)
            .and_then(|index| self.inner.get_mut(index))
            .ok_or(MatrixError::OutOfBounds(shape))?;

        f(value);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{HostTensor, Shape, Tensor};

    #[test]
    fn copies_a_compact_rank_two_tile() {
        let source = HostTensor::from_vec(
            (0..20).map(|value| value as f32).collect(),
            Shape::new([4, 5]),
        )
        .expect("source shape should match its elements");

        let tile = source
            .copy_tile([1, 1], Shape::new([2, 3]))
            .expect("tile should be within the source matrix");

        assert_eq!(tile.shape().dims(), &[2, 3]);
        assert_eq!(tile.as_slice(), &[6.0, 7.0, 8.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn rejects_a_tile_outside_the_source_matrix() {
        let source = HostTensor::<f32, 2>::zeros(Shape::new([4, 5]));

        assert!(source.copy_tile([3, 4], Shape::new([2, 2])).is_err());
    }
}
