use tensor::{MatrixTensor, NaiveTensor, QuantizedFp, Shape};

use crate::model::{ModelBackend, ModelError};

pub struct NaiveModelBackend<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F> NaiveModelBackend<F> {
    pub const fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F> Default for NaiveModelBackend<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: QuantizedFp> ModelBackend<F> for NaiveModelBackend<F> {
    type Tensor<const R: usize> = NaiveTensor<F, R>;

    #[inline]
    fn upload_htod<const R: usize>(
        &self,
        source: &NaiveTensor<F, R>,
    ) -> Result<Self::Tensor<R>, ModelError> {
        Ok(source.clone())
    }

    #[inline]
    fn download_dtoh<const R: usize>(
        &self,
        source: &Self::Tensor<R>,
    ) -> Result<NaiveTensor<F, R>, ModelError> {
        Ok(source.clone())
    }

    #[inline]
    fn alloc<const R: usize>(&self, shape: Shape<R>) -> Result<Self::Tensor<R>, ModelError> {
        Ok(NaiveTensor::zeros(shape))
    }

    fn try_matmul<const BLOCK_X: u32, const BLOCK_Y: u32, const THREADS_PER_BLOCK: u32>(
        &self,
        _a: &Self::Tensor<2>,
        _b: &Self::Tensor<2>,
        _target: &mut Self::Tensor<2>,
    ) -> Result<(), ModelError> {
        unimplemented!("Implement try_matmul");
    }

    fn try_add<const THREADS_PER_BLOCK: u32>(
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
}

#[cfg(test)]
mod tests {
    use tensor::{NaiveTensor, Shape};

    use super::{ModelBackend, NaiveModelBackend};

    fn matrix(values: &[f32; 4]) -> NaiveTensor<f32, 2> {
        let mut tensor = NaiveTensor::zeros(Shape::new([2, 2]));
        tensor.as_mut_slice().copy_from_slice(values);
        tensor
    }

    #[test]
    fn correctly_adds_two_matrices() {
        let backend = NaiveModelBackend::<f32>::new();
        let a = matrix(&[1.0, 2.0, 3.0, 4.0]);
        let b = matrix(&[5.0, 6.0, 7.0, 8.0]);
        let mut target = NaiveTensor::zeros(Shape::new([2, 2]));

        backend
            .try_add::<256>(&a, &b, &mut target)
            .expect("2x2 matrices should be addable");

        assert_eq!(target.as_slice(), &[6.0, 8.0, 10.0, 12.0]);
    }
}
