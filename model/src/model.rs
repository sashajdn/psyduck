use tensor::{MatrixTensor, NaiveTensor, QuantizedFp, Shape, Tensor};

#[derive(thiserror::Error, Debug)]
pub enum ModelError {
    #[error(transparent)]
    ShapeMismatch(#[from] tensor::ShapeMismatch),
}

pub trait ModelBackend<F: QuantizedFp> {
    type Tensor<const R: usize>: Tensor<F, R>;

    fn upload<const R: usize>(target: &[F], shape: Shape<R>) -> Self::Tensor<R>;
    fn download<const R: usize>(source: &Self::Tensor<R>, target: &mut [F]) -> Self::Tensor<R>;
    fn alloc<const R: usize>(shape: Shape<R>) -> Self::Tensor<R>;

    // Primitives.
    fn try_matmul(
        a: &Self::Tensor<2>,
        b: &Self::Tensor<2>,
        target: &mut Self::Tensor<2>,
    ) -> Result<(), ModelError>;

    fn try_add(
        a: &Self::Tensor<2>,
        b: &Self::Tensor<2>,
        target: &mut Self::Tensor<2>,
    ) -> Result<(), ModelError>;
}

pub struct NaiveModelBackend<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: QuantizedFp> ModelBackend<F> for NaiveModelBackend<F> {
    type Tensor<const R: usize> = NaiveTensor<F, R>;

    fn upload<const R: usize>(_target: &[F], _shape: Shape<R>) -> Self::Tensor<R> {
        unimplemented!("implement upload for NaiveModelBackend");
    }

    fn download<const R: usize>(_source: &Self::Tensor<R>, _target: &mut [F]) -> Self::Tensor<R> {
        unimplemented!("implement download for NaiveModelBackend");
    }

    fn alloc<const R: usize>(_shape: Shape<R>) -> Self::Tensor<R> {
        unimplemented!("implement alloc for NaiveModelBackend");
    }

    fn try_matmul(
        a: &Self::Tensor<2>,
        b: &Self::Tensor<2>,
        _target: &mut Self::Tensor<2>,
    ) -> Result<(), ModelError> {
        unimplemented!("Implement try_matmul");
    }

    fn try_add(
        a: &Self::Tensor<2>,
        b: &Self::Tensor<2>,
        target: &mut Self::Tensor<2>,
    ) -> Result<(), ModelError> {
        a.validate_add_shape(b)?;
        a.validate_add_shape(target)?;

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
    use tensor::{NaiveTensor, Shape, Tensor};

    use super::{ModelBackend, NaiveModelBackend};

    fn matrix(values: &[f32; 4]) -> NaiveTensor<f32, 2> {
        let mut tensor = NaiveTensor::zeros(Shape::new([2, 2]));
        tensor.as_mut_slice().copy_from_slice(values);
        tensor
    }

    #[test]
    fn correctly_adds_two_matrices() {
        let a = matrix(&[1.0, 2.0, 3.0, 4.0]);
        let b = matrix(&[5.0, 6.0, 7.0, 8.0]);
        let mut target = NaiveTensor::zeros(Shape::new([2, 2]));

        NaiveModelBackend::<f32>::try_add(&a, &b, &mut target)
            .expect("2x2 matrices should be addable");

        assert_eq!(target.as_slice(), &[6.0, 8.0, 10.0, 12.0]);
    }
}
