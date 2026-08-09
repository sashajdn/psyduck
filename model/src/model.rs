use tensor::{MatrixTensor, NaiveTensor, QuantizedFp, Shape, Tensor};

#[derive(thiserror::Error, Debug)]
pub enum ModelError {
    #[error(transparent)]
    ShapeMismatch(#[from] tensor::ShapeMismatch),
}

pub trait ModelBackend<F: QuantizedFp<F>> {
    type Tensor<const R: usize>: Tensor<F, R>;

    fn upload<const R: usize>(target: &[F], shape: Shape<R>) -> Self::Tensor<R>;
    fn download<const R: usize>(source: &Self::Tensor<R>, target: &mut [F]) -> Self::Tensor<R>;
    fn alloc<const R: usize>(shape: Shape<R>) -> Self::Tensor<R>;

    // Primitives.
    fn try_matmul(
        a: &Self::Tensor<2>,
        b: &Self::Tensor<2>,
        target: &Self::Tensor<2>,
    ) -> Result<(), ModelError>;
}

pub struct Model<B: ModelBackend<F>, F: QuantizedFp<F>, const R: usize> {
    _phantom: std::marker::PhantomData<(B, F)>,
}

impl<B: ModelBackend<F>, F: QuantizedFp<F>, const R: usize> Model<B, F, R> {
    fn load(weights: B::Tensor<R>) {
        todo!();
    }

    fn forward(&self, input: B::Tensor<R>) -> B::Tensor<R> {
        todo!();
    }
}

pub struct NaiveModelBackend<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: QuantizedFp<F>> ModelBackend<F> for NaiveModelBackend<F> {
    type Tensor<const R: usize> = NaiveTensor<F, R>;

    fn upload<const R: usize>(target: &[F], shape: Shape<R>) -> Self::Tensor<R> {
        todo!()
    }

    fn download<const R: usize>(source: &Self::Tensor<R>, target: &mut [F]) -> Self::Tensor<R> {
        todo!()
    }

    fn alloc<const R: usize>(shape: Shape<R>) -> Self::Tensor<R> {
        todo!()
    }

    fn try_matmul(
        a: &Self::Tensor<2>,
        b: &Self::Tensor<2>,
        target: &Self::Tensor<2>,
    ) -> Result<(), ModelError> {
        a.validate_matmul_shape(b)?;

        todo!()
    }
}
