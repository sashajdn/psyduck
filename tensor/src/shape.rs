#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Shape<const R: usize> {
    dims: [usize; R],
}

impl<const R: usize> Shape<R> {
    #[inline]
    pub const fn new(dims: [usize; R]) -> Self {
        Self { dims }
    }

    #[inline]
    pub const fn dims(&self) -> &[usize; R] {
        &self.dims
    }

    #[inline]
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
}

impl Shape<2> {
    #[inline]
    pub const fn rows(&self) -> usize {
        self.dims[0]
    }

    #[inline]
    pub const fn cols(&self) -> usize {
        self.dims[1]
    }
}

impl<const R: usize> From<[usize; R]> for Shape<R> {
    #[inline]
    fn from(dims: [usize; R]) -> Self {
        Self::new(dims)
    }
}
