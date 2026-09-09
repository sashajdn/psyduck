pub(crate) trait OutputElementWiseModifier<F> {
    fn apply(target: &mut F, value: F);
}

pub(crate) struct Overwrite;
pub(crate) struct Accumulate;

impl<F> OutputElementWiseModifier<F> for Overwrite {
    #[inline(always)]
    fn apply(target: &mut F, value: F) {
        *target = value;
    }
}

impl<F> OutputElementWiseModifier<F> for Accumulate
where
    F: std::ops::AddAssign,
{
    #[inline(always)]
    fn apply(target: &mut F, value: F) {
        *target += value;
    }
}
