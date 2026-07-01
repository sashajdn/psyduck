use syn::{
    Token,
    parse::{Parse, ParseStream},
};

pub mod registry;

struct ArgType<P, T> {
    value: P,
    _p: std::marker::PhantomData<T>,
}

impl<P: Parse, T: Parse> Parse for ArgType<P, T> {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let _ = input.parse::<T>()?;
        let _ = input.parse::<Token![=]>()?;
        let value = input.parse()?;

        Ok(Self {
            value,
            _p: std::marker::PhantomData,
        })
    }
}
