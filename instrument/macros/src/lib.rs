use proc_macro_crate::{FoundCrate, crate_name};
use quote::{format_ident, quote};

mod attributes;
mod expand;

fn instrument_crate_path() -> proc_macro2::TokenStream {
    match crate_name("instrument") {
        Ok(FoundCrate::Itself) => quote! { crate },
        Ok(FoundCrate::Name(name)) => {
            let ident = format_ident!("{}", name);
            quote! { ::#ident }
        }
        Err(_) => quote! { ::instrument },
    }
}

#[proc_macro_attribute]
pub fn registry(
    args: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let item_struct = syn::parse_macro_input!(item as syn::ItemStruct);
    let args = syn::parse_macro_input!(args as attributes::registry::RegistryArgs);

    expand::registry::gen_registry(item_struct, args)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
