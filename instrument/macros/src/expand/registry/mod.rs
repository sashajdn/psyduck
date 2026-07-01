use crate::attributes::registry::RegistryArgs;
use proc_macro2::{Ident, TokenStream};
use quote::{ToTokens, format_ident, quote};
use syn::{Error, ItemStruct};

mod decorated;
mod r#static;

pub fn gen_registry(mut item_struct: ItemStruct, args: RegistryArgs) -> Result<TokenStream, Error> {
    if args.decorated {
        decorated::gen_decorated_registry(&mut item_struct, &args)
    } else {
        r#static::gen_static_registry(&item_struct, &args)
    }
}

fn gen_instrument_registry(name: &str, args: &RegistryArgs) -> (Ident, TokenStream) {
    let krate = crate::instrument_crate_path();
    let registry_struct_ident = format_ident!("{}MetricsRegistry", name);

    let fields = args
        .metrics
        .iter()
        .map(|metric| {
            let name = Ident::new(&metric.name, registry_struct_ident.span());
            let r#type = metric.r#type.to_otel_token_stream(&krate);
            quote! {
                pub #name: #r#type
            }
        })
        .collect::<Vec<_>>();

    let fields_creation = args
        .metrics
        .iter()
        .map(|metric| {
            let name = Ident::new(&metric.name, registry_struct_ident.span());
            let new_field = metric.to_token_stream();
            quote! {
                #name: #new_field
            }
        })
        .collect::<Vec<_>>();

    let registry_struct_definition = quote! {
        pub struct #registry_struct_ident {
            #(#fields),*
        }

        impl #registry_struct_ident {
            pub fn new(meter: #krate::opentelemetry::metrics::Meter) -> Self {
                Self {
                    #(#fields_creation),*
                }
            }
        }
    };

    (registry_struct_ident, registry_struct_definition)
}
