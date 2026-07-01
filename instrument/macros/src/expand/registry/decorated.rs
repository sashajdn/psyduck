use crate::attributes::registry::RegistryArgs;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{Error, Field, Fields, FieldsNamed, ItemStruct, parse::Parser, spanned::Spanned};

pub fn gen_decorated_registry(
    item_struct: &mut ItemStruct,
    args: &RegistryArgs,
) -> Result<TokenStream, Error> {
    if args.meter_fn.is_some() {
        return Err(Error::new(
            item_struct.span(),
            "meter_fn is not supported on decorated registries",
        ));
    }

    if args.gated {
        return Err(Error::new(
            item_struct.span(),
            "gated is not supported on decorated registries",
        ));
    }

    let (registry_field, registry_definition) = gen_registry(item_struct, args);

    match &mut item_struct.fields {
        Fields::Named(fields) => {
            fields.named.push(registry_field);
        }
        Fields::Unit => {
            item_struct.fields = Fields::Named(FieldsNamed {
                brace_token: Default::default(),
                named: Default::default(),
            });

            match &mut item_struct.fields {
                Fields::Named(fields_named) => {
                    fields_named.named.push(registry_field);
                }
                _ => unreachable!(),
            };
        }
        Fields::Unnamed(un) => {
            return Err(Error::new(
                un.span(),
                "decorated registries do not support tuple structs",
            ));
        }
    }

    Ok(quote! {
        #item_struct

        #registry_definition
    })
}

fn gen_registry(item_struct: &ItemStruct, args: &RegistryArgs) -> (Field, TokenStream) {
    let (custom_registry_ident, custom_registry_fields) =
        super::gen_instrument_registry(&item_struct.ident.to_string(), args);

    let field = Field::parse_named
        .parse2(quote! { pub metrics: #custom_registry_ident })
        .unwrap();

    (field, custom_registry_fields)
}
