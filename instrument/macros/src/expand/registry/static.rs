use crate::attributes::registry::RegistryArgs;
use convert_case::{Case, Casing};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Error, ItemStruct};

pub fn gen_static_registry(
    item_struct: &ItemStruct,
    args: &RegistryArgs,
) -> Result<TokenStream, Error> {
    let registry = if args.gated {
        gen_gated_registry(item_struct, args)?
    } else {
        gen_registry(item_struct, args)?
    };

    Ok(quote! {
        #item_struct

        #registry
    })
}

fn gen_registry(item_struct: &ItemStruct, args: &RegistryArgs) -> Result<TokenStream, Error> {
    let registry_struct = item_struct.ident.clone();
    let registry_struct_name = registry_struct.to_string();
    let (impl_generics, ty_generics, where_clause) = item_struct.generics.split_for_impl();

    let (registry_ident, registry_definition) =
        super::gen_instrument_registry(&registry_struct_name, args);
    let meter = gen_meter(&registry_struct_name, args);
    let registry_lock = format_ident!(
        "{}_INIT",
        registry_ident.to_string().to_case(Case::UpperSnake)
    );

    Ok(quote! {
        #registry_definition

        impl #impl_generics #registry_struct #ty_generics #where_clause {
            pub fn metrics() -> &'static #registry_ident {
                static #registry_lock: std::sync::OnceLock<#registry_ident> = std::sync::OnceLock::new();

                #registry_lock.get_or_init(|| {
                    #meter
                    #registry_ident::new(meter)
                })
            }
        }
    })
}

fn gen_gated_registry(item_struct: &ItemStruct, args: &RegistryArgs) -> Result<TokenStream, Error> {
    let krate = crate::instrument_crate_path();
    let registry_struct = item_struct.ident.clone();
    let registry_struct_name = registry_struct.to_string();
    let (impl_generics, ty_generics, where_clause) = item_struct.generics.split_for_impl();

    let (registry_ident, registry_definition) =
        super::gen_instrument_registry(&registry_struct_name, args);
    let meter = gen_meter(&registry_struct_name, args);

    let screaming = registry_ident.to_string().to_case(Case::UpperSnake);
    let enabled_flag = format_ident!("{}_METRICS_ENABLED", screaming);
    let registry_lock = format_ident!("{}_INIT", screaming);
    let noop_registry_lock = format_ident!("{}_NOOP_INIT", screaming);

    Ok(quote! {
        #registry_definition

        static #enabled_flag: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);

        impl #impl_generics #registry_struct #ty_generics #where_clause {
            pub fn metrics() -> &'static #registry_ident {
                if #enabled_flag.load(std::sync::atomic::Ordering::Relaxed) {
                    static #registry_lock: std::sync::OnceLock<#registry_ident> = std::sync::OnceLock::new();
                    #registry_lock.get_or_init(|| {
                        #meter
                        #registry_ident::new(meter)
                    })
                } else {
                    static #noop_registry_lock: std::sync::OnceLock<#registry_ident> = std::sync::OnceLock::new();
                    #noop_registry_lock.get_or_init(|| {
                        use #krate::opentelemetry::metrics::MeterProvider as _;
                        let meter = #krate::opentelemetry_sdk::metrics::SdkMeterProvider::builder()
                            .build()
                            .meter("noop");
                        #registry_ident::new(meter)
                    })
                }
            }

            pub fn enable_metrics() {
                #enabled_flag.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
    })
}

fn gen_meter(struct_name: &str, args: &RegistryArgs) -> TokenStream {
    let krate = crate::instrument_crate_path();
    args.clone()
        .meter_fn
        .map(|meter_fn| {
            let meter_fn = format_ident!("{}", meter_fn.value());
            quote! {
                let meter = #meter_fn();
            }
        })
        .unwrap_or_else(|| {
            quote! {
                let meter = #krate::opentelemetry::global::meter(#struct_name);
            }
        })
}
