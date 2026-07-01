use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, quote};
use std::cmp::Ordering;
use syn::{
    Expr, Ident, Lit, LitBool, LitStr, Token,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    spanned::Spanned,
};

#[derive(Clone, Debug)]
pub struct RegistryArgs {
    pub(crate) metrics: Punctuated<Metric, Token![,]>,
    pub(crate) meter_fn: Option<LitStr>,
    pub(crate) decorated: bool,
    pub(crate) gated: bool,
}

#[derive(Clone, Debug)]
pub struct Metric {
    pub(crate) name: String,
    pub(crate) description: Option<String>,
    pub(crate) unit: Option<String>,
    pub(crate) r#type: MetricType,
    pub(crate) boundaries: Option<Vec<f64>>,
}

impl ToTokens for Metric {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let description = self.description.as_deref().unwrap_or("<no description>");
        let unit = self.unit.as_deref().unwrap_or("<no unit>");
        let name = &self.name;
        let r#type = Ident::from(self.r#type.clone());

        let mut builder = quote! {
            meter
                .#r#type(#name)
                .with_unit(#unit)
                .with_description(#description)
        };

        if self.r#type.is_histogram()
            && let Some(bounds) = &self.boundaries
        {
            builder = quote! { #builder.with_boundaries(vec![ #( #bounds ),* ]) };
        }

        tokens.extend(quote! { #builder.build() });
    }
}

#[derive(Clone, Debug)]
pub enum MetricType {
    U64Counter,
    F64Counter,
    U64ObservableCounter,
    F64ObservableCounter,
    I64UpDownCounter,
    F64UpDownCounter,
    I64ObservableUpDownCounter,
    F64ObservableUpDownCounter,
    U64ObservableGauge,
    I64ObservableGauge,
    F64ObservableGauge,
    U64Gauge,
    I64Gauge,
    F64Gauge,
    F64Histogram,
    U64Histogram,
}

impl MetricType {
    pub(crate) fn is_histogram(&self) -> bool {
        matches!(self, Self::F64Histogram | Self::U64Histogram)
    }

    pub(crate) fn to_otel_token_stream(&self, krate: &TokenStream) -> TokenStream {
        match self {
            Self::U64Counter => quote! { #krate::opentelemetry::metrics::Counter<u64> },
            Self::F64Counter => quote! { #krate::opentelemetry::metrics::Counter<f64> },
            Self::U64ObservableCounter => {
                quote! { #krate::opentelemetry::metrics::ObservableCounter<u64> }
            }
            Self::F64ObservableCounter => {
                quote! { #krate::opentelemetry::metrics::ObservableCounter<f64> }
            }
            Self::I64UpDownCounter => quote! { #krate::opentelemetry::metrics::UpDownCounter<i64> },
            Self::F64UpDownCounter => quote! { #krate::opentelemetry::metrics::UpDownCounter<f64> },
            Self::I64ObservableUpDownCounter => {
                quote! { #krate::opentelemetry::metrics::ObservableUpDownCounter<i64> }
            }
            Self::F64ObservableUpDownCounter => {
                quote! { #krate::opentelemetry::metrics::ObservableUpDownCounter<f64> }
            }
            Self::U64ObservableGauge => {
                quote! { #krate::opentelemetry::metrics::ObservableGauge<u64> }
            }
            Self::I64ObservableGauge => {
                quote! { #krate::opentelemetry::metrics::ObservableGauge<i64> }
            }
            Self::F64ObservableGauge => {
                quote! { #krate::opentelemetry::metrics::ObservableGauge<f64> }
            }
            Self::U64Gauge => quote! { #krate::opentelemetry::metrics::Gauge<u64> },
            Self::I64Gauge => quote! { #krate::opentelemetry::metrics::Gauge<i64> },
            Self::F64Gauge => quote! { #krate::opentelemetry::metrics::Gauge<f64> },
            Self::F64Histogram => quote! { #krate::opentelemetry::metrics::Histogram<f64> },
            Self::U64Histogram => quote! { #krate::opentelemetry::metrics::Histogram<u64> },
        }
    }
}

impl TryFrom<Ident> for MetricType {
    type Error = syn::Error;

    fn try_from(ident: Ident) -> Result<Self, Self::Error> {
        match ident.to_string().as_str() {
            "u64_counter" => Ok(Self::U64Counter),
            "f64_counter" => Ok(Self::F64Counter),
            "u64_observable_counter" => Ok(Self::U64ObservableCounter),
            "f64_observable_counter" => Ok(Self::F64ObservableCounter),
            "i64_up_down_counter" => Ok(Self::I64UpDownCounter),
            "f64_up_down_counter" => Ok(Self::F64UpDownCounter),
            "i64_observable_up_down_counter" => Ok(Self::I64ObservableUpDownCounter),
            "f64_observable_up_down_counter" => Ok(Self::F64ObservableUpDownCounter),
            "u64_observable_gauge" => Ok(Self::U64ObservableGauge),
            "i64_observable_gauge" => Ok(Self::I64ObservableGauge),
            "f64_observable_gauge" => Ok(Self::F64ObservableGauge),
            "u64_gauge" => Ok(Self::U64Gauge),
            "i64_gauge" => Ok(Self::I64Gauge),
            "f64_gauge" => Ok(Self::F64Gauge),
            "f64_histogram" => Ok(Self::F64Histogram),
            "u64_histogram" => Ok(Self::U64Histogram),
            unknown => Err(syn::Error::new(
                ident.span(),
                format!("unknown instrument `{unknown}`"),
            )),
        }
    }
}

impl From<MetricType> for Ident {
    fn from(metric_type: MetricType) -> Self {
        match metric_type {
            MetricType::U64Counter => Self::new("u64_counter", Span::call_site()),
            MetricType::F64Counter => Self::new("f64_counter", Span::call_site()),
            MetricType::U64ObservableCounter => {
                Self::new("u64_observable_counter", Span::call_site())
            }
            MetricType::F64ObservableCounter => {
                Self::new("f64_observable_counter", Span::call_site())
            }
            MetricType::I64UpDownCounter => Self::new("i64_up_down_counter", Span::call_site()),
            MetricType::F64UpDownCounter => Self::new("f64_up_down_counter", Span::call_site()),
            MetricType::I64ObservableUpDownCounter => {
                Self::new("i64_observable_up_down_counter", Span::call_site())
            }
            MetricType::F64ObservableUpDownCounter => {
                Self::new("f64_observable_up_down_counter", Span::call_site())
            }
            MetricType::U64ObservableGauge => Self::new("u64_observable_gauge", Span::call_site()),
            MetricType::I64ObservableGauge => Self::new("i64_observable_gauge", Span::call_site()),
            MetricType::F64ObservableGauge => Self::new("f64_observable_gauge", Span::call_site()),
            MetricType::U64Gauge => Self::new("u64_gauge", Span::call_site()),
            MetricType::I64Gauge => Self::new("i64_gauge", Span::call_site()),
            MetricType::F64Gauge => Self::new("f64_gauge", Span::call_site()),
            MetricType::F64Histogram => Self::new("f64_histogram", Span::call_site()),
            MetricType::U64Histogram => Self::new("u64_histogram", Span::call_site()),
        }
    }
}

impl Parse for RegistryArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut args = Self {
            metrics: Default::default(),
            meter_fn: None,
            decorated: false,
            gated: false,
        };

        while !input.is_empty() {
            let lookahead = input.lookahead1();

            if lookahead.peek(kw::metrics) {
                if !args.metrics.is_empty() {
                    return Err(input.error("expected only a single `metrics` argument"));
                }
                let Metrics(metrics) = input.parse()?;
                args.metrics = metrics;
            } else if lookahead.peek(kw::meter_fn) {
                if args.meter_fn.is_some() {
                    return Err(input.error("expected only a single `meter_fn` argument"));
                }
                let target = input.parse::<super::ArgType<LitStr, kw::meter_fn>>()?.value;
                args.meter_fn = Some(target);
            } else if lookahead.peek(kw::decorated) {
                let target = input
                    .parse::<super::ArgType<LitBool, kw::decorated>>()?
                    .value;
                args.decorated = target.value;
            } else if lookahead.peek(kw::gated) {
                let target = input.parse::<super::ArgType<LitBool, kw::gated>>()?.value;
                args.gated = target.value;
            } else if lookahead.peek(Token![,]) {
                let _ = input.parse::<Token![,]>()?;
            } else {
                return Err(lookahead.error());
            }
        }

        Ok(args)
    }
}

struct Metrics(Punctuated<Metric, Token![,]>);

impl Parse for Metrics {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let _ = input.parse::<kw::metrics>();
        let content;
        let _ = syn::parenthesized!(content in input);
        let metrics = content.parse_terminated(Metric::parse, Token![,])?;
        Ok(Self(metrics))
    }
}

impl Parse for Metric {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let content;
        syn::braced!(content in input);

        let mut name = None;
        let mut description = None;
        let mut unit = None;
        let mut r#type = None;
        let mut boundaries = None;

        while !content.is_empty() {
            let ident: Ident = content.parse()?;
            let _: Token![=] = content.parse()?;
            let value: Expr = content.parse()?;
            let _ = content.parse::<Token![,]>();

            match ident.to_string().as_str() {
                "name" => {
                    if let Expr::Lit(expr_lit) = value
                        && let Lit::Str(lit_str) = expr_lit.lit
                    {
                        name = Some(lit_str.value());
                    }
                }
                "instrument" => {
                    if let Expr::Path(expr_path) = value {
                        let segment = expr_path.path.segments.first().ok_or_else(|| {
                            syn::Error::new(expr_path.span(), "missing instrument")
                        })?;
                        r#type = Some(MetricType::try_from(segment.ident.clone())?);
                    }
                }
                "description" => {
                    if let Expr::Lit(expr_lit) = value
                        && let Lit::Str(lit_str) = expr_lit.lit
                    {
                        description = Some(lit_str.value());
                    }
                }
                "unit" => {
                    if let Expr::Lit(expr_lit) = value
                        && let Lit::Str(lit_str) = expr_lit.lit
                    {
                        unit = Some(lit_str.value());
                    }
                }
                "boundaries" => {
                    let arr = parse_float_array(value)?;
                    validate_monotonic_increasing(&arr)?;
                    boundaries = Some(arr);
                }
                token => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!("unknown metric attribute `{token}`"),
                    ));
                }
            }
        }

        let r#type = r#type
            .ok_or_else(|| syn::Error::new(input.span(), "missing `instrument` attribute"))?;

        if boundaries.is_some() && !r#type.is_histogram() {
            return Err(syn::Error::new(
                input.span(),
                "`boundaries` is only valid for histogram instruments",
            ));
        }

        Ok(Self {
            name: name.ok_or_else(|| syn::Error::new(input.span(), "missing `name` attribute"))?,
            description,
            unit,
            r#type,
            boundaries,
        })
    }
}

fn parse_float_array(expr: Expr) -> syn::Result<Vec<f64>> {
    match expr {
        Expr::Array(array) => {
            let mut out = Vec::with_capacity(array.elems.len());
            for elem in array.elems {
                match elem {
                    Expr::Lit(l) => match l.lit {
                        Lit::Float(f) => out.push(f.base10_parse::<f64>()?),
                        Lit::Int(i) => out.push(i.base10_parse::<f64>()?),
                        other => {
                            return Err(syn::Error::new(
                                other.span(),
                                "boundaries must be numeric literals",
                            ));
                        }
                    },
                    _ => {
                        return Err(syn::Error::new(
                            elem.span(),
                            "boundaries must be numeric literals",
                        ));
                    }
                }
            }
            Ok(out)
        }
        _ => Err(syn::Error::new(
            expr.span(),
            "`boundaries` must be an array literal, e.g. [0.0, 0.1, 0.2]",
        )),
    }
}

fn validate_monotonic_increasing(values: &[f64]) -> syn::Result<()> {
    if values.is_empty() {
        return Err(syn::Error::new(
            Span::call_site(),
            "boundaries cannot be empty",
        ));
    }

    for window in values.windows(2) {
        let first = window[0];
        let second = window[1];

        if !first.is_finite() || !second.is_finite() {
            return Err(syn::Error::new(
                Span::call_site(),
                "boundaries must be finite",
            ));
        }

        match first.partial_cmp(&second) {
            Some(Ordering::Less) => {}
            Some(Ordering::Greater) | Some(Ordering::Equal) => {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "boundaries must be strictly increasing",
                ));
            }
            None => {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "boundaries contain unordered values; use only finite numbers",
                ));
            }
        }
    }

    Ok(())
}

mod kw {
    syn::custom_keyword!(metrics);
    syn::custom_keyword!(meter_fn);
    syn::custom_keyword!(decorated);
    syn::custom_keyword!(gated);
}
