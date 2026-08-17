pub mod benchmark;
pub mod operation;

pub use opentelemetry;
pub use opentelemetry_prometheus;
pub use opentelemetry_sdk;
pub use prometheus;

#[cfg(feature = "macros")]
pub use instrument_macros::registry;

pub mod prometheus_exporter {
    use opentelemetry_sdk::metrics::SdkMeterProvider;
    use prometheus::Registry;

    pub fn meter_provider()
    -> Result<(Registry, SdkMeterProvider), opentelemetry_sdk::error::OTelSdkError> {
        let registry = Registry::new();
        let exporter = opentelemetry_prometheus::exporter()
            .with_registry(registry.clone())
            .build()?;
        let provider = SdkMeterProvider::builder().with_reader(exporter).build();

        Ok((registry, provider))
    }
}
