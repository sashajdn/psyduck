use instrument::{
    opentelemetry::metrics::{Meter, MeterProvider},
    registry,
};
use std::sync::OnceLock;

static METER: OnceLock<Meter> = OnceLock::new();

fn test_meter() -> Meter {
    METER.get().unwrap().clone()
}

#[registry(
    metrics(
        {
            name = "tokens_total",
            instrument = u64_counter,
            description = "generated token count",
        },
        {
            name = "decode_latency",
            instrument = f64_histogram,
            unit = "milliseconds",
            boundaries = [0.1, 0.5, 1.0, 5.0],
        }
    ),
    meter_fn = "test_meter",
)]
struct DecodeMetrics;

#[test]
fn registry_records_to_prometheus() {
    let (prometheus_registry, provider) =
        instrument::prometheus_exporter::meter_provider().unwrap();
    METER
        .set(provider.meter("registry_records_to_prometheus"))
        .unwrap();

    DecodeMetrics::metrics().tokens_total.add(2, &[]);
    DecodeMetrics::metrics().decode_latency.record(0.7, &[]);

    let families = prometheus_registry.gather();

    assert!(
        families
            .iter()
            .any(|family| family.name().starts_with("tokens_total"))
    );
    assert!(
        families
            .iter()
            .any(|family| family.name().starts_with("decode_latency"))
    );
}
