// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Helper functionality enabling submission of kernel benchmark metrics to datadog.
//!
//! Documentation is available here: <https://docs.datadoghq.com/api/latest/metrics/#submit-metrics>

use std::error::Error;
use std::time::SystemTime;

use datadog_api_client::datadog;
use datadog_api_client::datadogV2::api_metrics::MetricsAPI;
use datadog_api_client::datadogV2::api_metrics::SubmitMetricsOptionalParams;
use datadog_api_client::datadogV2::model::MetricContentEncoding;
use datadog_api_client::datadogV2::model::MetricIntakeType;
use datadog_api_client::datadogV2::model::MetricPayload;
use datadog_api_client::datadogV2::model::MetricPoint;
use datadog_api_client::datadogV2::model::MetricSeries;

/// Submit a recorded tps benchmark to datadog as a metric.
///
/// This can then be tracked as in a graph, with alerting if performance drops.
/// Different kernels correspond to different metrics, as do different transaction
/// kinds (e.g. `simple transfer, fa2-style` etc).
///
/// Care must be taken to limit the number of transaction kinds and/or kernels for this
/// reason.
pub async fn submit_kernel_tps_benchmark(
    kernel_name: &str,
    transaction_kind: &str,
    mean_tps: f64,
) -> Result<(), Box<dyn Error>> {
    submit_gauges(
        &[Gauge {
            name: format!("ci.riscv.benchmark.{kernel_name}"),
            value: mean_tps,
            tags: vec![transaction_kind.to_string()],
            unit: format!("TPS ({transaction_kind})"),
        }],
        SystemTime::now(),
    )
    .await
}

/// One gauge to record against a point in time.
pub struct Gauge {
    /// Metric name, by convention prefixed `ci.riscv.`.
    ///
    /// Keep the set of names small and stable: a name is a time series, and one that changes
    /// between runs cannot be graphed or alerted on.
    pub name: String,

    /// Value at this point in time.
    pub value: f64,

    /// Tags to record alongside, for splitting one metric by e.g. the shape it was measured at.
    pub tags: Vec<String>,

    /// Human-readable unit, shown on graphs.
    pub unit: String,
}

/// Submit gauges to datadog, all stamped with the same time.
///
/// One request carries the whole batch, so a run's metrics land together and share a timestamp,
/// which matters when they are graphed against each other.
pub async fn submit_gauges(gauges: &[Gauge], at: SystemTime) -> Result<(), Box<dyn Error>> {
    if gauges.is_empty() {
        return Ok(());
    }

    let timestamp = at.duration_since(SystemTime::UNIX_EPOCH)?.as_secs() as i64;

    let series = gauges
        .iter()
        .map(|gauge| {
            let point = MetricPoint::new().timestamp(timestamp).value(gauge.value);

            MetricSeries::new(gauge.name.clone(), vec![point])
                .type_(MetricIntakeType::GAUGE)
                .tags(gauge.tags.clone())
                .unit(gauge.unit.clone())
        })
        .collect();

    let payload = MetricPayload::new(series);

    let configuration = datadog::Configuration::new();
    let api = MetricsAPI::with_config(configuration);

    // Use GZIP. We don't pull in the zstd feature, but by default datadog-api-client still
    // sets the header as ZSTD1, but silently doesn't compress it. Instead, we can use
    // gzip, which is available.
    api.submit_metrics(
        payload,
        SubmitMetricsOptionalParams::default().content_encoding(MetricContentEncoding::GZIP),
    )
    .await?;

    Ok(())
}
