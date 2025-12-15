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
    let name = format!("ci.riscv.benchmark.{kernel_name}");
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();

    let benchmark_value = MetricPoint::new()
        .timestamp(timestamp as i64)
        .value(mean_tps);

    let series = MetricSeries::new(name, vec![benchmark_value])
        .type_(MetricIntakeType::GAUGE)
        .tags(vec![transaction_kind.to_string()])
        .unit(format!("TPS ({transaction_kind})"));

    let payload = MetricPayload::new(vec![series]);

    let configuration = datadog::Configuration::new();
    let api = MetricsAPI::with_config(configuration);

    api.submit_metrics(
        payload,
        SubmitMetricsOptionalParams::default().content_encoding(MetricContentEncoding::ZSTD1),
    )
    .await?;

    Ok(())
}
