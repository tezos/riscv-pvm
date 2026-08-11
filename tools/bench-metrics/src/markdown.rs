// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Renders gauges as a markdown table, for a pull request comment.

use comfy_table::Table;
use comfy_table::presets::ASCII_MARKDOWN;
use kernel_bench_utils::datadog::Gauge;

use crate::PREFIX;

/// Render the gauges as a markdown table.
///
/// Two things are dropped because they are the same on every row and so say nothing: the metric
/// prefix, and the tags when one run produced them all. A run that measured several shapes or
/// operations keeps the tags, since that is then the only thing telling the rows apart.
pub(crate) fn table(gauges: &[Gauge]) -> String {
    let tagged = gauges.iter().any(|gauge| gauge.tags != gauges[0].tags);

    let mut table = Table::new();
    table.load_style(ASCII_MARKDOWN);

    let mut header = vec!["Metric", "Value", "Unit"];

    if tagged {
        header.push("Tags");
    }

    table.set_header(header);

    for gauge in gauges {
        let (value, unit) = display(gauge);
        let mut row = vec![
            gauge
                .name
                .trim_start_matches(PREFIX)
                .trim_start_matches('.')
                .to_owned(),
            value,
            unit.to_owned(),
        ];

        if tagged {
            row.push(gauge.tags.join(", "));
        }

        table.add_row(row);
    }

    table.to_string()
}

/// How a gauge's value reads to a person, as the value and the unit it is now in.
///
/// Byte counts become MiB: in bytes they run to eight digits, and nobody compares those across
/// two comments.
fn display(gauge: &Gauge) -> (String, &str) {
    match gauge.unit.as_str() {
        "bytes" => (format!("{:.2}", gauge.value / (1024.0 * 1024.0)), "MiB"),
        "percent" => (format!("{:.1}", gauge.value), "%"),
        unit => (format!("{:.3}", gauge.value), unit),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauge(name: &str, value: f64, unit: &str, tags: &[&str]) -> Gauge {
        Gauge {
            name: format!("{PREFIX}.{name}"),
            value,
            tags: tags.iter().map(|tag| (*tag).to_owned()).collect(),
            unit: unit.to_owned(),
        }
    }

    /// One run of one shape: the tags are the same on every row, so the column is left out and
    /// bytes are shown in MiB.
    #[test]
    fn untagged_table_drops_the_tags_column() {
        let rendered = table(&[
            gauge(
                "repo_bytes_per_commit",
                2.0 * 1024.0 * 1024.0,
                "bytes",
                &["shape:pr"],
            ),
            gauge("dead_node_fraction", 33.25, "percent", &["shape:pr"]),
        ]);

        assert!(rendered.contains("| Metric"), "{rendered}");
        assert!(!rendered.contains("Tags"), "{rendered}");
        assert!(rendered.contains("| repo_bytes_per_commit"), "{rendered}");
        assert!(rendered.contains("2.00"), "{rendered}");
        assert!(rendered.contains("MiB"), "{rendered}");
        assert!(rendered.contains("33.2"), "{rendered}");
    }

    /// Rows that differ only in their tags keep them: without the column they would read as
    /// duplicates.
    #[test]
    fn differing_tags_earn_a_column() {
        let rendered = table(&[
            gauge("commit_ms", 1.5, "ms", &["shape:pr", "operation:commit"]),
            gauge("commit_ms", 2.5, "ms", &["shape:pr", "operation:checkout"]),
        ]);

        assert!(rendered.contains("Tags"), "{rendered}");
        assert!(rendered.contains("operation:commit"), "{rendered}");
        assert!(rendered.contains("operation:checkout"), "{rendered}");
        assert!(rendered.contains("1.500"), "{rendered}");
    }
}
