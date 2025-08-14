// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::collections::HashMap;
use std::path::Path;
use std::{fs::read_to_string, time::Duration};

use tezos_smart_rollup::utils::inbox::file::InboxFile;

use crate::{
    Result,
    common::{EXPECTED_LEVELS, LogLine},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Start,
    End,
}

#[derive(Debug)]
struct Event {
    elapsed: Duration,
    name: String,
    kind: Kind,
}

#[derive(Debug)]
struct Frame {
    name: String,
    start: Duration,
    children: HashMap<String, Node>,
}

#[derive(Debug, Clone)]
struct Node {
    name: String,
    total: Duration,
    children: HashMap<String, Node>,
    subtree_store_total: Duration,
}

impl Node {
    fn new(name: String) -> Self {
        Self {
            name,
            total: Duration::default(),
            children: HashMap::new(),
            subtree_store_total: Duration::default(),
        }
    }

    fn merge(&mut self, mut other: Node) {
        self.total += other.total;
        for (k, v) in other.children.drain() {
            self.children
                .entry(k)
                .and_modify(|e| e.merge(v.clone()))
                .or_insert(v);
        }
    }

    /// Time spent outside of children
    fn unaccounted_time(&self) -> Duration {
        let child_sum = self
            .children
            .values()
            .fold(Duration::default(), |acc, c| acc + c.total);
        self.total.saturating_sub(child_sum)
    }
}

fn parse_message(msg: &str) -> Option<(String, Kind)> {
    let rest = msg.strip_prefix("[Profiling] ")?;
    if let Some(name) = rest.strip_suffix(" start") {
        return Some((name.to_string(), Kind::Start));
    }
    if let Some(name) = rest.strip_suffix(" end") {
        return Some((name.to_string(), Kind::End));
    }
    None
}

fn is_calibration(msg: &str) -> bool {
    msg.trim() == "[Profiling] Calibration"
}

/// Compute mean calibration time from the leading calibration lines
fn mean_calibration(lines: &[LogLine]) -> Option<(Duration, usize)> {
    let mut debug_log_times = Vec::new();
    for l in lines {
        if is_calibration(&l.message) {
            debug_log_times.push(l.elapsed);
        } else {
            break;
        }
    }
    let mean = if debug_log_times.len() >= 2 {
        let mut sum = Duration::default();
        for w in debug_log_times.windows(2) {
            let dt = w[1].checked_sub(w[0]).unwrap_or_default();
            sum += dt;
        }
        sum.div_f64((debug_log_times.len() - 1) as f64)
    } else {
        return None;
    };
    Some((mean, debug_log_times.len()))
}

/// Build calibrated events by subtracting from each event's timestamp the time it took
/// to print the log lines before it
fn calibrated_events_from_lines(lines: &[LogLine], mean_calib: Duration) -> Vec<Event> {
    let mut events = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        let offset = mean_calib.mul_f64(i as f64);
        let elapsed = l.elapsed.saturating_sub(offset);

        if let Some((name, kind)) = parse_message(&l.message) {
            events.push(Event {
                elapsed,
                name,
                kind,
            });
        }
    }
    events
}

fn build_aggregated(events: Vec<Event>) -> (Node, Vec<String>) {
    let mut warnings = Vec::new();
    let mut root = Node::new("*".to_string());

    // Assuming `events` is not empty as this point
    let end_t = events[events.len() - 1].elapsed;
    root.total = end_t.saturating_sub(events[0].elapsed);

    let mut stack: Vec<Frame> = Vec::new();

    for ev in events {
        match ev.kind {
            Kind::Start => {
                stack.push(Frame {
                    name: ev.name,
                    start: ev.elapsed,
                    children: HashMap::new(),
                });
            }
            Kind::End => {
                let mut frame = match stack.pop() {
                    Some(f) => f,
                    None => {
                        warnings.push(format!(
                            "End without matching start for {} at t={:?}",
                            ev.name, ev.elapsed
                        ));
                        continue;
                    }
                };

                if frame.name != ev.name {
                    warnings.push(format!(
                        "Mismatched end: got end of {}, but top of stack is {} at t={:?}",
                        ev.name, frame.name, ev.elapsed
                    ));
                    if let Some(pos) = stack.iter().rposition(|n| n.name == ev.name) {
                        while stack.len() > pos + 1 {
                            let dropped = stack.pop().unwrap();
                            warnings.push(format!(
                                "Dropping unclosed frame {} (started at {:?})",
                                dropped.name, dropped.start
                            ));
                        }
                        frame = stack.pop().unwrap();
                    }
                }
                let total = ev.elapsed.saturating_sub(frame.start);
                let mut node = Node {
                    name: frame.name,
                    total,
                    children: HashMap::new(),
                    subtree_store_total: Duration::default(),
                };

                // Attach the already-aggregated children collected while the frame was open.
                for (_, child) in frame.children.drain() {
                    node.children
                        .entry(child.name.clone())
                        .and_modify(|e| e.merge(child.clone()))
                        .or_insert(child);
                }

                if let Some(parent) = stack.last_mut() {
                    parent
                        .children
                        .entry(node.name.clone())
                        .and_modify(|e| e.merge(node.clone()))
                        .or_insert(node);
                } else {
                    root.children
                        .entry(node.name.clone())
                        .and_modify(|e| e.merge(node.clone()))
                        .or_insert(node);
                }
            }
        }
    }

    while let Some(mut dangling) = stack.pop() {
        warnings.push(format!(
            "Unclosed frame {} (started at {:?}), closing at end {:?}",
            dangling.name, dangling.start, end_t
        ));
        let total = end_t.saturating_sub(dangling.start);
        let mut node = Node {
            name: dangling.name,
            total,
            children: HashMap::new(),
            subtree_store_total: Duration::default(),
        };
        for (_, child) in dangling.children.drain() {
            node.children
                .entry(child.name.clone())
                .and_modify(|e| e.merge(child.clone()))
                .or_insert(child);
        }
        root.children
            .entry(node.name.clone())
            .and_modify(|e| e.merge(node.clone()))
            .or_insert(node);
    }

    compute_store_totals(&mut root);
    (root, warnings)
}

fn is_store(name: &str) -> bool {
    name.starts_with("store_")
}

fn compute_store_totals(node: &mut Node) -> Duration {
    // Storage access nodes are always leaves
    if is_store(&node.name) {
        node.subtree_store_total = node.total;
        return node.subtree_store_total;
    }
    let mut sum = Duration::default();
    for child in node.children.values_mut() {
        sum += compute_store_totals(child);
    }
    node.subtree_store_total = sum;
    sum
}

fn print_tree(root: &Node) {
    let total = if root.total.is_zero() {
        root.children
            .values()
            .fold(Duration::default(), |acc, c| acc + c.total)
    } else {
        root.total
    };
    let unaccounted = root.unaccounted_time();
    let store = root.subtree_store_total;

    println!(
        "{} total: {:?} | storage: {:?} ({:>.1}%) | unaccounted: {:?} ({:>.1}%)",
        root.name,
        total,
        store,
        percent(store, total),
        unaccounted,
        percent(unaccounted, total),
    );

    let mut children: Vec<&Node> = root.children.values().collect();
    children.sort_by(|a, b| b.total.cmp(&a.total));

    for (i, child) in children.iter().enumerate() {
        print_node(child, total, &String::new(), children.len() == i + 1);
    }
}

fn percent(num: Duration, den: Duration) -> f64 {
    if den.is_zero() {
        0.
    } else {
        num.div_duration_f64(den) * 100.0
    }
}

fn print_node(node: &Node, parent_total: Duration, prefix: &String, is_last: bool) {
    let branch = if is_last { "└─" } else { "├─" };
    let next_prefix = if is_last {
        format!("{prefix}   ")
    } else {
        format!("{prefix}│  ")
    };

    let total = node.total;
    let pct_parent = percent(total, parent_total);

    if is_store(&node.name) {
        println!(
            "{prefix}{branch} {}  total: {:?} | {:>.1}% of parent",
            node.name, total, pct_parent
        );
    } else {
        let store = node.subtree_store_total;
        let unaccounted = node.unaccounted_time();
        println!(
            "{prefix}{branch} {} | total: {:?} | {:>.1}% of parent | storage: {:?} ({:>.1}%) | unaccounted: {:?} ({:>.1}%)",
            node.name,
            total,
            pct_parent,
            store,
            percent(store, total),
            unaccounted,
            percent(unaccounted, total),
        );
    }

    let mut children: Vec<&Node> = node.children.values().collect();
    children.sort_by(|a, b| b.total.cmp(&a.total));
    for (i, child) in children.iter().enumerate() {
        let last = i + 1 == children.len();
        print_node(child, total, &next_prefix, last);
    }
}

pub fn handle_profile(
    inbox: Box<Path>,
    log_file: Box<Path>,
    #[allow(unused, reason = "TODO")] expected_transfers: usize,
) -> Result<()> {
    let inbox = InboxFile::load(&inbox)?;
    if inbox.0.len() != EXPECTED_LEVELS {
        return Err(format!(
            "Inbox contains {} levels, expected {EXPECTED_LEVELS}",
            inbox.0.len()
        )
        .into());
    }

    let log = read_to_string(log_file)?
        .lines()
        .map(serde_json::from_str::<LogLine>)
        .collect::<Result<Vec<_>, _>>()?;

    let (mean_calib, calib_count) =
        mean_calibration(&log).ok_or("No leading calibration lines found")?;

    let events = calibrated_events_from_lines(&log[calib_count..], mean_calib);
    if events.is_empty() {
        return Err("No profiling start/end events found after calibration lines.".into());
    }

    let (root, warnings) = build_aggregated(events);

    if !warnings.is_empty() {
        eprintln!("Warnings:");
        for w in &warnings {
            eprintln!("  - {w}");
        }
        eprintln!();
    }

    print_tree(&root);
    Ok(())
}
