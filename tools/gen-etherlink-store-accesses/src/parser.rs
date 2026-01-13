// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::collections::HashMap;

use serde::Deserialize;
use serde::Serialize;

// Line prefixes
const DEBUG_PREFIX: &str = "[Debug]";
const OTEL_PREFIX: &str = "[OTel]";

// Section markers
const STAGE_ONE_MARKER: &str = "Entering stage one.";
const BIP_MARKER: &str = "Computing the BlockInProgress";
const APPLY_TX_MARKER: &str = "apply_transaction";
const REGISTER_VALID_TX_MARKER: &str = "register_valid_transaction";

/// Type of line in the input trace file.
#[derive(Debug)]
pub enum LineType {
    StorageEvent(Box<TraceEvent>),
    EnteringStageOne,
    ComputingBlockInProgress,
    ApplyTransaction,
    RegisterValidTransaction,
    Other,
}

/// Trace event deserialised from a JSON line of the input file.
#[derive(Debug)]
pub struct TraceEvent {
    pub is_enter: bool,
    pub name: String,
    pub path: Option<String>,
    pub from_path: Option<String>,
    pub to_path: Option<String>,
    pub size: Option<u64>,
    pub max_bytes: Option<u64>,
}

impl<'de> Deserialize<'de> for TraceEvent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;

        #[derive(Deserialize)]
        struct Trace {
            fields: serde_json::Value,
            span: serde_json::Value,
        }

        let trace = Trace::deserialize(deserializer)?;

        // Extract is_enter from fields.message
        let message = trace
            .fields
            .get("message")
            .and_then(|v| v.as_str())
            .ok_or_else(|| D::Error::custom("missing or invalid message field"))?;
        let is_enter = message == "enter";

        // Extract span fields
        let span = &trace.span;
        let name = span
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| D::Error::custom("missing or invalid name field"))?
            .to_string();
        let path = span
            .get("path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let from_path = span
            .get("from_path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let to_path = span
            .get("to_path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let size = span.get("size").and_then(|v| v.as_u64());
        let max_bytes = span.get("max_bytes").and_then(|v| v.as_u64());

        Ok(TraceEvent {
            is_enter,
            name,
            path,
            from_path,
            to_path,
            size,
            max_bytes,
        })
    }
}

/// Storage access produced from the input [`TraceEvent`]
#[derive(Debug, Serialize)]
pub struct StorageAccess {
    pub operation: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_path: Option<String>,
}

/// Top-level format of the output file.
#[derive(Debug, Serialize)]
pub struct OutputFormat {
    pub setup: Vec<StorageAccess>,
    pub transaction: Vec<StorageAccess>,
    pub block_creation: Vec<StorageAccess>,
}

/// Classify a line from the trace file.
pub fn classify_line(line: &str) -> LineType {
    if line.starts_with(DEBUG_PREFIX) {
        if line.contains(STAGE_ONE_MARKER) {
            return LineType::EnteringStageOne;
        }
        if line.contains(BIP_MARKER) {
            return LineType::ComputingBlockInProgress;
        }
        return LineType::Other;
    }

    if line.starts_with(OTEL_PREFIX) {
        // OTel events are printed without a newline. Extract what follows
        // after the OTel event(s) and classify that instead.
        if let Some(json_start) = line.find(r#"{"level":"#) {
            return classify_line(&line[json_start..]);
        }
        return LineType::Other;
    }

    // Try parsing the whole line as JSON
    if let Ok(event) = serde_json::from_str::<TraceEvent>(line) {
        // Check for section markers from JSON events
        if event.name == APPLY_TX_MARKER && event.is_enter {
            return LineType::ApplyTransaction;
        }
        if event.name == REGISTER_VALID_TX_MARKER {
            return LineType::RegisterValidTransaction;
        }

        // Check if it is a store operation
        if is_store_operation(&event.name) {
            return LineType::StorageEvent(Box::new(event));
        }
    }

    LineType::Other
}

fn is_store_operation(name: &str) -> bool {
    // This ignore the `store_move` operation because it doesn't result in any
    // operations on an actual database.
    matches!(
        name,
        "store_read"
            | "store_read_slice"
            | "store_read_all"
            | "store_write"
            | "store_write_all"
            | "store_has"
            | "store_delete"
            | "store_value_size"
            | "store_copy"
            | "__internal_store_get_hash"
    )
}

/// Builder for pairing enter/exit events into storage accesses.
pub struct StoreAccessBuilder {
    // Store accesses themselves are blocking so can never be interleaved.
    // This builder is used to guard against potential event reorderings in the trace file.
    pending_enters: HashMap<String, TraceEvent>,
}

impl StoreAccessBuilder {
    pub fn new() -> Self {
        Self {
            pending_enters: HashMap::new(),
        }
    }

    /// Process a trace event and return a store access on an exit event.
    ///
    /// Returns [`None`] in case of an unmatched exit event.
    pub fn process_event(&mut self, event: TraceEvent) -> Option<StorageAccess> {
        if event.is_enter {
            // Store enter event, keyed by operation + path
            let key = self.make_key(&event);
            self.pending_enters.insert(key, event);
            None
        } else {
            let key = self.make_key(&event);
            if let Some(enter_event) = self.pending_enters.remove(&key) {
                Some(self.condense_pair(enter_event, event))
            } else {
                panic!("No matching enter event for {event:?}");
            }
        }
    }

    /// Create a key for an event based on operation name and path. This assumes the trace does
    /// not contain interleaved store accesses of exactly the same type and path.
    fn make_key(&self, event: &TraceEvent) -> String {
        let path = event
            .path
            .as_ref()
            .or(event.from_path.as_ref())
            .map(|p| extract_path(p))
            .unwrap_or_default();
        format!("{}:{}", event.name, path)
    }

    /// Condense an enter/exit pair into a single storage access.
    fn condense_pair(&self, enter: TraceEvent, exit: TraceEvent) -> StorageAccess {
        let operation = enter.name.clone();

        // Get size from exit event (if present)
        // For store_read, size comes from max_bytes field
        // For read operations, default to Some(0) if size is missing (e.g., due to errors)
        let size = if operation == "store_read" {
            exit.max_bytes
        } else if operation == "store_read_all" || operation == "store_read_slice" {
            Some(exit.size.unwrap_or(0))
        } else {
            exit.size
        };

        // Copy operations use from_path and to_path
        if operation == "store_copy" {
            StorageAccess {
                operation,
                path: None,
                size: None,
                from_path: enter.from_path.map(|p| extract_path(&p)),
                to_path: enter.to_path.map(|p| extract_path(&p)),
            }
        } else {
            // All other operations use path field
            let path = enter.path.map(|p| extract_path(&p));
            StorageAccess {
                operation,
                path,
                size,
                from_path: None,
                to_path: None,
            }
        }
    }
}

/// Extract inner path from wrapper types like RefPath or OwnedPath.
///
/// Examples:
/// - `RefPath { inner: "/path" }` -> "/path"
/// - `OwnedPath { inner: "/path" }` -> "/path"
fn extract_path(path_str: &str) -> String {
    // Handle: RefPath { inner: "/path" } or OwnedPath { inner: "/path" }
    if let Some(start) = path_str.find('"') {
        if let Some(end) = path_str.rfind('"') {
            return path_str[start + 1..end].to_string();
        }
    }
    path_str.to_string()
}
