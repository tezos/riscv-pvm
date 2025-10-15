// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::fmt;
use std::fs::read_to_string;
use std::path::Path;
use std::time::Duration;

use serde::Deserialize;
use tezos_smart_rollup::utils::inbox::file::InboxFile;

use crate::Result;

const EXPECTED_LEVELS: usize = 1;
const EXPECTED_EXTRA_TXS: usize = 5;

pub fn handle_results(
    inbox: Box<Path>,
    log_file: Box<Path>,
    expected_transfers: usize,
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
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let txs = extract_transactions_from_log(log.as_slice())?;
    let transfers = check_expected_execution(txs.as_slice(), expected_transfers)?;
    let all_metrics = [check_transfer_metrics(transfers)?];

    let mut all_results = comfy_table::Table::new();
    all_results.load_preset(comfy_table::presets::ASCII_MARKDOWN);
    all_results.set_header(["Run", "Transfers", "Duration", "TPS"]);
    all_results
        .column_mut(2)
        .unwrap()
        .set_cell_alignment(comfy_table::CellAlignment::Right);
    all_results
        .column_mut(3)
        .unwrap()
        .set_cell_alignment(comfy_table::CellAlignment::Right);

    for (run, metrics) in all_metrics.iter().enumerate() {
        all_results.add_row([
            format!("{}", run + 1),
            format!("{}", metrics.transfers),
            format!("{:?}", metrics.duration),
            format!("{:.3}", metrics.tps),
        ]);
    }

    println!();
    println!("{all_results}");

    Ok(())
}

fn extract_transactions_from_log(log: &[LogLine]) -> Result<Vec<Tx>> {
    let mut txs: Vec<Tx> = Vec::new();
    let mut tx_builder: Option<TxBuilder> = None;
    for line in log {
        if line.message.starts_with(TX_OUTCOME_LINE) {
            // If a transaction is already in progress (i.e., this is not
            // the first transaction), build it using the new transaction's
            // start time as its end time.
            if let Some(txb) = tx_builder.take() {
                txs.push(txb.build(Some(line.elapsed))?)
            };
            let txb = TxBuilder::new(line.elapsed);
            let tx_outcome = parse_execution_outcome(&line.message)
                .ok_or("Could not extract transaction execution outcome")?;
            tx_builder = Some(txb.outcome(tx_outcome));
        }
        if line.message.starts_with(TX_END_LINE) {
            let txb = tx_builder
                .ok_or("Parsed unexpected transaction end: no transaction in progress")?;
            tx_builder = Some(txb.finish()?);
        }
    }

    // If a final transaction is in progress, built it with no known end time
    if let Some(txb) = tx_builder.take() {
        txs.push(txb.build(None)?);
    }

    Ok(txs)
}

/// The inbox is assumed to contain a single level with the following structure:
/// Account 1 gets tokens from faucet.
/// Account 1 deploys the ERC20 contract. This is split across 3 messages.
/// Account 1 calls `mint`.
/// Account 1 calls `transfer` to Account 2. (`expected_transfers` times)
/// Account 1 calls `balanceOf` Account 1.
/// Account 1 calls `balanceOf` Account 2.
///
/// Check correct execution of the inbox messages:
/// - The number of transactions is `expected_transfers` + 5 and matches the expected flow.
/// - The final balances are correct.
fn check_expected_execution(txs: &[Tx], expected_transfers: usize) -> Result<&[Tx]> {
    let expected_txs = expected_transfers + EXPECTED_EXTRA_TXS;
    if txs.len() != expected_txs {
        return Err(format!(
            "Executed {} transactions, expected {expected_txs}",
            txs.len()
        )
        .into());
    }

    if !matches!(txs[1].outcome.result, TxExecutionResult::Success {
        output: OutputType::Create(_),
        ..
    }) {
        return Err("Expected the second transaction to be a contract deployment".into());
    }

    let mut account1 = check_transfer(&txs[2])?;
    let mut account2 = 0;

    for tx in txs[3..txs.len() - 2].iter() {
        let amount = check_transfer(tx)?;
        account1 -= amount;
        account2 += amount;
    }

    let account1_queried = check_call_balance_of(&txs[txs.len() - 2])?;
    let account2_queried = check_call_balance_of(&txs[txs.len() - 1])?;

    if account1 != account1_queried || account2 != account2_queried {
        return Err(
            format!("Final balances: {account1}, {account2}. Expected {account1_queried}, {account2_queried}.").into(),
        );
    }
    Ok(&txs[3..txs.len() - 2])
}

fn check_transfer(tx: &Tx) -> Result<u64> {
    match &tx.outcome.result {
        TxExecutionResult::Success { logs, .. } => {
            let log = logs
                .first()
                .ok_or(format!("Expected this contract call to have a log {tx:?}"))?;
            let amount = u64_from_ethereum_u256_bytes(log.data.as_slice()).ok_or(format!(
                "Expected this contract call to have log data {tx:?}"
            ))?;
            Ok(amount)
        }
        _ => Err(format!("Expected a successful contract call, got {tx:?}").into()),
    }
}

fn check_call_balance_of(tx: &Tx) -> Result<u64> {
    match &tx.outcome.result {
        TxExecutionResult::Success { output, .. } => match output {
            OutputType::Call(bytes) => u64_from_ethereum_u256_bytes(bytes.as_slice())
                .ok_or(format!("Expected this contract call to have a return value {tx:?}").into()),
            _ => Err(format!("Expected a Call output, got {tx:?}").into()),
        },
        _ => Err(format!("Expected a successful contract call, got {tx:?}").into()),
    }
}

#[derive(Clone, Debug, Default)]
struct TransferMetrics {
    transfers: usize,
    duration: Duration,
    tps: f64,
}

impl fmt::Display for TransferMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ERC20 transfers took {:?} @ {:.3} TPS",
            self.transfers, self.duration, self.tps
        )
    }
}

fn check_transfer_metrics(txs: &[Tx]) -> Result<TransferMetrics> {
    let transfers = txs.len();
    let durations: Result<Vec<Duration>, _> = txs
        .iter()
        .map(|e| e.duration.ok_or("All transfers should be timed"))
        .collect();
    let duration: Duration = durations?.into_iter().sum();
    let tps = (transfers as f64) / duration.as_secs_f64();

    Ok(TransferMetrics {
        transfers,
        duration,
        tps,
    })
}

#[derive(Deserialize, Debug, PartialEq)]
struct LogLine {
    elapsed: Duration,
    message: String,
}

#[derive(Debug)]
struct TxLog {
    #[expect(unused)]
    address: String,
    data: Vec<u8>,
}

#[derive(Debug)]
enum TxExecutionResult {
    Success {
        #[expect(unused)]
        reason: String,
        logs: Vec<TxLog>,
        output: OutputType,
    },
    Other(#[expect(unused)] String),
}

/// Output of executing a transaction
#[derive(Debug)]
enum OutputType {
    /// Holds returned value
    Call(Vec<u8>),
    /// Holds contract address
    Create(#[expect(unused)] String),
}

#[derive(Debug)]
struct TxOutcome {
    result: TxExecutionResult,
}

#[derive(Debug)]
struct Tx {
    outcome: TxOutcome,
    duration: Option<Duration>,
}

struct TxBuilder {
    start: Duration,
    outcome: Option<TxOutcome>,
    finished: bool,
}

impl TxBuilder {
    fn new(start: Duration) -> Self {
        Self {
            start,
            outcome: None,
            finished: false,
        }
    }

    fn outcome(mut self, outcome: TxOutcome) -> Self {
        self.outcome = Some(outcome);
        self
    }

    fn finish(self) -> Result<Self> {
        if self.outcome.is_none() {
            return Err("Could not finish transaction: no outcome extracted".into());
        }
        Ok(Self {
            finished: true,
            ..self
        })
    }

    fn build(self, end: Option<Duration>) -> Result<Tx> {
        if !self.finished {
            return Err("Unexpected end of transaction".into());
        }
        let outcome = self.outcome.ok_or("Inconsistent tx builder state")?;
        Ok(Tx {
            outcome,
            duration: end.map(|t| t - self.start),
        })
    }
}

fn parse_execution_outcome(line: &str) -> Option<TxOutcome> {
    if !line.contains("ExecutionOutcome") {
        return None;
    }

    let result = parse_result(line)?;

    Some(TxOutcome { result })
}

fn parse_result(s: &str) -> Option<TxExecutionResult> {
    let result_start = s.find("result: ")?;
    let result_str = &s[result_start + 8..];

    if result_str.starts_with("Success {") {
        let reason = result_str
            .split("reason: ")
            .nth(1)?
            .split(',')
            .next()?
            .trim()
            .to_string();

        let logs = parse_tx_logs_array(result_str);

        let output = parse_output(result_str)?;

        Some(TxExecutionResult::Success {
            reason,
            logs,
            output,
        })
    } else {
        let raw = result_str.split('}').next().unwrap_or("").to_string();
        Some(TxExecutionResult::Other(raw))
    }
}

fn parse_tx_logs_array(result_str: &str) -> Vec<TxLog> {
    if result_str.contains("logs: []") {
        return vec![];
    }

    let mut logs = Vec::new();
    if let Some(logs_start) = result_str.find("logs: [Log {") {
        if let Some(single_log) = parse_single_tx_log(&result_str[logs_start..]) {
            logs.push(single_log);
        }
    }
    logs
}

fn parse_single_tx_log(log_str: &str) -> Option<TxLog> {
    let address = log_str
        .split("address: ")
        .nth(1)?
        .split(',')
        .next()?
        .trim()
        .to_string();

    // Parse data from LogData
    let data_hex = log_str
        .split("data: 0x")
        .nth(1)?
        .split_whitespace()
        .next()?
        .split('}')
        .next()?
        .trim();
    let data = hex::decode(data_hex).ok()?;

    Some(TxLog { address, data })
}

fn parse_output(result_str: &str) -> Option<OutputType> {
    let output_start = result_str.find("output: ")?;
    let output_str = &result_str[output_start + 8..];

    if output_str.starts_with("Call(") {
        let hex_start = output_str.find("0x")?;
        let hex_str = output_str[hex_start + 2..].split(')').next()?.trim();
        let bytes = hex::decode(hex_str).ok()?;
        Some(OutputType::Call(bytes))
    } else if output_str.starts_with("Create(") {
        // For Create, extract the address if present
        let address = output_str
            .split("Some(")
            .nth(1)
            .and_then(|s| s.split(')').next())
            .unwrap_or("")
            .to_string();
        Some(OutputType::Create(address))
    } else {
        None
    }
}

fn u64_from_ethereum_u256_bytes(bytes: &[u8]) -> Option<u64> {
    if bytes.is_empty() {
        None
    } else {
        let raw_bytes: [u8; 32] = bytes.try_into().ok()?;
        Some(primitive_types::U256::from_big_endian(&raw_bytes).as_u64())
    }
}

const TX_OUTCOME_LINE: &str = "[Debug] Transaction executed, outcome:";
const TX_END_LINE: &str = "[Debug] Applying FeeUpdates";
