// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! PVM-related operational errors

/// Errors that occur during operations of the PVM
#[derive(Debug, thiserror::Error)]
pub enum OperationalError {
    #[error("Durable storage error: {0}")]
    DurableStorage(#[from] octez_riscv_durable_storage::errors::OperationalError),
}
