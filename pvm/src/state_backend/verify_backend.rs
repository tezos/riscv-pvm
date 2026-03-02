// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use octez_riscv_data::hash::Hash;
use octez_riscv_data::mode::utils::CaughtNotFoundOrPanic;
use octez_riscv_data::mode::utils::NotFound;

use crate::pvm::outbox::OutboxProofError;
use crate::state_backend::ProofError;

/// Error during proof verification
#[derive(Debug, thiserror::Error)]
pub enum ProofVerificationFailure {
    #[error("Deserialisation error: {0}")]
    BadDeserialisation(#[from] ProofError),

    #[error("Stepper error")]
    StepperError,

    #[error("Stepper panic")]
    StepperPanic(Box<dyn std::any::Any + Send>),

    #[error("Attempted to access absent data")]
    AbsentDataAccess(#[from] NotFound),

    #[error("Proof cannot be used for hashing after the verification step")]
    BadProofForHashing,

    #[error("Final state hash mismatch (expected {expected}, computed {computed})")]
    FinalHashMismatch { expected: Hash, computed: Hash },

    #[error(transparent)]
    OutboxProofError(#[from] OutboxProofError),
}

impl From<CaughtNotFoundOrPanic> for ProofVerificationFailure {
    fn from(error: CaughtNotFoundOrPanic) -> Self {
        match error {
            CaughtNotFoundOrPanic::NotFound(not_found) => Self::AbsentDataAccess(not_found),
            CaughtNotFoundOrPanic::Other(panic_info) => Self::StepperPanic(panic_info),
        }
    }
}
