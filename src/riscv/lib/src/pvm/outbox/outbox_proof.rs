// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use bincode::Encode;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
use octez_riscv_data::serialisation::serialise;

#[derive(Debug, Encode)]
pub struct OutputInfo {
    pub level: usize,
    pub index: usize,
}

pub struct Output {
    pub message: Vec<u8>,
    pub info: OutputInfo,
}

#[derive(Debug, Encode)]
pub struct OutboxProof {
    pub proof: MerkleProof,
    pub info: OutputInfo,
}

impl OutboxProof {
    pub(crate) fn new(proof: MerkleProof, level: usize, index: usize) -> Self {
        Self {
            proof,
            info: OutputInfo { level, index },
        }
    }

    /// Get the state hash of the proof.
    pub fn state_hash(&self) -> Hash {
        self.proof.root_hash()
    }

    pub fn serialise(&self) -> Vec<u8> {
        serialise(self).expect("Serialisation of an outbox proof should not fail")
    }
}
