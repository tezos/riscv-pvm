// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Durable storage interfaces

use bincode::Decode;
use bincode::Encode;
use octez_riscv_data::clone::CloneState;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::NodeFold;
use octez_riscv_data::merkle_proof::DeserialiserNode;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Provable;
use octez_riscv_durable_storage::errors::OperationalError;
use octez_riscv_durable_storage::registry::CloneRegistryMode;
use octez_riscv_durable_storage::registry::Registry;

/// Implementing types provide an interface for durable storage
pub trait DurableStorage<M: Mode>: Sized {
    fn try_clone(&self) -> Result<Self, OperationalError>
    where
        M: CloneRegistryMode;
}

impl<M: Mode> DurableStorage<M> for Registry<M> {
    fn try_clone(&self) -> Result<Self, OperationalError>
    where
        M: CloneRegistryMode,
    {
        M::try_clone(self)
    }
}

/// Dummy implementation for Durable Storage
///
/// This type's purpose is primarily to aid with the integration of the durable storage into the PVM.
/// The idea is that you can always use this type in place of the PVM's "durable storage" type
/// parameter, in order to make the entire code base compile.
///
/// A non-goal is to provide a run-time working implementation of a durable storage system.
#[derive(Debug, Default, Clone, Encode, Decode, PartialEq, Eq)]
pub struct DurableStorageDummy;

impl<M: Mode> DurableStorage<M> for DurableStorageDummy {
    fn try_clone(&self) -> Result<Self, OperationalError> {
        Ok(Self)
    }
}

impl<F: Fold> Foldable<F> for DurableStorageDummy {
    fn fold(&self, builder: F) -> <F as Fold>::Folded {
        builder.into_node_fold().done()
    }
}

impl FromProof for DurableStorageDummy {
    fn from_proof<Proof: octez_riscv_data::merkle_proof::Deserialiser>(
        proof: Proof,
    ) -> octez_riscv_data::merkle_proof::SuspendedResult<Proof, Self> {
        let node = proof.into_node()?;
        node.done(Self)
    }
}

impl<'normal> Provable<'normal> for DurableStorageDummy {
    type Prover = Self;

    fn start_proof(&'normal self) -> Self::Prover {
        Self
    }
}

impl CloneState for DurableStorageDummy {
    fn clone_state(&self) -> Self {
        Self
    }
}
