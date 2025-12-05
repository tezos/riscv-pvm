// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use octez_riscv_data::clone::CloneState;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::NodeFold;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::DeserialiserNode;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::SuspendedResult;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use perfect_derive::perfect_derive;
use tezos_smart_rollup_constants::riscv::REVEAL_REQUEST_MAX_SIZE;

use crate::state::NewState;
use crate::state_backend::Cell;
use crate::state_backend::DynCells;
use crate::state_backend::ManagerAlloc;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerClone;
use crate::state_backend::ManagerDeserialise;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerSerialise;

/// Request content of reveal
#[perfect_derive(Clone, PartialEq, Eq)]
pub struct RevealRequest<M: ManagerBase> {
    /// Reveal request payload
    pub bytes: DynCells<M>,
    /// Size of reveal request payload
    pub size: Cell<u64, M>,
}

impl<M: ManagerBase> RevealRequest<M> {
    pub fn to_vec(&self) -> Vec<u8>
    where
        M: ManagerRead,
    {
        use std::cmp::min;

        let size = self.size.read() as usize;
        let mut buffer = vec![0u8; min(size, REVEAL_REQUEST_MAX_SIZE)];
        self.bytes.read_all(0, &mut buffer);
        buffer
    }
}

impl RevealRequest<Normal> {
    /// Return a proof-generating version of this RevealRequest.
    pub fn start_proof(&self) -> RevealRequest<Prove<'_>> {
        RevealRequest {
            bytes: self.bytes.start_proof(),
            size: self.size.start_proof(),
        }
    }
}

impl<M: ManagerBase> NewState<M> for RevealRequest<M> {
    fn new() -> Self
    where
        M: ManagerAlloc,
    {
        Self {
            bytes: DynCells::new(REVEAL_REQUEST_MAX_SIZE),
            size: Cell::new(),
        }
    }
}

impl<M: ManagerClone> CloneState for RevealRequest<M> {
    fn clone_state(&self) -> Self {
        Self {
            bytes: self.bytes.clone_state(),
            size: self.size.clone_state(),
        }
    }
}

impl<M, F> Foldable<F> for RevealRequest<M>
where
    M: ManagerBase,
    F: Fold,
    DynCells<M>: Foldable<F>,
    Cell<u64, M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> F::Folded {
        let mut builder = builder.into_node_fold();
        builder.add(&self.bytes);
        builder.add(&self.size);
        builder.done()
    }
}

impl FromProof for RevealRequest<Verify> {
    fn from_proof<D: Deserialiser>(proof: D) -> SuspendedResult<D, Self> {
        let proof = proof.into_node()?;

        let (proof, bytes) = proof.next_branch()?;
        let (proof, size) = proof.next_branch()?;

        proof.done(RevealRequest { bytes, size })
    }
}

impl<M: ManagerSerialise> Encode for RevealRequest<M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.bytes.encode(encoder)?;
        self.size.encode(encoder)?;
        Ok(())
    }
}

impl<C, M: ManagerDeserialise> Decode<C> for RevealRequest<M> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self {
            bytes: Decode::decode(decoder)?,
            size: Decode::decode(decoder)?,
        })
    }
}
