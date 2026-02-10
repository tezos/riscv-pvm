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
use octez_riscv_data::components::atom::Atom;
use octez_riscv_data::components::atom::AtomMode;
use octez_riscv_data::components::atom::CloneAtomMode;
use octez_riscv_data::components::atom::EncodeAtomMode;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::NodeFold;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::DeserialiserNode;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::SuspendedResult;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Provable;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use perfect_derive::perfect_derive;
use tezos_smart_rollup_constants::riscv::REVEAL_REQUEST_MAX_SIZE;

/// Request content of reveal
#[perfect_derive(Clone, PartialEq, Eq)]
pub struct RevealRequest<M: Mode> {
    /// Reveal request payload
    pub bytes: Atom<[u8; REVEAL_REQUEST_MAX_SIZE], M>,
    /// Size of reveal request payload
    pub size: Atom<u64, M>,
}

impl<M: Mode> RevealRequest<M> {
    /// Read the reveal request as a vector.
    pub fn to_vec(&self) -> Vec<u8>
    where
        M: AtomMode,
    {
        self.bytes[..self.size.read() as usize].to_vec()
    }
}

impl<'normal> Provable<'normal> for RevealRequest<Normal> {
    type Prover = RevealRequest<Prove<'normal>>;

    fn start_proof(&'normal self) -> Self::Prover {
        RevealRequest {
            bytes: self.bytes.start_proof(),
            size: self.size.start_proof(),
        }
    }
}

impl<M: AtomMode> Default for RevealRequest<M> {
    fn default() -> Self {
        Self {
            bytes: Atom::new([0; REVEAL_REQUEST_MAX_SIZE]),
            size: Atom::default(),
        }
    }
}

impl<M: CloneAtomMode> CloneState for RevealRequest<M> {
    fn clone_state(&self) -> Self {
        Self {
            bytes: self.bytes.clone_state(),
            size: self.size.clone_state(),
        }
    }
}

impl<M, F> Foldable<F> for RevealRequest<M>
where
    M: Mode,
    F: Fold,
    Atom<[u8; REVEAL_REQUEST_MAX_SIZE], M>: Foldable<F>,
    Atom<u64, M>: Foldable<F>,
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

impl<M: EncodeAtomMode> Encode for RevealRequest<M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.bytes.encode(encoder)?;
        self.size.encode(encoder)?;
        Ok(())
    }
}

impl<C> Decode<C> for RevealRequest<Normal> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self {
            bytes: Decode::decode(decoder)?,
            size: Decode::decode(decoder)?,
        })
    }
}
