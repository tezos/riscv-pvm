// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2024 Nomadic Labs <contact@nomadic-labs.com>
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
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use perfect_derive::perfect_derive;

use crate::machine_state::csregisters;
use crate::machine_state::memory::Address;
use crate::machine_state::registers;
use crate::machine_state::registers::XValue;
use crate::machine_state::reservation_set::ReservationSet;
use crate::state_backend as backend;
use crate::state_context::projection::AtomProj;
use crate::state_context::projection::MachineCoreCons;
use crate::state_context::projection::impl_projection;

/// RISC-V hart state
#[perfect_derive(Clone, PartialEq, Eq, Default)]
pub struct HartState<M: Mode> {
    /// Integer registers
    pub xregisters: registers::XRegisters<M>,

    /// Floating-point number registers
    pub fregisters: registers::FRegisters<M>,

    /// Control and state registers
    pub csregisters: csregisters::CSRegisters<M>,

    /// Program counter
    pub pc: Atom<Address, M>,

    /// Reservation set address
    pub reservation_set: ReservationSet<M>,
}

impl<M: Mode> HartState<M> {
    /// Reset the hart state.
    pub fn reset(&mut self, pc: Address)
    where
        M: backend::ManagerWrite,
    {
        self.xregisters.reset();
        self.fregisters.reset();
        self.csregisters.reset();
        self.pc.write(pc);
        self.reservation_set.reset();
    }
}

impl HartState<Normal> {
    /// Return a proof-generating version of this HartState.
    pub fn start_proof(&self) -> HartState<Prove<'_>> {
        HartState {
            xregisters: self.xregisters.start_proof(),
            fregisters: self.fregisters.start_proof(),
            csregisters: self.csregisters.start_proof(),
            pc: self.pc.start_proof(),
            reservation_set: self.reservation_set.start_proof(),
        }
    }
}

impl<M: backend::ManagerClone> CloneState for HartState<M> {
    fn clone_state(&self) -> Self {
        Self {
            xregisters: self.xregisters.clone_state(),
            fregisters: self.fregisters.clone_state(),
            csregisters: self.csregisters.clone_state(),
            pc: self.pc.clone_state(),
            reservation_set: self.reservation_set.clone_state(),
        }
    }
}

impl<M: Mode, F: Fold> Foldable<F> for HartState<M>
where
    registers::XRegisters<M>: Foldable<F>,
    registers::FRegisters<M>: Foldable<F>,
    csregisters::CSRegisters<M>: Foldable<F>,
    Atom<Address, M>: Foldable<F>,
    ReservationSet<M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> F::Folded {
        let mut builder = builder.into_node_fold();
        builder.add(&self.xregisters);
        builder.add(&self.fregisters);
        builder.add(&self.csregisters);
        builder.add(&self.pc);
        builder.add(&self.reservation_set);
        builder.done()
    }
}

impl FromProof for HartState<Verify> {
    fn from_proof<D: Deserialiser>(proof: D) -> SuspendedResult<D, Self> {
        let proof = proof.into_node()?;

        let (proof, xregisters) = proof.next_branch()?;
        let (proof, fregisters) = proof.next_branch()?;
        let (proof, csregisters) = proof.next_branch()?;
        let (proof, pc) = proof.next_branch()?;
        let (proof, reservation_set) = proof.next_branch()?;

        proof.done(HartState {
            xregisters,
            fregisters,
            csregisters,
            pc,
            reservation_set,
        })
    }
}

impl<M: EncodeAtomMode> Encode for HartState<M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.xregisters.encode(encoder)?;
        self.fregisters.encode(encoder)?;
        self.csregisters.encode(encoder)?;
        self.pc.encode(encoder)?;
        self.reservation_set.encode(encoder)?;
        Ok(())
    }
}

impl<C> Decode<C> for HartState<Normal> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self {
            xregisters: Decode::decode(decoder)?,
            fregisters: Decode::decode(decoder)?,
            csregisters: Decode::decode(decoder)?,
            pc: Decode::decode(decoder)?,
            reservation_set: Decode::decode(decoder)?,
        })
    }
}

impl_projection! {
    pub(crate) projection ProgramCounterProj {
        subject = MachineCoreCons,
        target_projection = AtomProj<XValue>,
        path = hart.pc,
    }
}
