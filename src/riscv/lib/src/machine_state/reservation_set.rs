// SPDX-FileCopyrightText: 2024 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Reservation set for Load-Reserved/Store-Conditional instructions
//! in the RISC-V A extension
//!
//! Section 8.2 - Unprivileged spec

use std::ops::BitAnd;

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use octez_riscv_data::clone::CloneState;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::Suspended;
use octez_riscv_data::merkle_proof::SuspendedResult;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use perfect_derive::perfect_derive;

use crate::machine_state::backend;
/// Executing a LR.x instructions registers a reservation set on the address
/// from which data was loaded. The success of a SC.x instruction is conditional
/// on there being a valid reservation which includes the word or doubleword
/// being stored. Every SC.x, wether successful or not, invalidates the hart's
/// reservation set.
///
/// "The invalidation of a hart’s reservation when it executes an LR or SC implies
/// that a hart can only hold one reservation at a time, and that an SC can only
/// pair with the most recent LR, and LR with the next following SC, in program
/// order."
use crate::machine_state::backend::Cell;
use crate::state::NewState;

#[perfect_derive(Clone)]
pub struct ReservationSet<M: backend::ManagerBase> {
    pub(crate) start_addr: Cell<u64, M>,
}

/// Layout for [ReservationSet]
pub type ReservationSetLayout = backend::Atom<u64>;

/// The size of the reservation set is 8 bytes in order to accommodate
/// LR.D/SC.D instructions which work on doubles.
///
/// "An implementation can register an arbitrarily large reservation set on
/// each LR, provided the reservation set includes all bytes of the addressed
/// data word or doubleword. [...] The Unix platform is expected to require of
/// main memory that the reservation set be of fixed size, contiguous, naturally
/// aligned, and no greater than the virtual memory page size."
///
/// With an 8-byte reservation set, we can align addresses by bitwise ANDing with
/// the bitmask.
pub(crate) const RES_SET_BITMASK: u64 = !0b111;

pub(crate) const UNSET_VALUE: u64 = u64::MAX;

impl<M: backend::ManagerBase> ReservationSet<M> {
    /// Bind the reservation set cell to the given allocated space
    pub fn bind(space: backend::AllocatedOf<ReservationSetLayout, M>) -> Self {
        Self { start_addr: space }
    }

    /// Given a manager morphism `f : &M -> N`, return the layout's allocated structure containing
    /// the constituents of `N` that were produced from the constituents of `&M`.
    pub fn struct_ref<'a, F: backend::FnManager<'a, M>>(
        &'a self,
    ) -> backend::AllocatedOf<ReservationSetLayout, F::Output> {
        self.start_addr.struct_ref::<F>()
    }

    /// Unset any reservation
    pub fn reset(&mut self)
    where
        M: backend::ManagerWrite,
    {
        self.write(UNSET_VALUE);
    }

    #[inline(always)]
    fn write(&mut self, addr: u64)
    where
        M: backend::ManagerWrite,
    {
        self.start_addr.write(addr)
    }

    #[inline(always)]
    fn read(&self) -> u64
    where
        M: backend::ManagerRead,
    {
        self.start_addr.read()
    }

    /// Set the reservation set as `addr` aligned to the nearest double
    #[inline]
    pub fn set(&mut self, addr: u64)
    where
        M: backend::ManagerWrite,
    {
        self.write(addr.bitand(RES_SET_BITMASK));
    }

    /// Check wether the `addr` is within the reservation set
    pub fn test_and_unset(&mut self, addr: u64) -> bool
    where
        M: backend::ManagerRead + backend::ManagerWrite,
    {
        let start_addr = self.read();
        // Regardless of success or failure, executing an SC.x instruction
        // invalidates any reservation held by this hart.
        self.reset();
        start_addr != UNSET_VALUE && start_addr == addr.bitand(RES_SET_BITMASK)
    }
}

impl ReservationSet<Normal> {
    /// Return a proof-generating version of this ReservationSet.
    pub fn start_proof(&self) -> ReservationSet<Prove<'_>> {
        ReservationSet {
            start_addr: self.start_addr.start_proof(),
        }
    }
}

impl<M: backend::ManagerBase> NewState<M> for ReservationSet<M> {
    fn new() -> Self
    where
        M: backend::ManagerAlloc,
    {
        ReservationSet {
            start_addr: Cell::new(),
        }
    }
}

impl<M: backend::ManagerClone> CloneState for ReservationSet<M> {
    fn clone_state(&self) -> Self {
        Self {
            start_addr: self.start_addr.clone_state(),
        }
    }
}

impl<M, F> Foldable<F> for ReservationSet<M>
where
    M: backend::ManagerBase,
    F: Fold,
    Cell<u64, M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> F::Folded {
        self.start_addr.fold(builder)
    }
}

impl FromProof for ReservationSet<Verify> {
    fn from_proof<D: Deserialiser>(proof: D) -> SuspendedResult<D, Self> {
        let result = Cell::from_proof(proof)?;
        let result = result.map(|start_addr| Self { start_addr });
        Ok(result)
    }
}

impl<M: backend::ManagerSerialise> Encode for ReservationSet<M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.start_addr.encode(encoder)?;
        Ok(())
    }
}

impl<C, M: backend::ManagerDeserialise> Decode<C> for ReservationSet<M> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self {
            start_addr: Decode::decode(decoder)?,
        })
    }
}
