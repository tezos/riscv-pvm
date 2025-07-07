// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2024 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use crate::machine_state::csregisters;
use crate::machine_state::memory::Address;
use crate::machine_state::registers;
use crate::machine_state::registers::XValue;
use crate::machine_state::reservation_set;
use crate::machine_state::reservation_set::ReservationSet;
use crate::state::NewState;
use crate::state_backend as backend;
use crate::state_backend::Atom;
use crate::state_backend::Cell;
use crate::state_backend::CellProj;
use crate::state_context::StateContext;
use crate::state_context::projection::MachineCoreCons;
use crate::state_context::projection::impl_projection;

/// RISC-V hart state
pub struct HartState<M: backend::ManagerBase> {
    /// Integer registers
    pub xregisters: registers::XRegisters<M>,

    /// Floating-point number registers
    pub fregisters: registers::FRegisters<M>,

    /// Control and state registers
    pub csregisters: csregisters::CSRegisters<M>,

    /// Program counter
    pub pc: Cell<Address, M>,

    /// Reservation set address
    pub reservation_set: ReservationSet<M>,
}

/// Layout of [HartState]
pub type HartStateLayout = (
    registers::XRegistersLayout,
    registers::FRegistersLayout,
    csregisters::CSRegistersLayout,
    Atom<Address>,                         // Program counter layout
    reservation_set::ReservationSetLayout, // Reservation set layout
);

impl<M: backend::ManagerBase> HartState<M> {
    /// Bind the hart state to the given allocated space.
    pub fn bind(space: backend::AllocatedOf<HartStateLayout, M>) -> Self {
        Self {
            xregisters: registers::XRegisters::bind(space.0),
            fregisters: registers::FRegisters::bind(space.1),
            csregisters: csregisters::CSRegisters::bind(space.2),
            pc: space.3,
            reservation_set: ReservationSet::bind(space.4),
        }
    }

    /// Given a manager morphism `f : &M -> N`, return the layout's allocated structure containing
    /// the constituents of `N` that were produced from the constituents of `&M`.
    pub fn struct_ref<'a, F: backend::FnManager<backend::Ref<'a, M>>>(
        &'a self,
    ) -> backend::AllocatedOf<HartStateLayout, F::Output> {
        (
            self.xregisters.struct_ref::<F>(),
            self.fregisters.struct_ref::<F>(),
            self.csregisters.struct_ref::<F>(),
            self.pc.struct_ref::<F>(),
            self.reservation_set.struct_ref::<F>(),
        )
    }

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

impl<M: backend::ManagerBase> NewState<M> for HartState<M> {
    fn new() -> Self
    where
        M: backend::ManagerAlloc,
    {
        Self {
            xregisters: registers::XRegisters::new(),
            fregisters: registers::FRegisters::new(),
            csregisters: csregisters::CSRegisters::new(),
            pc: Cell::new(),
            reservation_set: ReservationSet::new(),
        }
    }
}

impl<M: backend::ManagerClone> Clone for HartState<M> {
    fn clone(&self) -> Self {
        Self {
            xregisters: self.xregisters.clone(),
            fregisters: self.fregisters.clone(),
            csregisters: self.csregisters.clone(),
            pc: self.pc.clone(),
            reservation_set: self.reservation_set.clone(),
        }
    }
}

impl_projection! {
    projection ProgramCounterProj {
        subject = MachineCoreCons,
        target_projection = CellProj<XValue>,
        path = hart.pc,
    }
}

/// Update the program counter in the given state context.
#[inline]
pub(crate) fn write_pc<SC: StateContext + ?Sized>(state: &mut SC, value: SC::X64) {
    state.write_proj::<ProgramCounterProj>((), value);
}
