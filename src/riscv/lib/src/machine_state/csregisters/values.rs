// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2024-2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use super::CSRegisters;
use super::effects::NoEffect;
use super::effects::handle_csr_effect;
use super::root::RootCSRegister;
use crate::bits::Bits64;
use crate::state::NewState;
use crate::state_backend::AllocatedOf;
use crate::state_backend::Cell;
use crate::state_backend::CommitmentLayout;
use crate::state_backend::EffectCell;
use crate::state_backend::EffectCellLayout;
use crate::state_backend::FnManager;
use crate::state_backend::Layout;
use crate::state_backend::ManagerAlloc;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerReadWrite;
use crate::state_backend::ManagerSerialise;
use crate::state_backend::ManagerWrite;
use crate::state_backend::PartialHashError;
use crate::state_backend::ProofLayout;
use crate::state_backend::ProofTree;
use crate::state_backend::Ref;
use crate::state_backend::RefProofGenOwnedAlloc;
use crate::state_backend::RefVerifierAlloc;
use crate::state_backend::VerifierAlloc;
use crate::state_backend::hash::Hash;
use crate::state_backend::hash::HashError;
use crate::state_backend::owned_backend::Owned;
use crate::state_backend::proof_backend::merkle::AccessInfoAggregatable;
use crate::state_backend::proof_backend::merkle::MerkleTree;
use crate::state_backend::proof_backend::proof::deserialiser::Deserialiser;
use crate::state_backend::proof_backend::proof::deserialiser::Partial;
use crate::state_backend::proof_backend::proof::deserialiser::Suspended;
use crate::state_backend::proof_layout;
use crate::state_backend::verify_backend;
use crate::storage::binary;

/// Representation of a value in a CSR
pub type CSRRepr = u64;

/// Value of a Control or State register
#[derive(
    Copy,
    Clone,
    Debug,
    derive_more::Display,
    derive_more::From,
    derive_more::Into,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
)]
#[repr(transparent)]
pub struct CSRValue(CSRRepr);

impl CSRValue {
    /// Access the underlying representation.
    pub fn repr(self) -> CSRRepr {
        self.0
    }
}

impl Bits64 for CSRValue {
    const WIDTH: usize = CSRRepr::WIDTH;

    fn from_bits(value: u64) -> Self {
        Self(value)
    }

    fn to_bits(&self) -> u64 {
        self.repr()
    }
}

type RawValue<M> = EffectCell<CSRRepr, NoEffect, M>;

/// Values of all control and state registers
pub type CSRValues<M> = CSRValuesF<RawValue<M>>;

impl<M: ManagerBase> CSRValues<M> {
    /// Bind the CSR values to the given allocated regions.
    pub fn bind(space: AllocatedOf<CSRValuesLayout, M>) -> Self {
        space.map(EffectCell::bind)
    }

    /// Given a manager morphism `f : &M -> N`, return the layout's allocated structure containing
    /// the constituents of `N` that were produced from the constituents of `&M`.
    pub fn struct_ref<'a, F: FnManager<Ref<'a, M>>>(
        &'a self,
    ) -> AllocatedOf<CSRValuesLayout, F::Output> {
        self.as_ref().map(|raw| raw.struct_ref::<F>())
    }
}

impl<M: ManagerBase> NewState<M> for CSRValues<M> {
    fn new(manager: &mut M) -> Self
    where
        M: ManagerAlloc,
    {
        let manager = std::cell::RefCell::new(manager);
        CSRValuesF::new_with(|| RawValue::new(*manager.borrow_mut()))
    }
}

impl<M: ManagerBase> CSRegisters<M> {
    /// Perform a general read of a CSR.
    #[inline(always)]
    pub(super) fn general_raw_read(&self, csr: RootCSRegister) -> CSRRepr
    where
        M: ManagerRead,
    {
        self.registers.select_ref(csr, RawValue::read)
    }

    /// Perform a general write of a CSR.
    #[inline(always)]
    pub(super) fn general_raw_write(&mut self, csr: RootCSRegister, value: CSRRepr)
    where
        M: ManagerWrite,
    {
        let effect = self.registers.select_mut(csr, |raw| raw.write(value));

        handle_csr_effect(self, effect);
    }

    /// Perform a general replace of a CSR.
    #[inline(always)]
    pub(super) fn general_raw_replace(&mut self, csr: RootCSRegister, value: CSRRepr) -> CSRRepr
    where
        M: ManagerReadWrite,
    {
        let (old_value, effect) = self.registers.select_mut(csr, |raw| raw.replace(value));

        handle_csr_effect(self, effect);

        old_value
    }
}

/// Layout for the values of CSRs
pub struct CSRValuesLayout;

impl Layout for CSRValuesLayout {
    type Allocated<M: ManagerBase> = CSRValuesF<AllocatedOf<EffectCellLayout<CSRRepr>, M>>;
}

impl CommitmentLayout for CSRValuesLayout {
    fn state_hash<M: ManagerSerialise>(state: AllocatedOf<Self, M>) -> Result<Hash, HashError> {
        Hash::blake2b_hash(state)
    }
}

impl ProofLayout for CSRValuesLayout {
    fn to_merkle_tree(state: RefProofGenOwnedAlloc<Self>) -> Result<MerkleTree, HashError> {
        let serialised = binary::serialise(&state)?;
        MerkleTree::make_merkle_leaf(serialised, state.aggregate_access_info())
    }

    fn to_verifier_alloc<D: Deserialiser>(
        proof: D,
    ) -> Result<D::Suspended<VerifierAlloc<Self>>, proof_layout::FromProofError> {
        fn make_absent() -> AllocatedOf<CSRValuesLayout, verify_backend::Verifier> {
            CSRValuesF::new_with(|| Cell::bind(verify_backend::Region::Absent))
        }

        let leaf = proof.into_leaf::<AllocatedOf<CSRValuesLayout, Owned>>()?;
        let handler = leaf.map(move |leaf| match leaf {
            Partial::Absent | Partial::Blinded(_) => make_absent(),
            Partial::Present(data) => data.map(Cell::from_owned),
        });

        Ok(handler)
    }

    fn partial_state_hash(
        state: RefVerifierAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        match state.try_map::<_, PartialHashError>(Cell::<_, Owned>::try_from) {
            Ok(state) => Ok(Self::state_hash(state)?),
            Err(_) => proof.partial_hash_leaf(),
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CSRValuesF<Raw> {
    pub cycle: Raw,
    pub time: Raw,
    pub instret: Raw,
    pub hpmcounter3: Raw,
    pub hpmcounter4: Raw,
    pub hpmcounter5: Raw,
    pub hpmcounter6: Raw,
    pub hpmcounter7: Raw,
    pub hpmcounter8: Raw,
    pub hpmcounter9: Raw,
    pub hpmcounter10: Raw,
    pub hpmcounter11: Raw,
    pub hpmcounter12: Raw,
    pub hpmcounter13: Raw,
    pub hpmcounter14: Raw,
    pub hpmcounter15: Raw,
    pub hpmcounter16: Raw,
    pub hpmcounter17: Raw,
    pub hpmcounter18: Raw,
    pub hpmcounter19: Raw,
    pub hpmcounter20: Raw,
    pub hpmcounter21: Raw,
    pub hpmcounter22: Raw,
    pub hpmcounter23: Raw,
    pub hpmcounter24: Raw,
    pub hpmcounter25: Raw,
    pub hpmcounter26: Raw,
    pub hpmcounter27: Raw,
    pub hpmcounter28: Raw,
    pub hpmcounter29: Raw,
    pub hpmcounter30: Raw,
    pub hpmcounter31: Raw,
    pub fcsr: Raw,
}

impl<Raw> CSRValuesF<Raw> {
    /// Create a new CSR values structure. The given functions are used to initialise each CSR
    /// value.
    #[inline]
    fn new_with(mut make_raw: impl FnMut() -> Raw) -> Self {
        CSRValuesF::<()>::default().map(|_| make_raw())
    }

    /// Transform each field representing a CSR value into a different type.
    #[inline]
    fn map<Raw2>(self, mut map_raw: impl FnMut(Raw) -> Raw2) -> CSRValuesF<Raw2> {
        CSRValuesF {
            cycle: map_raw(self.cycle),
            time: map_raw(self.time),
            instret: map_raw(self.instret),
            hpmcounter3: map_raw(self.hpmcounter3),
            hpmcounter4: map_raw(self.hpmcounter4),
            hpmcounter5: map_raw(self.hpmcounter5),
            hpmcounter6: map_raw(self.hpmcounter6),
            hpmcounter7: map_raw(self.hpmcounter7),
            hpmcounter8: map_raw(self.hpmcounter8),
            hpmcounter9: map_raw(self.hpmcounter9),
            hpmcounter10: map_raw(self.hpmcounter10),
            hpmcounter11: map_raw(self.hpmcounter11),
            hpmcounter12: map_raw(self.hpmcounter12),
            hpmcounter13: map_raw(self.hpmcounter13),
            hpmcounter14: map_raw(self.hpmcounter14),
            hpmcounter15: map_raw(self.hpmcounter15),
            hpmcounter16: map_raw(self.hpmcounter16),
            hpmcounter17: map_raw(self.hpmcounter17),
            hpmcounter18: map_raw(self.hpmcounter18),
            hpmcounter19: map_raw(self.hpmcounter19),
            hpmcounter20: map_raw(self.hpmcounter20),
            hpmcounter21: map_raw(self.hpmcounter21),
            hpmcounter22: map_raw(self.hpmcounter22),
            hpmcounter23: map_raw(self.hpmcounter23),
            hpmcounter24: map_raw(self.hpmcounter24),
            hpmcounter25: map_raw(self.hpmcounter25),
            hpmcounter26: map_raw(self.hpmcounter26),
            hpmcounter27: map_raw(self.hpmcounter27),
            hpmcounter28: map_raw(self.hpmcounter28),
            hpmcounter29: map_raw(self.hpmcounter29),
            hpmcounter30: map_raw(self.hpmcounter30),
            hpmcounter31: map_raw(self.hpmcounter31),
            fcsr: map_raw(self.fcsr),
        }
    }

    /// Attempt to transform each field representing a CSR value into a different type.
    #[inline]
    fn try_map<Raw2, Err>(
        self,
        mut try_map_raw: impl FnMut(Raw) -> Result<Raw2, Err>,
    ) -> Result<CSRValuesF<Raw2>, Err> {
        Ok(CSRValuesF {
            cycle: try_map_raw(self.cycle)?,
            time: try_map_raw(self.time)?,
            instret: try_map_raw(self.instret)?,
            hpmcounter3: try_map_raw(self.hpmcounter3)?,
            hpmcounter4: try_map_raw(self.hpmcounter4)?,
            hpmcounter5: try_map_raw(self.hpmcounter5)?,
            hpmcounter6: try_map_raw(self.hpmcounter6)?,
            hpmcounter7: try_map_raw(self.hpmcounter7)?,
            hpmcounter8: try_map_raw(self.hpmcounter8)?,
            hpmcounter9: try_map_raw(self.hpmcounter9)?,
            hpmcounter10: try_map_raw(self.hpmcounter10)?,
            hpmcounter11: try_map_raw(self.hpmcounter11)?,
            hpmcounter12: try_map_raw(self.hpmcounter12)?,
            hpmcounter13: try_map_raw(self.hpmcounter13)?,
            hpmcounter14: try_map_raw(self.hpmcounter14)?,
            hpmcounter15: try_map_raw(self.hpmcounter15)?,
            hpmcounter16: try_map_raw(self.hpmcounter16)?,
            hpmcounter17: try_map_raw(self.hpmcounter17)?,
            hpmcounter18: try_map_raw(self.hpmcounter18)?,
            hpmcounter19: try_map_raw(self.hpmcounter19)?,
            hpmcounter20: try_map_raw(self.hpmcounter20)?,
            hpmcounter21: try_map_raw(self.hpmcounter21)?,
            hpmcounter22: try_map_raw(self.hpmcounter22)?,
            hpmcounter23: try_map_raw(self.hpmcounter23)?,
            hpmcounter24: try_map_raw(self.hpmcounter24)?,
            hpmcounter25: try_map_raw(self.hpmcounter25)?,
            hpmcounter26: try_map_raw(self.hpmcounter26)?,
            hpmcounter27: try_map_raw(self.hpmcounter27)?,
            hpmcounter28: try_map_raw(self.hpmcounter28)?,
            hpmcounter29: try_map_raw(self.hpmcounter29)?,
            hpmcounter30: try_map_raw(self.hpmcounter30)?,
            hpmcounter31: try_map_raw(self.hpmcounter31)?,
            fcsr: try_map_raw(self.fcsr)?,
        })
    }

    /// Create a referencing structure of the CSR values.
    #[inline]
    fn as_ref(&self) -> CSRValuesF<&Raw> {
        CSRValuesF {
            cycle: &self.cycle,
            time: &self.time,
            instret: &self.instret,
            hpmcounter3: &self.hpmcounter3,
            hpmcounter4: &self.hpmcounter4,
            hpmcounter5: &self.hpmcounter5,
            hpmcounter6: &self.hpmcounter6,
            hpmcounter7: &self.hpmcounter7,
            hpmcounter8: &self.hpmcounter8,
            hpmcounter9: &self.hpmcounter9,
            hpmcounter10: &self.hpmcounter10,
            hpmcounter11: &self.hpmcounter11,
            hpmcounter12: &self.hpmcounter12,
            hpmcounter13: &self.hpmcounter13,
            hpmcounter14: &self.hpmcounter14,
            hpmcounter15: &self.hpmcounter15,
            hpmcounter16: &self.hpmcounter16,
            hpmcounter17: &self.hpmcounter17,
            hpmcounter18: &self.hpmcounter18,
            hpmcounter19: &self.hpmcounter19,
            hpmcounter20: &self.hpmcounter20,
            hpmcounter21: &self.hpmcounter21,
            hpmcounter22: &self.hpmcounter22,
            hpmcounter23: &self.hpmcounter23,
            hpmcounter24: &self.hpmcounter24,
            hpmcounter25: &self.hpmcounter25,
            hpmcounter26: &self.hpmcounter26,
            hpmcounter27: &self.hpmcounter27,
            hpmcounter28: &self.hpmcounter28,
            hpmcounter29: &self.hpmcounter29,
            hpmcounter30: &self.hpmcounter30,
            hpmcounter31: &self.hpmcounter31,
            fcsr: &self.fcsr,
        }
    }

    /// Select a field representing a CSR value and apply a function to it, returing its result.
    #[inline(always)]
    fn select_ref<R>(&self, csr: RootCSRegister, fold_raw: impl FnOnce(&Raw) -> R) -> R {
        match csr {
            RootCSRegister::cycle => fold_raw(&self.cycle),
            RootCSRegister::time => fold_raw(&self.time),
            RootCSRegister::instret => fold_raw(&self.instret),
            RootCSRegister::hpmcounter3 => fold_raw(&self.hpmcounter3),
            RootCSRegister::hpmcounter4 => fold_raw(&self.hpmcounter4),
            RootCSRegister::hpmcounter5 => fold_raw(&self.hpmcounter5),
            RootCSRegister::hpmcounter6 => fold_raw(&self.hpmcounter6),
            RootCSRegister::hpmcounter7 => fold_raw(&self.hpmcounter7),
            RootCSRegister::hpmcounter8 => fold_raw(&self.hpmcounter8),
            RootCSRegister::hpmcounter9 => fold_raw(&self.hpmcounter9),
            RootCSRegister::hpmcounter10 => fold_raw(&self.hpmcounter10),
            RootCSRegister::hpmcounter11 => fold_raw(&self.hpmcounter11),
            RootCSRegister::hpmcounter12 => fold_raw(&self.hpmcounter12),
            RootCSRegister::hpmcounter13 => fold_raw(&self.hpmcounter13),
            RootCSRegister::hpmcounter14 => fold_raw(&self.hpmcounter14),
            RootCSRegister::hpmcounter15 => fold_raw(&self.hpmcounter15),
            RootCSRegister::hpmcounter16 => fold_raw(&self.hpmcounter16),
            RootCSRegister::hpmcounter17 => fold_raw(&self.hpmcounter17),
            RootCSRegister::hpmcounter18 => fold_raw(&self.hpmcounter18),
            RootCSRegister::hpmcounter19 => fold_raw(&self.hpmcounter19),
            RootCSRegister::hpmcounter20 => fold_raw(&self.hpmcounter20),
            RootCSRegister::hpmcounter21 => fold_raw(&self.hpmcounter21),
            RootCSRegister::hpmcounter22 => fold_raw(&self.hpmcounter22),
            RootCSRegister::hpmcounter23 => fold_raw(&self.hpmcounter23),
            RootCSRegister::hpmcounter24 => fold_raw(&self.hpmcounter24),
            RootCSRegister::hpmcounter25 => fold_raw(&self.hpmcounter25),
            RootCSRegister::hpmcounter26 => fold_raw(&self.hpmcounter26),
            RootCSRegister::hpmcounter27 => fold_raw(&self.hpmcounter27),
            RootCSRegister::hpmcounter28 => fold_raw(&self.hpmcounter28),
            RootCSRegister::hpmcounter29 => fold_raw(&self.hpmcounter29),
            RootCSRegister::hpmcounter30 => fold_raw(&self.hpmcounter30),
            RootCSRegister::hpmcounter31 => fold_raw(&self.hpmcounter31),
            RootCSRegister::fcsr => fold_raw(&self.fcsr),
        }
    }

    /// Select a field representing a CSR value and apply a function to it, returing its result.
    #[inline(always)]
    fn select_mut<R>(&mut self, csr: RootCSRegister, fold_raw: impl FnOnce(&mut Raw) -> R) -> R {
        match csr {
            RootCSRegister::cycle => fold_raw(&mut self.cycle),
            RootCSRegister::time => fold_raw(&mut self.time),
            RootCSRegister::instret => fold_raw(&mut self.instret),
            RootCSRegister::hpmcounter3 => fold_raw(&mut self.hpmcounter3),
            RootCSRegister::hpmcounter4 => fold_raw(&mut self.hpmcounter4),
            RootCSRegister::hpmcounter5 => fold_raw(&mut self.hpmcounter5),
            RootCSRegister::hpmcounter6 => fold_raw(&mut self.hpmcounter6),
            RootCSRegister::hpmcounter7 => fold_raw(&mut self.hpmcounter7),
            RootCSRegister::hpmcounter8 => fold_raw(&mut self.hpmcounter8),
            RootCSRegister::hpmcounter9 => fold_raw(&mut self.hpmcounter9),
            RootCSRegister::hpmcounter10 => fold_raw(&mut self.hpmcounter10),
            RootCSRegister::hpmcounter11 => fold_raw(&mut self.hpmcounter11),
            RootCSRegister::hpmcounter12 => fold_raw(&mut self.hpmcounter12),
            RootCSRegister::hpmcounter13 => fold_raw(&mut self.hpmcounter13),
            RootCSRegister::hpmcounter14 => fold_raw(&mut self.hpmcounter14),
            RootCSRegister::hpmcounter15 => fold_raw(&mut self.hpmcounter15),
            RootCSRegister::hpmcounter16 => fold_raw(&mut self.hpmcounter16),
            RootCSRegister::hpmcounter17 => fold_raw(&mut self.hpmcounter17),
            RootCSRegister::hpmcounter18 => fold_raw(&mut self.hpmcounter18),
            RootCSRegister::hpmcounter19 => fold_raw(&mut self.hpmcounter19),
            RootCSRegister::hpmcounter20 => fold_raw(&mut self.hpmcounter20),
            RootCSRegister::hpmcounter21 => fold_raw(&mut self.hpmcounter21),
            RootCSRegister::hpmcounter22 => fold_raw(&mut self.hpmcounter22),
            RootCSRegister::hpmcounter23 => fold_raw(&mut self.hpmcounter23),
            RootCSRegister::hpmcounter24 => fold_raw(&mut self.hpmcounter24),
            RootCSRegister::hpmcounter25 => fold_raw(&mut self.hpmcounter25),
            RootCSRegister::hpmcounter26 => fold_raw(&mut self.hpmcounter26),
            RootCSRegister::hpmcounter27 => fold_raw(&mut self.hpmcounter27),
            RootCSRegister::hpmcounter28 => fold_raw(&mut self.hpmcounter28),
            RootCSRegister::hpmcounter29 => fold_raw(&mut self.hpmcounter29),
            RootCSRegister::hpmcounter30 => fold_raw(&mut self.hpmcounter30),
            RootCSRegister::hpmcounter31 => fold_raw(&mut self.hpmcounter31),
            RootCSRegister::fcsr => fold_raw(&mut self.fcsr),
        }
    }
}

impl<Raw> AccessInfoAggregatable for CSRValuesF<Raw>
where
    Raw: AccessInfoAggregatable + serde::Serialize,
{
    fn aggregate_access_info(&self) -> bool {
        let children = [
            self.cycle.aggregate_access_info(),
            self.time.aggregate_access_info(),
            self.instret.aggregate_access_info(),
            self.hpmcounter3.aggregate_access_info(),
            self.hpmcounter4.aggregate_access_info(),
            self.hpmcounter5.aggregate_access_info(),
            self.hpmcounter6.aggregate_access_info(),
            self.hpmcounter7.aggregate_access_info(),
            self.hpmcounter8.aggregate_access_info(),
            self.hpmcounter9.aggregate_access_info(),
            self.hpmcounter10.aggregate_access_info(),
            self.hpmcounter11.aggregate_access_info(),
            self.hpmcounter12.aggregate_access_info(),
            self.hpmcounter13.aggregate_access_info(),
            self.hpmcounter14.aggregate_access_info(),
            self.hpmcounter15.aggregate_access_info(),
            self.hpmcounter16.aggregate_access_info(),
            self.hpmcounter17.aggregate_access_info(),
            self.hpmcounter18.aggregate_access_info(),
            self.hpmcounter19.aggregate_access_info(),
            self.hpmcounter20.aggregate_access_info(),
            self.hpmcounter21.aggregate_access_info(),
            self.hpmcounter22.aggregate_access_info(),
            self.hpmcounter23.aggregate_access_info(),
            self.hpmcounter24.aggregate_access_info(),
            self.hpmcounter25.aggregate_access_info(),
            self.hpmcounter26.aggregate_access_info(),
            self.hpmcounter27.aggregate_access_info(),
            self.hpmcounter28.aggregate_access_info(),
            self.hpmcounter29.aggregate_access_info(),
            self.hpmcounter30.aggregate_access_info(),
            self.hpmcounter31.aggregate_access_info(),
            self.fcsr.aggregate_access_info(),
        ];
        children.iter().any(|&x| x)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use strum::IntoEnumIterator;

    use super::*;

    /// Ensure that [`CSRValues::fold_ref`] and [`CSRValuesF::fold_mut`] refer to the same CSR
    /// value field provided the same [`RootCSRegister`].
    #[test]
    fn fold_ref_mut_consistent() {
        let counter = AtomicUsize::new(0);

        let mut example = CSRValuesF::new_with(|| counter.fetch_add(1, Ordering::SeqCst));

        for csr in RootCSRegister::iter() {
            let lhs = example.select_ref(csr, |x| *x);
            let rhs = example.select_mut(csr, |x| *x);
            assert_eq!(lhs, rhs);
        }
    }

    /// Ensure that [`CSRValues::as_ref`] obtained correct references.
    #[test]
    fn as_ref_consistent() {
        let counter = AtomicUsize::new(0);

        let example = CSRValuesF::new_with(|| counter.fetch_add(1, Ordering::SeqCst));

        let example_copy = example.as_ref().map(|x| *x);

        assert_eq!(example, example_copy);
    }
}
