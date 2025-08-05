// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::mem::offset_of;
use std::ops::Index;
use std::ops::RangeInclusive;

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use cranelift::codegen::ir;
use cranelift::prelude::FunctionBuilder;
use cranelift::prelude::isa::TargetFrontendConfig;
use octez_riscv_data::clone::CloneState;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::seq_tree::IndexableSeqAsTree;
use octez_riscv_data::merkle_proof;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::Suspended;
use octez_riscv_data::merkle_proof::SuspendedResult;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use perfect_derive::perfect_derive;

use super::Address;
use super::MemoryConfig;
use super::address_to_page_index;
use crate::array_utils::boxed_from_fn;
use crate::instruction_context::ICB;
use crate::instruction_context::arithmetic::Arithmetic;
use crate::jit::state_context::JitStateContext;
use crate::machine_state::MachineCoreState;
use crate::state::NewState;
use crate::state_backend::Cell;
use crate::state_backend::CellProj;
use crate::state_backend::ManagerAlloc;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerClone;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerSerialise;
use crate::state_backend::ManagerWrite;
use crate::state_backend::NarrowlySized;
use crate::state_backend::owned_backend::Owned;
use crate::state_backend::proof_backend::merkle::MERKLE_ARITY;
use crate::state_context::StateContext;
use crate::state_context::projection::ArrayProj;
use crate::state_context::projection::ArrayProjParam;
use crate::state_context::projection::BoxProj;
use crate::state_context::projection::Projection;
use crate::state_context::projection::TypeCons;

/// Tracks access permissions for each page
#[perfect_derive(Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct PagePermissions<const PAGES: usize, M: ManagerBase> {
    pages: Box<[Cell<bool, M>; PAGES]>,
}

impl<const PAGES: usize, M: ManagerBase> PagePermissions<PAGES, M> {
    /// Check if the memory at `address..address+length` can be accessed.
    ///
    /// # Safety
    ///
    /// The address and length must be valid for an address space consisting of a number of `PAGES`.
    /// This function is not defined for address and length combinations which are out of bounds.
    #[inline]
    pub unsafe fn can_access(&self, pages: RangeInclusive<u64>) -> bool
    where
        M: ManagerRead,
    {
        for page in pages {
            if unsafe { !self.pages.get_unchecked(page as usize).read() } {
                return false;
            }
        }

        true
    }

    /// Same as [`Self::can_access`], but slightly faster. Requires additional invariants to be upheld.
    /// The generic parameter `E` is used to specify the type of the element being accessed. It
    /// also determines the length of the access.
    ///
    /// # Safety
    ///
    /// The length must be non-zero and less than the page size. Otherwise, same as
    /// [`Self::can_access`].
    #[inline]
    pub unsafe fn can_access_narrow<E>(&self, address: Address) -> bool
    where
        E: NarrowlySized,
        M: ManagerRead,
    {
        let start_page = address_to_page_index(address);
        if unsafe { !self.pages.get_unchecked(start_page).read() } {
            return false;
        }

        let end_address = address
            .wrapping_add(E::NARROW_SIZE.get() as Address)
            .wrapping_sub(1);

        let end_page = address_to_page_index(end_address);
        unsafe { self.pages.get_unchecked(end_page).read() }
    }

    /// Change the access permissions for the given range.
    pub fn modify_access(&mut self, pages: RangeInclusive<u64>, accessible: bool)
    where
        M: ManagerWrite,
    {
        pages.filter(|&page| page < PAGES as u64).for_each(|page| {
            self.pages[page as usize].write(accessible);
        })
    }

    /// Reset access permissions on all pages.
    pub fn reset(&mut self)
    where
        M: ManagerWrite,
    {
        self.pages.iter_mut().for_each(|page| page.write(false));
    }
}

impl<const PAGES: usize> PagePermissions<PAGES, Normal> {
    /// Return a proof-generating version of this PagePermissions.
    pub fn start_proof(&self) -> PagePermissions<PAGES, Prove<'_>> {
        let Ok(pages) = self
            .pages
            .iter()
            .map(Cell::start_proof)
            .collect::<Vec<_>>()
            .try_into()
        else {
            unreachable!("Collecting into an array of the same length should always succeed")
        };

        PagePermissions { pages }
    }
}

impl<const PAGES: usize, M: ManagerBase> NewState<M> for PagePermissions<PAGES, M> {
    fn new() -> Self
    where
        M: ManagerAlloc,
    {
        PagePermissions {
            pages: boxed_from_fn(|| Cell::new()),
        }
    }
}

impl<const PAGES: usize, M: ManagerClone> CloneState for PagePermissions<PAGES, M> {
    fn clone_state(&self) -> Self {
        Self {
            pages: self.pages.clone_state(),
        }
    }
}

impl<C, const PAGES: usize> Decode<C> for PagePermissions<PAGES, Normal> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let pages = Decode::decode(decoder)?;
        Ok(Self { pages })
    }
}

impl<const PAGES: usize, M: ManagerSerialise> Encode for PagePermissions<PAGES, M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.pages.encode(encoder)
    }
}

impl<const PAGES: usize, M, F> Foldable<F> for PagePermissions<PAGES, M>
where
    M: ManagerBase,
    F: Fold,
    Cell<bool, M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> F::Folded {
        let page_generator = |idx| self.pages.index(idx);
        IndexableSeqAsTree::new(PAGES, MERKLE_ARITY, &page_generator).fold(builder)
    }
}
impl<const PAGES: usize> FromProof for PagePermissions<PAGES, Verify> {
    fn from_proof<D: Deserialiser>(proof: D) -> SuspendedResult<D, Self> {
        let result = merkle_proof::Many::<_, MERKLE_ARITY, PAGES>::from_proof(proof)?;
        let result = result.map(|pages| Self {
            pages: pages.into_boxed_array(),
        });
        Ok(result)
    }
}

/// TODO
pub struct PagePermissionsCons<const PAGES: usize>;

impl<const PAGES: usize> TypeCons for PagePermissionsCons<PAGES> {
    type Applied<MC: MemoryConfig, M: ManagerBase> = PagePermissions<PAGES, M>;
}

/// TODO
type InnerProj<const PAGES: usize> = BoxProj<ArrayProj<CellProj<bool>, PAGES>>;

/// TODO
pub struct PagePermissionsProj<const PAGES: usize>;

impl<const PAGES: usize> Projection for PagePermissionsProj<PAGES> {
    type Subject = PagePermissionsCons<PAGES>;

    type Target = bool;

    type Parameter<SC: StateContext + ?Sized> = SC::Value<u64>;

    fn project_ref<'a, MC: MemoryConfig, M: ManagerRead + ManagerWrite + 'a>(
        state: &'a <Self::Subject as TypeCons>::Applied<MC, M>,
        param: Self::Parameter<MachineCoreState<MC, M>>,
    ) -> &'a Self::Target {
        InnerProj::project_ref::<MC, M>(&state.pages, ArrayProjParam {
            index: param,
            inner_param: (),
        })
    }

    fn project_read<'a, MC: MemoryConfig, M: ManagerRead + ManagerWrite + 'a>(
        state: &'a <Self::Subject as TypeCons>::Applied<MC, M>,
        param: Self::Parameter<MachineCoreState<MC, M>>,
    ) -> Self::Target
    where
        Self::Target: Copy,
    {
        InnerProj::project_read::<MC, M>(&state.pages, ArrayProjParam {
            index: param,
            inner_param: (),
        })
    }

    fn project_write<'a, MC: MemoryConfig, M: ManagerRead + ManagerWrite + 'a>(
        state: &'a mut <Self::Subject as TypeCons>::Applied<MC, M>,
        param: Self::Parameter<MachineCoreState<MC, M>>,
        value: Self::Target,
    ) {
        InnerProj::project_write::<MC, M>(
            &mut state.pages,
            ArrayProjParam {
                index: param,
                inner_param: (),
            },
            value,
        )
    }

    fn build_owned_pointer_offset<MC: MemoryConfig, SC: JitStateContext>(
        target_config: &TargetFrontendConfig,
        builder: &mut FunctionBuilder,
        base: ir::Value,
        offset: ir::immediates::Offset32,
        param: Self::Parameter<SC>,
    ) -> (ir::Value, ir::immediates::Offset32) {
        let field_offset = offset_of!(PagePermissions<PAGES, Owned>, pages)
            .try_into()
            .expect("Offset should fit into positive i64");
        let offset = offset
            .try_add_i64(field_offset)
            .expect("Offset should not overflow");

        InnerProj::<PAGES>::build_owned_pointer_offset::<MC, SC>(
            target_config,
            builder,
            base,
            offset,
            ArrayProjParam {
                index: param,
                inner_param: (),
            },
        )
    }
}
