// SPDX-FileCopyrightText: 2024 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Merkle trees used for proof generation by the PVM

use std::num::NonZeroUsize;

// TODO RV-322: Choose optimal Merkleisation parameters for main memory.
/// Size of the Merkle leaf used for Merkleising [`DynArrays`].
///
/// [`DynArrays`]: [`crate::state_backend::layout::DynArray`]
pub const MERKLE_LEAF_SIZE: NonZeroUsize = NonZeroUsize::new(4096).unwrap();

// TODO RV-322: Choose optimal Merkleisation parameters for main memory.
/// Arity of the Merkle tree used for Merkleising [`DynArrays`].
///
/// [`DynArrays`]: [`crate::state_backend::layout::DynArray`]
pub const MERKLE_ARITY: usize = 4;
