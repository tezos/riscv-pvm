// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! The leaf serialisation codec used by durable storage.
//!
//! Durable storage hashes and proves its state with the [`rkyv`](octez_riscv_data::codec::Rkyv)
//! leaf codec (the PVM stays on bincode). To keep that choice in one place, this module re-exports
//! the fold types specialised to that codec. Modules fold/prove/verify durable-storage state by
//! importing these aliases (`use crate::codec::HashFold`, ...) rather than the generic
//! `octez_riscv_data` fold types, so the codec is fixed uniformly without threading a type
//! parameter through every impl.

/// The leaf codec used throughout durable storage.
pub(crate) use octez_riscv_data::codec::Rkyv as Codec;

/// [`octez_riscv_data::hash::HashFold`] specialised to the durable-storage [`Codec`].
pub(crate) type HashFold = octez_riscv_data::hash::HashFold<Codec>;

/// [`octez_riscv_data::hash::PartialHashFold`] specialised to the durable-storage [`Codec`].
pub(crate) type PartialHashFold = octez_riscv_data::hash::PartialHashFold<Codec>;

/// [`octez_riscv_data::merkle_proof::proof_tree::MerkleProofFold`] specialised to the
/// durable-storage [`Codec`].
pub(crate) type MerkleProofFold =
    octez_riscv_data::merkle_proof::proof_tree::MerkleProofFold<Codec>;
