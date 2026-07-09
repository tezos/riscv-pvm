// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! durable-storage-local helpers for rkyv serialisation of on-disk blobs.
//!
//! These mirror the shape of the bincode `serialise`/`deserialise` helpers that
//! previously lived in [`octez_riscv_data::serialisation`], but produce and
//! consume rkyv archives. They are intentionally kept local to durable-storage:
//! only the on-disk/blob format uses rkyv, while the shared `data` crate (and
//! therefore the PVM) keeps bincode for its Merkle hashing and proofs.

use rkyv::Archive;
use rkyv::Deserialize;
use rkyv::Serialize;
use rkyv::api::high::HighDeserializer;
use rkyv::api::high::HighSerializer;
use rkyv::api::high::HighValidator;
use rkyv::bytecheck::CheckBytes;
use rkyv::rancor::Error;
use rkyv::ser::allocator::ArenaHandle;
use rkyv::util::AlignedVec;

/// Serialise `value` into its rkyv byte representation.
pub(crate) fn rkyv_serialise<T>(value: &T) -> Result<AlignedVec, Error>
where
    T: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, Error>>,
{
    rkyv::to_bytes::<Error>(value)
}

/// Deserialise a value of type `T` from its rkyv byte representation.
///
/// The input slice is first copied into an [`AlignedVec`], because rkyv's
/// validated access requires the buffer to satisfy the archived type's
/// alignment, and blob/RocksDB slices have arbitrary alignment.
pub(crate) fn rkyv_deserialise<T>(bytes: &[u8]) -> Result<T, Error>
where
    T: Archive,
    T::Archived:
        for<'a> CheckBytes<HighValidator<'a, Error>> + Deserialize<T, HighDeserializer<Error>>,
{
    let mut aligned = AlignedVec::<16>::new();
    aligned.extend_from_slice(bytes);
    rkyv::from_bytes::<T, Error>(&aligned)
}
