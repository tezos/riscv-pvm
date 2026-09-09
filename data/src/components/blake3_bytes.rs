// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Length-committed BLAKE3 hashing scheme for a byte array.
//!
//! A value's hash is `H(v) = combine(H_len, blake3::hash(bytes))` — the length committed as its
//! own leaf, combined with the BLAKE3 hash of the raw bytes. Committing the length is what makes
//! partial proofs of a value's bytes possible at all: it pins the canonical shape of BLAKE3's
//! internal chunk tree, so a proof may collapse any untouched subtree to a single chaining value
//! without giving an attacker room to reinterpret it at another length. The full design, the
//! security argument, and the numbered invariants (`I1`..`I8`) are in
//! `docs/state-framework/blake3-bytes.mdx`.
//!
//! This module currently defines the committed hash only. The proof machinery that rides the
//! chunk tree, and the modal component that folds into the state pipelines, follow.

use crate::hash::Hash;

/// BLAKE3 chunk length: the leaf granularity of the internal chunk tree (1024 bytes).
pub const CHUNK_LEN: usize = blake3::CHUNK_LEN;

/// Commit a value's length as its own leaf: `H_len = blake3(len_le_bytes)`.
///
/// The length is hashed as its `u64` little-endian encoding (no bincode framing) so it is a
/// fixed, consensus-relevant preimage. This is the leaf that pins the value's length in the
/// committed hash (see [`hash_value`]); dropping it is what let the predecessor length-free
/// scheme be forged by a cross-span graft (invariant I3).
fn hash_len(len: usize) -> Hash {
    Hash::hash_bytes(&(len as u64).to_le_bytes())
}

/// Normal-mode hash of a byte array under the length-committed BLAKE3 scheme.
///
/// `H(v) = combine(H_len, blake3::hash(bytes))` — the length leaf [`hash_len`] combined with
/// the BLAKE3 hash of the raw bytes (invariant I8). The explicit length commitment pins the
/// value's length, which is what makes within-value partial proofs with *unrestricted* blinding
/// secure. This is **not** equal to `blake3::hash(bytes)`: it wraps it in one more combine.
pub fn hash_value(bytes: &[u8]) -> Hash {
    Hash::combine_hashes([hash_len(bytes.len()), Hash::hash_bytes(bytes)])
}

#[cfg(test)]
mod tests;
