// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Space accounting harness for committed durable storage.
//!
//! This exists to answer one question: of the bytes a commit occupies on disk, how many are still
//! needed to serve it?
//!
//! The two column families behave completely differently, which is the whole point of measuring:
//!
//! - Values are keyed by your key, so writing a key again makes the previous value an obsolete
//!   version that compaction discards. The value column family stays proportional to the key space
//!   that is live in it.
//! - Merkle node bodies are keyed by content hash, so every version of every node is a distinct
//!   key that nothing ever deletes. The blob column family accumulates every node ever written.
//!
//! So for the blob column family this harness reports a **live** figure (the bodies reachable from
//! the committed root, found by walking the stored representation) and a **dead** figure
//! (everything else in it). Dead bytes are what garbage collection could reclaim and what nothing
//! reclaims today.
//!
//! Keys and values are derived from their indices rather than stored, so a run at ten million keys
//! does not need a ten-million-entry key list in memory. Derivation is deterministic, so a given
//! seed and shape always produce the same state.
//!
//! Prepopulating a large registry costs far more than the commits being measured, so the base state
//! is recorded in the repository directory and reused when the shape matches. Point `--repo-dir` at
//! a volume with room to spare and successive runs skip straight to the commit sequence. Each run
//! resets the repository to that base state before measuring, so reusing a directory does not carry
//! earlier runs' commits into the figures.

mod measure;
mod report;
mod sample;
mod scenario;

pub use scenario::SpaceConfig;
