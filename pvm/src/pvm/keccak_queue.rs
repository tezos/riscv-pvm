// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Parallel keccak-256 request queue
//!
//! This module provides a FIFO queue of pending keccak-256 hash requests stored in
//! provable PVM state, together with a background worker (Normal mode only) that
//! pre-computes results so that [`keccak256_dequeue`] can return with minimal blocking.
//!
//! [`keccak256_dequeue`]: crate::pvm::tezos::SBI_TEZOS_KECCAK256_DEQUEUE

use std::fmt;
use std::sync::mpsc;
use std::thread;

use bincode::Decode;
use bincode::Encode;
use octez_riscv_data::clone::CloneState;
use octez_riscv_data::components::atom::Atom;
use octez_riscv_data::components::fifo_queue::FifoQueue;
use octez_riscv_data::mode::Modal;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use sha3::Digest;
use sha3::Keccak256;

use crate::pvm::tezos::MAX_PVM_MEMORY_ACCESS;

// ── KeccakRequest ────────────────────────────────────────────────────────────

/// A single pending keccak-256 hash request stored in the provable queue.
///
/// The message bytes are zero-padded to a fixed `MAX_PVM_MEMORY_ACCESS`-byte buffer so
/// that the type has a stable, fixed size in the Merkle tree.
#[derive(Clone, PartialEq, Eq, Encode, Decode)]
pub(crate) struct KeccakRequest {
    /// Actual length of the message (≤ `MAX_PVM_MEMORY_ACCESS`).
    pub(crate) len: u64,
    /// Message bytes, zero-padded to `MAX_PVM_MEMORY_ACCESS`.
    pub(crate) data: Box<[u8; MAX_PVM_MEMORY_ACCESS]>,
}

impl KeccakRequest {
    /// Create a new request from a byte slice.
    pub(crate) fn new(bytes: &[u8]) -> Self {
        debug_assert!(bytes.len() <= MAX_PVM_MEMORY_ACCESS);
        let mut data = Box::new([0u8; MAX_PVM_MEMORY_ACCESS]);
        data[..bytes.len()].copy_from_slice(bytes);
        Self {
            len: bytes.len() as u64,
            data,
        }
    }

    /// Return the message bytes as a slice.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.data[..self.len as usize]
    }
}

impl Default for KeccakRequest {
    fn default() -> Self {
        Self {
            len: 0,
            data: Box::new([0u8; MAX_PVM_MEMORY_ACCESS]),
        }
    }
}

impl CloneState for KeccakRequest {
    fn clone_state(&self) -> Self {
        self.clone()
    }
}

impl fmt::Debug for KeccakRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeccakRequest")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

// ── KeccakWorker ─────────────────────────────────────────────────────────────

/// Background thread that pre-computes keccak-256 hashes.
///
/// The worker is long-lived (one per PVM instance in Normal mode). Requests are sent via
/// a channel and results are received in FIFO order.
pub(crate) struct KeccakWorker {
    request_tx: mpsc::Sender<Vec<u8>>,
    result_rx: mpsc::Receiver<[u8; 32]>,
    // The JoinHandle is kept so the thread is not detached.
    _thread: thread::JoinHandle<()>,
}

impl KeccakWorker {
    /// Spawn the background worker thread.
    pub(crate) fn spawn() -> Self {
        let (request_tx, request_rx) = mpsc::channel::<Vec<u8>>();
        let (result_tx, result_rx) = mpsc::channel::<[u8; 32]>();

        let thread = thread::spawn(move || {
            while let Ok(data) = request_rx.recv() {
                let hash: [u8; 32] = Keccak256::digest(&data).into();
                if result_tx.send(hash).is_err() {
                    break;
                }
            }
        });

        Self {
            request_tx,
            result_rx,
            _thread: thread,
        }
    }

    pub(crate) fn send(&self, data: &[u8]) {
        // Ignore errors; if the channel is closed we fall back to sync computation.
        let _ = self.request_tx.send(data.to_vec());
    }

    pub(crate) fn recv(&self) -> Result<[u8; 32], mpsc::RecvError> {
        self.result_rx.recv()
    }
}

// ── KeccakWorkerCell ─────────────────────────────────────────────────────────

/// Wrapper around [`KeccakWorker`] that satisfies `Clone`, `PartialEq`, and `CloneState`.
///
/// Cloning always produces an empty cell (no worker). This is the correct semantics for
/// state snapshots: a cloned PVM state has no pre-computed results and falls back to
/// synchronous hashing on the first dequeue.
pub(crate) struct KeccakWorkerCell {
    pub(crate) worker: Option<KeccakWorker>,
}

impl Default for KeccakWorkerCell {
    fn default() -> Self {
        Self { worker: None }
    }
}

impl Clone for KeccakWorkerCell {
    fn clone(&self) -> Self {
        Self { worker: None }
    }
}

impl CloneState for KeccakWorkerCell {
    fn clone_state(&self) -> Self {
        Self { worker: None }
    }
}

impl PartialEq for KeccakWorkerCell {
    /// The worker is not part of observable state; two cells are always equal.
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for KeccakWorkerCell {}

impl fmt::Debug for KeccakWorkerCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeccakWorkerCell")
            .field("active", &self.worker.is_some())
            .finish()
    }
}

// ── CloneState for () ────────────────────────────────────────────────────────

// The unit type is the worker representation in Prove/Verify mode.
impl CloneState for () {
    fn clone_state(&self) -> Self {}
}

// ── KeccakWorkerTemplate ─────────────────────────────────────────────────────

/// Modal template selecting the worker cell in Normal mode and a ZST otherwise.
pub(crate) enum KeccakWorkerTemplate {}

impl Modal for KeccakWorkerTemplate {
    type Normal = KeccakWorkerCell;
    type Prove<'normal> = ();
    type Verify = ();
}

// ── KeccakWorkerMode ─────────────────────────────────────────────────────────

/// Mode-specific dispatch for the parallel keccak queue.
///
/// In **Normal** mode, `keccak_enqueue` sends work to the background thread and
/// `keccak_dequeue` blocks until the thread returns the result (falling back to
/// synchronous computation if no worker is present, e.g. after a state clone).
///
/// In **Prove/Verify** mode both operations are synchronous: `keccak_enqueue` is a
/// no-op and `keccak_dequeue` computes the hash immediately from the stored bytes.
///
/// The supertrait where clause guarantees that `Self::Select<KeccakWorkerTemplate>`
/// satisfies the traits needed by `Tezos<M>`'s `Default`, `CloneState`, and
/// `perfect_derive` impls.
pub(crate) trait KeccakWorkerMode: Mode
where
    Self::Select<KeccakWorkerTemplate>: Default + Clone + CloneState + PartialEq + Eq,
{
    /// Called after enqueueing. In Normal mode, dispatches to the background worker.
    fn keccak_enqueue(worker: &mut Self::Select<KeccakWorkerTemplate>, data: &[u8]);

    /// Called on dequeue. Returns the keccak-256 hash of the message.
    ///
    /// `stored_data` is the raw message bytes stored in the queue entry; it is used as
    /// a fallback when no pre-computed result is available.
    fn keccak_dequeue(
        worker: &mut Self::Select<KeccakWorkerTemplate>,
        stored_data: &[u8],
    ) -> [u8; 32];
}

impl KeccakWorkerMode for Normal {
    fn keccak_enqueue(cell: &mut KeccakWorkerCell, data: &[u8]) {
        let worker = cell.worker.get_or_insert_with(KeccakWorker::spawn);
        worker.send(data);
    }

    fn keccak_dequeue(cell: &mut KeccakWorkerCell, stored_data: &[u8]) -> [u8; 32] {
        // Try to receive a pre-computed result from the background worker.
        let result = cell.worker.as_ref().and_then(|w| w.recv().ok());
        if let Some(hash) = result {
            return hash;
        }
        // Worker not present (after clone) or dead: compute synchronously.
        cell.worker = None;
        Keccak256::digest(stored_data).into()
    }
}

impl<'a> KeccakWorkerMode for Prove<'a> {
    fn keccak_enqueue(_: &mut (), _: &[u8]) {}
    fn keccak_dequeue(_: &mut (), stored_data: &[u8]) -> [u8; 32] {
        Keccak256::digest(stored_data).into()
    }
}

impl KeccakWorkerMode for Verify {
    fn keccak_enqueue(_: &mut (), _: &[u8]) {}
    fn keccak_dequeue(_: &mut (), stored_data: &[u8]) -> [u8; 32] {
        Keccak256::digest(stored_data).into()
    }
}

// ── KeccakQueue type alias ───────────────────────────────────────────────────

/// The provable keccak request queue stored in [`Tezos<M>`].
///
/// Each entry is an [`Atom`]-wrapped [`KeccakRequest`] to integrate with the
/// modal proof machinery.
///
/// [`Tezos<M>`]: crate::pvm::tezos::Tezos
pub(crate) type KeccakQueue<M> = FifoQueue<Atom<KeccakRequest, M>, M>;
