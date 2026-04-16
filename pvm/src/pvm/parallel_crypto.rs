// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Generic parallel-crypto infrastructure
//!
//! This module provides [`CryptoJobQueue<A, M>`], a provable FIFO queue of
//! pending [`CryptoAlgorithm`] requests together with a non-provable background
//! worker (Normal mode only).  The same component powers both the keccak-256
//! hash queue and the secp256k1 signature-verification queue.

use std::fmt;
use std::sync::mpsc;
use std::thread;

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use libsecp256k1::Message;
use libsecp256k1::PublicKey;
use libsecp256k1::Signature as SecpSig;
use octez_riscv_data::clone::CloneState;
use octez_riscv_data::components::atom::Atom;
use octez_riscv_data::components::atom::AtomMode;
use octez_riscv_data::components::atom::CloneAtomMode;
use octez_riscv_data::components::atom::EncodeAtomMode;
use octez_riscv_data::components::fifo_queue::FifoQueue;
use octez_riscv_data::components::vector::CloneVectorMode;
use octez_riscv_data::components::vector::EncodeVectorMode;
use octez_riscv_data::components::vector::VectorMode;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::Unfold;
use octez_riscv_data::foldable::UnfoldError;
use octez_riscv_data::foldable::Unfoldable;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::Suspended;
use octez_riscv_data::merkle_proof::SuspendedResult;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Provable;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use sha3::Digest;
use sha3::Keccak256;

use crate::pvm::tezos::MAX_PVM_MEMORY_ACCESS;

// ── CryptoAlgorithm ──────────────────────────────────────────────────────────

/// A stateless, pure crypto algorithm that can be executed in a background thread.
///
/// Implementors define the request (`Input`) and response (`Output`) types, plus
/// the computation itself via [`execute`](CryptoAlgorithm::execute).
pub(crate) trait CryptoAlgorithm: Send + Sync + 'static {
    /// The request type stored in the provable queue.
    type Input: Clone
        + Encode
        + Decode<()>
        + Default
        + PartialEq
        + Eq
        + CloneState
        + Send
        + 'static;
    /// The result type produced by the background thread.
    type Output: Copy + Send + 'static;

    /// Compute the algorithm purely from `input`.  Called on the background thread
    /// in Normal mode, and inline in Prove/Verify mode.
    fn execute(input: &Self::Input) -> Self::Output;
}

// ── Request types ─────────────────────────────────────────────────────────────

/// Pending keccak-256 request: message bytes zero-padded to [`MAX_PVM_MEMORY_ACCESS`].
#[derive(Clone, PartialEq, Eq, Encode, Decode)]
pub(crate) struct KeccakRequest {
    pub(crate) len: u64,
    pub(crate) data: Box<[u8; MAX_PVM_MEMORY_ACCESS]>,
}

impl KeccakRequest {
    pub(crate) fn new(bytes: &[u8]) -> Self {
        debug_assert!(bytes.len() <= MAX_PVM_MEMORY_ACCESS);
        let mut data = Box::new([0u8; MAX_PVM_MEMORY_ACCESS]);
        data[..bytes.len()].copy_from_slice(bytes);
        Self { len: bytes.len() as u64, data }
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.data[..self.len as usize]
    }
}

impl Default for KeccakRequest {
    fn default() -> Self {
        Self { len: 0, data: Box::new([0u8; MAX_PVM_MEMORY_ACCESS]) }
    }
}

impl CloneState for KeccakRequest {
    fn clone_state(&self) -> Self { self.clone() }
}

impl fmt::Debug for KeccakRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeccakRequest").field("len", &self.len).finish_non_exhaustive()
    }
}

/// Pending secp256k1 signature-verification request.
#[derive(Clone, PartialEq, Eq, Encode, Decode)]
pub(crate) struct Secp256k1Request {
    pub(crate) public_key: [u8; 65],
    pub(crate) signature: [u8; 64],
    pub(crate) message_hash: [u8; 32],
}

impl Default for Secp256k1Request {
    fn default() -> Self {
        Self {
            public_key: [0u8; 65],
            signature: [0u8; 64],
            message_hash: [0u8; 32],
        }
    }
}

impl CloneState for Secp256k1Request {
    fn clone_state(&self) -> Self { self.clone() }
}

impl fmt::Debug for Secp256k1Request {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Secp256k1Request").finish_non_exhaustive()
    }
}

// ── Concrete algorithm implementations ───────────────────────────────────────

/// keccak-256: `Input = KeccakRequest`, `Output = [u8; 32]`.
pub(crate) struct KeccakAlgorithm;

impl CryptoAlgorithm for KeccakAlgorithm {
    type Input = KeccakRequest;
    type Output = [u8; 32];

    fn execute(input: &KeccakRequest) -> [u8; 32] {
        Keccak256::digest(input.as_bytes()).into()
    }
}

/// secp256k1-verify: `Input = Secp256k1Request`, `Output = bool`.
pub(crate) struct Secp256k1Algorithm;

impl CryptoAlgorithm for Secp256k1Algorithm {
    type Input = Secp256k1Request;
    type Output = bool;

    fn execute(input: &Secp256k1Request) -> bool {
        let Ok(pk) = PublicKey::parse(&input.public_key) else { return false };
        let Ok(sig) = SecpSig::parse_standard(&input.signature) else { return false };
        let msg = Message::parse(&input.message_hash);
        libsecp256k1::verify(&msg, &sig, &pk)
    }
}

// ── CryptoWorker / CryptoWorkerCell ──────────────────────────────────────────

/// Background thread that executes [`CryptoAlgorithm::execute`] for each request.
pub(crate) struct CryptoWorker<A: CryptoAlgorithm> {
    request_tx: mpsc::Sender<A::Input>,
    result_rx: mpsc::Receiver<A::Output>,
    _thread: thread::JoinHandle<()>,
}

impl<A: CryptoAlgorithm> CryptoWorker<A> {
    pub(crate) fn spawn() -> Self {
        let (request_tx, request_rx) = mpsc::channel::<A::Input>();
        let (result_tx, result_rx) = mpsc::channel::<A::Output>();

        let thread = thread::spawn(move || {
            while let Ok(input) = request_rx.recv() {
                let output = A::execute(&input);
                if result_tx.send(output).is_err() {
                    break;
                }
            }
        });

        Self { request_tx, result_rx, _thread: thread }
    }

    fn send(&self, input: A::Input) {
        let _ = self.request_tx.send(input);
    }

    fn recv(&self) -> Result<A::Output, mpsc::RecvError> {
        self.result_rx.recv()
    }
}

/// Wrapper that satisfies `Clone/PartialEq/Default/CloneState` without bounds on
/// `A`.  Cloning always produces an empty cell (worker = None).
pub(crate) struct CryptoWorkerCell<A: CryptoAlgorithm> {
    worker: Option<CryptoWorker<A>>,
}

impl<A: CryptoAlgorithm> Default for CryptoWorkerCell<A> {
    fn default() -> Self { Self { worker: None } }
}

impl<A: CryptoAlgorithm> Clone for CryptoWorkerCell<A> {
    fn clone(&self) -> Self { Self { worker: None } }
}

impl<A: CryptoAlgorithm> CloneState for CryptoWorkerCell<A> {
    fn clone_state(&self) -> Self { Self { worker: None } }
}

impl<A: CryptoAlgorithm> PartialEq for CryptoWorkerCell<A> {
    /// The worker is non-observable state; always equal.
    fn eq(&self, _: &Self) -> bool { true }
}

impl<A: CryptoAlgorithm> Eq for CryptoWorkerCell<A> {}

impl<A: CryptoAlgorithm> fmt::Debug for CryptoWorkerCell<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CryptoWorkerCell").field("active", &self.worker.is_some()).finish()
    }
}

// ── CryptoMode ───────────────────────────────────────────────────────────────

/// Mode-specific dispatch for a [`CryptoAlgorithm`].
///
/// In **Normal** mode `dispatch` submits work to the background thread and
/// `receive` blocks until the thread delivers a result (falling back to
/// synchronous computation if no worker is available, e.g. after a state clone).
///
/// In **Prove/Verify** mode `dispatch` is a no-op and `receive` calls
/// [`CryptoAlgorithm::execute`] synchronously.
///
/// Because the methods take the concrete `CryptoWorkerCell<A>` rather than a
/// modal associated type, using this trait introduces **no cascading bounds**.
pub(crate) trait CryptoMode<A: CryptoAlgorithm>: Mode {
    fn dispatch(cell: &mut CryptoWorkerCell<A>, input: A::Input);
    fn receive(cell: &mut CryptoWorkerCell<A>, fallback: &A::Input) -> A::Output;
}

impl<A: CryptoAlgorithm> CryptoMode<A> for Normal {
    fn dispatch(cell: &mut CryptoWorkerCell<A>, input: A::Input) {
        cell.worker.get_or_insert_with(CryptoWorker::spawn).send(input);
    }

    fn receive(cell: &mut CryptoWorkerCell<A>, fallback: &A::Input) -> A::Output {
        let result = cell.worker.as_ref().and_then(|w| w.recv().ok());
        if let Some(output) = result {
            return output;
        }
        // Worker not started (e.g. after state clone) or died — compute synchronously.
        cell.worker = None;
        A::execute(fallback)
    }
}

impl<'a, A: CryptoAlgorithm> CryptoMode<A> for Prove<'a> {
    fn dispatch(_: &mut CryptoWorkerCell<A>, _: A::Input) {}
    fn receive(_: &mut CryptoWorkerCell<A>, fallback: &A::Input) -> A::Output {
        A::execute(fallback)
    }
}

impl<A: CryptoAlgorithm> CryptoMode<A> for Verify {
    fn dispatch(_: &mut CryptoWorkerCell<A>, _: A::Input) {}
    fn receive(_: &mut CryptoWorkerCell<A>, fallback: &A::Input) -> A::Output {
        A::execute(fallback)
    }
}

// ── PvmCryptoMode ─────────────────────────────────────────────────────────────

/// Combined bound for all parallel crypto algorithms supported by the PVM.
/// Used in `handle_tezos`, `eval_max`, etc. instead of repeating each
/// `CryptoMode<…>` bound individually.
pub(crate) trait PvmCryptoMode:
    Mode + CryptoMode<KeccakAlgorithm> + CryptoMode<Secp256k1Algorithm>
{
}

impl PvmCryptoMode for Normal {}
impl<'a> PvmCryptoMode for Prove<'a> {}
impl PvmCryptoMode for Verify {}

// ── CryptoJobQueue ────────────────────────────────────────────────────────────

/// A provable FIFO queue of [`CryptoAlgorithm`] requests, combined with a
/// non-provable background worker cell.
///
/// The queue (stored as [`FifoQueue<Atom<A::Input, M>, M>`]) participates fully
/// in the Merkle tree.  The worker cell is excluded from all proof-related impls.
pub(crate) struct CryptoJobQueue<A: CryptoAlgorithm, M: Mode> {
    /// Provable part — included in Foldable / Encode / Decode.
    queue: FifoQueue<Atom<A::Input, M>, M>,
    /// Non-provable background worker — excluded from all Merkle-tree impls.
    worker: CryptoWorkerCell<A>,
}

impl<A: CryptoAlgorithm, M: AtomMode + VectorMode> CryptoJobQueue<A, M> {
    /// Enqueue `input`: dispatch it to the background worker (Normal mode) and
    /// store it in the provable queue.
    pub(crate) fn push(&mut self, input: A::Input)
    where
        M: CryptoMode<A>,
    {
        // Clone for the worker (moved through channel); original goes into queue.
        <M as CryptoMode<A>>::dispatch(&mut self.worker, input.clone());
        self.queue.enqueue(Atom::new(input));
    }

    /// Dequeue the front entry and return the algorithm output.
    ///
    /// In Normal mode the result is received from the background worker (blocking
    /// until ready).  In Prove/Verify mode it is computed synchronously from the
    /// stored request bytes.
    ///
    /// Returns `None` if the queue is empty.
    pub(crate) fn pop(&mut self) -> Option<A::Output>
    where
        M: CryptoMode<A>,
    {
        if self.queue.is_empty() {
            return None;
        }

        // Collect stored input for the synchronous fallback path.
        let stored: A::Input = {
            let front = self.queue.front().expect("queue is non-empty");
            // *front: Atom<A::Input, M>; **front: A::Input via Atom's Deref impl.
            (**front).clone()
        };
        // Release immutable borrow before advancing.
        self.queue.advance();

        Some(<M as CryptoMode<A>>::receive(&mut self.worker, &stored))
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

// ── Modal trait impls for CryptoJobQueue ──────────────────────────────────────

impl<A: CryptoAlgorithm, M: AtomMode + VectorMode> Default for CryptoJobQueue<A, M> {
    fn default() -> Self {
        Self { queue: FifoQueue::default(), worker: CryptoWorkerCell::default() }
    }
}

impl<A: CryptoAlgorithm, M: AtomMode + VectorMode> PartialEq for CryptoJobQueue<A, M>
where
    Atom<A::Input, M>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.queue == other.queue
        // worker excluded
    }
}

impl<A: CryptoAlgorithm, M: AtomMode + VectorMode> Eq for CryptoJobQueue<A, M> where
    Atom<A::Input, M>: Eq
{
}

impl<A: CryptoAlgorithm, M: CloneAtomMode + CloneVectorMode> Clone for CryptoJobQueue<A, M>
where
    Atom<A::Input, M>: Clone,
{
    fn clone(&self) -> Self {
        Self { queue: self.queue.clone(), worker: CryptoWorkerCell::default() }
    }
}

impl<A: CryptoAlgorithm, M: CloneAtomMode + CloneVectorMode> CloneState
    for CryptoJobQueue<A, M>
{
    fn clone_state(&self) -> Self {
        Self { queue: self.queue.clone_state(), worker: CryptoWorkerCell::default() }
    }
}

/// `CryptoJobQueue` folds to exactly the provable queue — the worker is excluded.
impl<A: CryptoAlgorithm, M, F> Foldable<F> for CryptoJobQueue<A, M>
where
    M: Mode,
    F: Fold,
    FifoQueue<Atom<A::Input, M>, M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> F::Folded {
        self.queue.fold(builder)
    }
}

impl<A: CryptoAlgorithm> Unfoldable for CryptoJobQueue<A, Normal>
where
    Atom<A::Input, Normal>: Unfoldable,
{
    fn unfold<U: Unfold>(src: U) -> Result<Self, UnfoldError> {
        let queue = FifoQueue::unfold(src)?;
        Ok(Self { queue, worker: CryptoWorkerCell::default() })
    }
}

impl<A: CryptoAlgorithm> FromProof for CryptoJobQueue<A, Verify>
where
    Atom<A::Input, Verify>: FromProof,
{
    fn from_proof<D: Deserialiser>(proof: D) -> SuspendedResult<D, Self> {
        let suspended = FifoQueue::<Atom<A::Input, Verify>, Verify>::from_proof(proof)?;
        Ok(suspended.map(|queue| Self { queue, worker: CryptoWorkerCell::default() }))
    }
}

impl<'n, A: CryptoAlgorithm> Provable<'n> for CryptoJobQueue<A, Normal>
where
    Atom<A::Input, Normal>: Provable<'n, Prover = Atom<A::Input, Prove<'n>>>,
{
    type Prover = CryptoJobQueue<A, Prove<'n>>;

    fn start_proof(&'n self) -> Self::Prover {
        CryptoJobQueue {
            queue: self.queue.start_proof(),
            worker: CryptoWorkerCell::default(),
        }
    }
}

impl<A: CryptoAlgorithm, M: EncodeAtomMode + EncodeVectorMode> Encode
    for CryptoJobQueue<A, M>
{
    fn encode<E: bincode::enc::Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.queue.encode(encoder)
        // worker excluded
    }
}

impl<A: CryptoAlgorithm> Decode<()> for CryptoJobQueue<A, Normal> {
    fn decode<D: Decoder<Context = ()>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self { queue: FifoQueue::decode(decoder)?, worker: CryptoWorkerCell::default() })
    }
}

// ── Type aliases ─────────────────────────────────────────────────────────────

pub(crate) type KeccakJobQueue<M> = CryptoJobQueue<KeccakAlgorithm, M>;
pub(crate) type Secp256k1JobQueue<M> = CryptoJobQueue<Secp256k1Algorithm, M>;
