// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Tezos-specific host functions for the PVM

use std::cmp::min;

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use ed25519_dalek::Signature;
use ed25519_dalek::Signer;
use ed25519_dalek::SigningKey;
use ed25519_dalek::VerifyingKey;
use libsecp256k1::Message;
use libsecp256k1::PublicKey;
use libsecp256k1::Signature as SecpSig;
use octez_riscv_data::clone::CloneState;
use octez_riscv_data::components::atom::Atom;
use octez_riscv_data::components::atom::CloneAtomMode;
use octez_riscv_data::components::atom::EncodeAtomMode;
use octez_riscv_data::components::data_space::DataSpaceMode;
use octez_riscv_data::components::vector::CloneVectorMode;
use octez_riscv_data::components::vector::EncodeVectorMode;
use octez_riscv_data::components::vector::VectorMode;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::NodeFold;
use octez_riscv_data::foldable::NodeUnfold;
use octez_riscv_data::foldable::Unfold;
use octez_riscv_data::foldable::UnfoldError;
use octez_riscv_data::foldable::Unfoldable;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::DeserialiserNode;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::SuspendedResult;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Provable;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use perfect_derive::perfect_derive;
use sha3::Digest;
use sha3::Keccak256;
use tezos_smart_rollup_constants::core::MAX_INPUT_MESSAGE_SIZE;
use tezos_smart_rollup_constants::riscv::REVEAL_DATA_MAX_SIZE;
use tezos_smart_rollup_constants::riscv::REVEAL_REQUEST_MAX_SIZE;
use tezos_smart_rollup_constants::riscv::SBI_TEZOS_BLAKE2B_HASH256;
use tezos_smart_rollup_constants::riscv::SBI_TEZOS_ED25519_SIGN;
use tezos_smart_rollup_constants::riscv::SBI_TEZOS_ED25519_VERIFY;
use tezos_smart_rollup_constants::riscv::SBI_TEZOS_INBOX_NEXT;
use tezos_smart_rollup_constants::riscv::SBI_TEZOS_KECCAK256_HASH;
use tezos_smart_rollup_constants::riscv::SBI_TEZOS_REVEAL;
use tezos_smart_rollup_constants::riscv::SBI_TEZOS_SECP256K1_VERIFY;
use tezos_smart_rollup_constants::riscv::SBI_TEZOS_WRITE_OUTPUT;
use tezos_smart_rollup_constants::riscv::SbiError;

/// Maximum size of pvm memory access by a host function in bytes
/// To limit size of proofs in refutation games
pub const MAX_PVM_MEMORY_ACCESS: usize = 4096;
pub const MAX_DURABLE_VALUE_IO: usize = 4096;

pub const SBI_TEZOS_DURABLE_REGISTRY_LEN: u64 = 0x1000;
pub const SBI_TEZOS_DURABLE_REGISTRY_RESIZE_TICK: u64 = 0x1001;
pub const SBI_TEZOS_DURABLE_REGISTRY_COPY_DATABASE: u64 = 0x1002;
pub const SBI_TEZOS_DURABLE_REGISTRY_MOVE_DATABASE: u64 = 0x1003;
pub const SBI_TEZOS_DURABLE_REGISTRY_CLEAR_DATABASE: u64 = 0x1004;
pub const SBI_TEZOS_DURABLE_DATABASE_EXISTS: u64 = 0x1010;
pub const SBI_TEZOS_DURABLE_DATABASE_DELETE: u64 = 0x1011;
pub const SBI_TEZOS_DURABLE_DATABASE_VALUE_LENGTH: u64 = 0x1012;
pub const SBI_TEZOS_DURABLE_DATABASE_READ: u64 = 0x1013;
pub const SBI_TEZOS_DURABLE_DATABASE_SET: u64 = 0x1014;
pub const SBI_TEZOS_DURABLE_DATABASE_WRITE: u64 = 0x1015;
pub const SBI_TEZOS_DURABLE_DATABASE_HASH: u64 = 0x1016;

// FIXME: Move these constants to the tezos_smart_rollup_constants crate (riscv module).
/// Enqueue a keccak-256 hash request; returns immediately.
pub const SBI_TEZOS_KECCAK256_ENQUEUE: u64 = 0x0c;
/// Dequeue the oldest keccak-256 result; blocks in Normal mode until the result is ready.
pub const SBI_TEZOS_KECCAK256_DEQUEUE: u64 = 0x0d;
/// Enqueue a secp256k1 signature-verification request; returns immediately.
pub const SBI_TEZOS_SECP256K1_ENQUEUE: u64 = 0x0e;
/// Dequeue the oldest secp256k1 verification result (1 = valid, 0 = invalid).
pub const SBI_TEZOS_SECP256K1_DEQUEUE: u64 = 0x0f;

use octez_riscv_data::components::atom::AtomMode;

use super::PvmStatus;
use super::parallel_crypto::CryptoMode;
use super::parallel_crypto::KeccakAlgorithm;
use super::parallel_crypto::KeccakJobQueue;
use super::parallel_crypto::KeccakRequest;
use super::parallel_crypto::PvmCryptoMode;
use super::parallel_crypto::Secp256k1Algorithm;
use super::parallel_crypto::Secp256k1JobQueue;
use super::parallel_crypto::Secp256k1Request;
use super::outbox::Outbox;
use super::outbox::OutboxMessage;
use super::reveals::RevealRequest;
use crate::machine_state::MachineCoreState;
use crate::machine_state::memory::Memory;
use crate::machine_state::memory::MemoryConfig;
use crate::machine_state::registers::XRegisters;
use crate::machine_state::registers::XValue;
use crate::machine_state::registers::a0;
use crate::machine_state::registers::a1;
use crate::machine_state::registers::a2;
use crate::machine_state::registers::a3;
use crate::machine_state::registers::a4;
use crate::machine_state::registers::a5;
use crate::machine_state::registers::a6;
use crate::pvm::durable_storage::RuntimeDurableStorage;
use crate::pvm::durable_storage::RuntimeError as DurableStorageRuntimeError;

/// Tezos-specific fields of the PVM.
#[perfect_derive(Clone, PartialEq, Eq)]
pub struct Tezos<M: Mode> {
    pub(crate) outbox: Outbox<M>,
    pub(crate) reveal_request: RevealRequest<M>,
    pub(crate) tick: Atom<u64, M>,
    pub(crate) message_counter: Atom<u64, M>,
    pub(crate) level: Atom<Option<u32>, M>,
    pub(crate) status: Atom<PvmStatus, M>,
    /// Parallel keccak-256 job queue (provable queue + background worker).
    pub(crate) keccak_job: KeccakJobQueue<M>,
    /// Parallel secp256k1-verify job queue (provable queue + background worker).
    pub(crate) secp256k1_job: Secp256k1JobQueue<M>,
}

impl<M: AtomMode + VectorMode> Default for Tezos<M> {
    fn default() -> Self {
        Self {
            outbox: Outbox::<M>::default(),
            reveal_request: RevealRequest::default(),
            tick: Atom::default(),
            message_counter: Atom::default(),
            level: Atom::default(),
            status: Atom::default(),
            keccak_job: KeccakJobQueue::default(),
            secp256k1_job: Secp256k1JobQueue::default(),
        }
    }
}

impl<'normal> Provable<'normal> for Tezos<Normal> {
    type Prover = Tezos<Prove<'normal>>;

    fn start_proof(&'normal self) -> Self::Prover {
        Tezos {
            outbox: self.outbox.start_proof(),
            reveal_request: self.reveal_request.start_proof(),
            tick: self.tick.start_proof(),
            message_counter: self.message_counter.start_proof(),
            level: self.level.start_proof(),
            status: self.status.start_proof(),
            keccak_job: self.keccak_job.start_proof(),
            secp256k1_job: self.secp256k1_job.start_proof(),
        }
    }
}

impl<M: CloneAtomMode + CloneVectorMode> CloneState for Tezos<M> {
    fn clone_state(&self) -> Self {
        Self {
            outbox: self.outbox.clone_state(),
            reveal_request: self.reveal_request.clone_state(),
            tick: self.tick.clone_state(),
            message_counter: self.message_counter.clone_state(),
            level: self.level.clone_state(),
            status: self.status.clone_state(),
            keccak_job: self.keccak_job.clone_state(),
            secp256k1_job: self.secp256k1_job.clone_state(),
        }
    }
}

impl<M, F> Foldable<F> for Tezos<M>
where
    M: Mode,
    F: Fold,
    Outbox<M>: Foldable<F>,
    RevealRequest<M>: Foldable<F>,
    Atom<PvmStatus, M>: Foldable<F>,
    Atom<Option<u32>, M>: Foldable<F>,
    Atom<u64, M>: Foldable<F>,
    KeccakJobQueue<M>: Foldable<F>,
    Secp256k1JobQueue<M>: Foldable<F>,
{
    fn fold(&self, builder: F) -> F::Folded {
        let mut builder = builder.into_node_fold();

        builder.add(&self.outbox);
        builder.add(&self.reveal_request);
        builder.add(&self.tick);
        builder.add(&self.message_counter);
        builder.add(&self.level);
        builder.add(&self.status);
        builder.add(&self.keccak_job);
        builder.add(&self.secp256k1_job);
        // worker cells excluded from provable state.

        builder.done()
    }
}

impl Unfoldable for Tezos<Normal> {
    fn unfold<U: Unfold>(src: U) -> Result<Self, UnfoldError> {
        let mut src = src.into_node()?;

        let outbox = src.next_branch()?;
        let reveal_request = src.next_branch()?;
        let tick = src.next_branch()?;
        let message_counter = src.next_branch()?;
        let level = src.next_branch()?;
        let status = src.next_branch()?;
        let keccak_job = src.next_branch()?;
        let secp256k1_job = src.next_branch()?;

        src.done(Self {
            outbox,
            reveal_request,
            tick,
            message_counter,
            level,
            status,
            keccak_job,
            secp256k1_job,
        })
    }
}

impl FromProof for Tezos<Verify> {
    fn from_proof<D: Deserialiser>(proof: D) -> SuspendedResult<D, Self> {
        let proof = proof.into_node()?;

        let (proof, outbox) = proof.next_branch()?;
        let (proof, reveal_request) = proof.next_branch()?;
        let (proof, tick) = proof.next_branch()?;
        let (proof, message_counter) = proof.next_branch()?;
        let (proof, level) = proof.next_branch()?;
        let (proof, status) = proof.next_branch()?;
        let (proof, keccak_job) = proof.next_branch()?;
        let (proof, secp256k1_job) = proof.next_branch()?;

        proof.done(Self {
            outbox,
            reveal_request,
            tick,
            message_counter,
            level,
            status,
            keccak_job,
            secp256k1_job,
        })
    }
}

impl<M: EncodeAtomMode + EncodeVectorMode> Encode for Tezos<M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.outbox.encode(encoder)?;
        self.reveal_request.encode(encoder)?;
        self.tick.encode(encoder)?;
        self.message_counter.encode(encoder)?;
        self.level.encode(encoder)?;
        self.status.encode(encoder)?;
        self.keccak_job.encode(encoder)?;
        self.secp256k1_job.encode(encoder)?;
        // worker cells excluded from serialisation.
        Ok(())
    }
}

impl Decode<()> for Tezos<Normal> {
    fn decode<D: Decoder<Context = ()>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self {
            outbox: Decode::decode(decoder)?,
            reveal_request: Decode::decode(decoder)?,
            tick: Decode::decode(decoder)?,
            message_counter: Decode::decode(decoder)?,
            level: Decode::decode(decoder)?,
            status: Decode::decode(decoder)?,
            keccak_job: Decode::decode(decoder)?,
            secp256k1_job: Decode::decode(decoder)?,
        })
    }
}

/// Write the SBI error code as the return value.
#[inline]
fn sbi_return_error<M: AtomMode>(xregisters: &mut XRegisters<M>, code: SbiError) {
    xregisters.write(a0, code as i64 as u64);
}

/// Write an arbitrary value as single return value.
#[inline]
fn sbi_return1<M: AtomMode>(xregisters: &mut XRegisters<M>, value: XValue) {
    // The SBI caller interprets the return value as a [i64]. We don't want the value to be
    // interpreted as negative because that indicates an error.
    if (value as i64) < 0 {
        return sbi_return_error(xregisters, SbiError::Failed);
    }

    xregisters.write(a0, value);
}

/// Run the given closure `inner` and write the corresponding SBI results to `machine`.
#[inline]
fn sbi_wrap<MC, M, F>(machine: &mut MachineCoreState<MC, M>, inner: F)
where
    MC: MemoryConfig,
    M: AtomMode,
    F: FnOnce(&mut MachineCoreState<MC, M>) -> Result<XValue, SbiError>,
{
    match inner(machine) {
        Ok(value) => sbi_return1(&mut machine.hart.xregisters, value),
        Err(error) => sbi_return_error(&mut machine.hart.xregisters, error),
    }
}

#[inline]
fn map_durable_storage_error(error: DurableStorageRuntimeError) -> SbiError {
    match error {
        DurableStorageRuntimeError::NotSupported => SbiError::NotSupported,
        DurableStorageRuntimeError::Durable(error) => match error {
            octez_riscv_durable_storage::errors::Error::InvalidArgument(_) => {
                SbiError::InvalidParam
            }
            octez_riscv_durable_storage::errors::Error::Operational(_) => SbiError::Failed,
        },
        DurableStorageRuntimeError::InvalidArgument(_) => SbiError::InvalidParam,
        DurableStorageRuntimeError::Operational(_) => SbiError::Failed,
    }
}

#[inline]
fn checked_usize(value: u64) -> Result<usize, SbiError> {
    usize::try_from(value).map_err(|_| SbiError::InvalidParam)
}

#[inline]
fn read_guest_bytes<MC, M>(
    machine: &mut MachineCoreState<MC, M>,
    address: u64,
    length: usize,
    max_length: usize,
) -> Result<Vec<u8>, SbiError>
where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode,
{
    if length > max_length {
        return Err(SbiError::InvalidParam);
    }

    let mut bytes = vec![0u8; length];
    machine.main_memory.read_all(address, &mut bytes)?;
    Ok(bytes)
}

/// Provide input information to the machine. Returns `false` in case the
/// machine wasn't expecting any input, otherwise returns `true`.
pub fn provide_input<MC, M>(
    status: &mut Atom<PvmStatus, M>,
    machine: &mut MachineCoreState<MC, M>,
    level: u32,
    counter: u32,
    payload: &[u8],
) -> bool
where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode,
{
    // This method should only do something when we're waiting for input.
    match status.read() {
        PvmStatus::WaitingForInput => {}
        _ => return false,
    }

    // We're evaluating again after this.
    status.write(PvmStatus::Evaluating);

    sbi_wrap(machine, |machine| {
        // These arguments should have been set by the previous SBI call.
        let arg_buffer_addr = machine.hart.xregisters.read(a0);
        let arg_buffer_size = machine.hart.xregisters.read(a1);
        let arg_level_addr = machine.hart.xregisters.read(a2);
        let arg_counter_addr = machine.hart.xregisters.read(a3);

        // The SBI caller expects the payload to be returned at [phys_dest_addr]
        // with at maximum [max_buffer_size] bytes written.
        let max_buffer_size = payload.len().min(arg_buffer_size as usize).min(
            // If we were to allow more data to be passed, we could run into problems with proof
            // sizes for inputs.
            MAX_INPUT_MESSAGE_SIZE,
        );

        machine
            .main_memory
            .write_all(arg_buffer_addr, &payload[..max_buffer_size])?;
        machine.main_memory.write(arg_level_addr, level)?;
        machine.main_memory.write(arg_counter_addr, counter)?;

        // At the moment, this case is unlikely to occur because we cap [max_buffer_size] at
        // [MAX_INPUT_MESSAGE_SIZE].
        Ok(max_buffer_size as u64)
    });

    true
}

/// Provide reveal data in response to a reveal request. Returns `false`
/// if the machine is not expecting reveal.
pub fn provide_reveal_response<MC, M>(
    status: &mut Atom<PvmStatus, M>,
    machine: &mut MachineCoreState<MC, M>,
    reveal_data: &[u8],
) -> bool
where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode,
{
    // This method should only do something when we're waiting for reveal.
    if status.read() != PvmStatus::WaitingForReveal {
        return false;
    }

    // We're evaluating again after this.
    status.write(PvmStatus::Evaluating);

    sbi_wrap(machine, |machine| {
        // These arguments should have been set by the previous SBI call.
        let arg_buffer_addr = machine.hart.xregisters.read(a2);
        let arg_buffer_size = machine.hart.xregisters.read(a3);

        let memory_write_size = min(
            REVEAL_DATA_MAX_SIZE,
            min(arg_buffer_size as usize, reveal_data.len()),
        );

        machine
            .main_memory
            .write_all(arg_buffer_addr, &reveal_data[..memory_write_size])?;

        Ok(memory_write_size as u64)
    });

    true
}

/// Handle a [SBI_TEZOS_INBOX_NEXT] call.
#[inline]
fn handle_tezos_inbox_next<M>(status: &mut Atom<PvmStatus, M>)
where
    M: AtomMode,
{
    // Prepare the EE state for an input tick.
    status.write(PvmStatus::WaitingForInput);
}

/// Handle a [SBI_TEZOS_WRITE_OUTPUT] call.
#[inline]
fn handle_tezos_write_output<MC, M>(
    machine: &mut MachineCoreState<MC, M>,
    outbox: &mut Outbox<M>,
    level: &Atom<Option<u32>, M>,
) where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode,
{
    sbi_wrap(machine, |machine| {
        // The outbox can't accept messages before the first inbox message is received
        let Some(current_level) = level.read() else {
            return Err(SbiError::FullOutbox);
        };

        let buffer_size = machine.hart.xregisters.read(a1) as usize;
        let mut msg = OutboxMessage::new(buffer_size)?;

        // Only read the message if it is not larger than `MAX_OUTPUT_SIZE`
        let buffer_addr = machine.hart.xregisters.read(a0);
        machine.main_memory.read_all(buffer_addr, &mut msg)?;

        outbox.write_message(msg, current_level)?;

        Ok(0)
    })
}

/// Produce a Ed25519 signature.
#[inline]
fn handle_tezos_ed25519_sign<MC, M>(machine: &mut MachineCoreState<MC, M>) -> Result<u64, SbiError>
where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode,
{
    let arg_sk_addr = machine.hart.xregisters.read(a0);
    let arg_msg_addr = machine.hart.xregisters.read(a1);
    let arg_msg_len = machine.hart.xregisters.read(a2);
    let arg_sig_addr = machine.hart.xregisters.read(a3);

    let mut sk_bytes = [0u8; 32];
    machine.main_memory.read_all(arg_sk_addr, &mut sk_bytes)?;
    let sk = SigningKey::try_from(sk_bytes.as_slice()).map_err(|_| SbiError::Failed)?;
    sk_bytes.fill(0);

    let mut msg_bytes = vec![0; arg_msg_len as usize];
    machine.main_memory.read_all(arg_msg_addr, &mut msg_bytes)?;

    let sig = sk.sign(msg_bytes.as_slice());
    let sig_bytes: [u8; 64] = sig.to_bytes();
    machine.main_memory.write_all(arg_sig_addr, &sig_bytes)?;

    Ok(sig_bytes.len() as u64)
}

/// Verify a Ed25519 signature.
#[inline]
fn handle_tezos_ed25519_verify<MC, M>(
    machine: &mut MachineCoreState<MC, M>,
) -> Result<u64, SbiError>
where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode,
{
    let arg_pk_addr = machine.hart.xregisters.read(a0);
    let arg_sig_addr = machine.hart.xregisters.read(a1);
    let arg_msg_addr = machine.hart.xregisters.read(a2);
    let arg_msg_len = machine.hart.xregisters.read(a3);

    let mut pk_bytes = [0u8; 32];
    machine.main_memory.read_all(arg_pk_addr, &mut pk_bytes)?;

    let mut sig_bytes = [0u8; 64];
    machine.main_memory.read_all(arg_sig_addr, &mut sig_bytes)?;

    let mut msg_bytes = vec![0u8; arg_msg_len as usize];
    machine.main_memory.read_all(arg_msg_addr, &mut msg_bytes)?;

    let pk = VerifyingKey::try_from(pk_bytes.as_slice()).map_err(|_| SbiError::Failed)?;
    let sig = Signature::from_slice(sig_bytes.as_slice()).map_err(|_| SbiError::Failed)?;
    let valid = pk.verify_strict(msg_bytes.as_slice(), &sig).is_ok();

    Ok(valid as u64)
}

/// Compute a BLAKE2B 256-bit digest.
#[inline]
fn handle_tezos_blake2b_hash256<MC, M>(
    machine: &mut MachineCoreState<MC, M>,
) -> Result<u64, SbiError>
where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode,
{
    let arg_out_addr = machine.hart.xregisters.read(a0);
    let arg_msg_addr = machine.hart.xregisters.read(a1);
    let arg_msg_len = machine.hart.xregisters.read(a2);

    let mut msg_bytes = vec![0u8; arg_msg_len as usize];
    machine.main_memory.read_all(arg_msg_addr, &mut msg_bytes)?;

    let hash = tezos_crypto_rs::blake2b::digest_256(msg_bytes.as_slice());
    machine
        .main_memory
        .write_all(arg_out_addr, hash.as_slice())?;

    Ok(hash.len() as u64)
}

/// Verify a Secp256k1 signature.
#[inline]
fn handle_tezos_secp256k1_verify<MC, M>(
    machine: &mut MachineCoreState<MC, M>,
) -> Result<u64, SbiError>
where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode,
{
    let arg_pk_addr = machine.hart.xregisters.read(a0);
    let arg_sig_addr = machine.hart.xregisters.read(a1);
    let arg_msg_addr = machine.hart.xregisters.read(a2);

    let pk_bytes: [u8; 65] = machine.main_memory.read(arg_pk_addr)?;
    let sig_bytes: [u8; 64] = machine.main_memory.read(arg_sig_addr)?;
    let msg_bytes: [u8; 32] = machine.main_memory.read(arg_msg_addr)?;

    let pk = PublicKey::parse(&pk_bytes).map_err(|_| SbiError::Failed)?;
    let sig = SecpSig::parse_standard(&sig_bytes).map_err(|_| SbiError::Failed)?;
    let msg = Message::parse(&msg_bytes);
    let valid = libsecp256k1::verify(&msg, &sig, &pk);

    Ok(valid as u64)
}

/// Compute a Keccak-256 digest.
#[inline]
fn handle_tezos_keccak256_hash<MC, M>(
    machine: &mut MachineCoreState<MC, M>,
) -> Result<u64, SbiError>
where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode,
{
    let arg_out_addr = machine.hart.xregisters.read(a0);
    let arg_msg_addr = machine.hart.xregisters.read(a1);
    let arg_msg_len = machine.hart.xregisters.read(a2);

    if (arg_msg_len as usize) > MAX_PVM_MEMORY_ACCESS {
        return Err(SbiError::InvalidParam);
    }

    let mut msg_bytes = vec![0u8; arg_msg_len as usize];
    machine.main_memory.read_all(arg_msg_addr, &mut msg_bytes)?;
    let hash: [u8; 32] = Keccak256::digest(&msg_bytes).into();
    machine.main_memory.write(arg_out_addr, hash)?;

    Ok(hash.len() as u64)
}

#[inline]
fn handle_tezos_durable_registry_len<DS: RuntimeDurableStorage>(
    durable_storage: &DS,
) -> Result<u64, SbiError> {
    durable_storage
        .registry_len()
        .map(|len| len as u64)
        .map_err(map_durable_storage_error)
}

#[inline]
fn handle_tezos_durable_registry_resize_tick<DS: RuntimeDurableStorage>(
    durable_storage: &mut DS,
    new_size: u64,
) -> Result<u64, SbiError> {
    durable_storage
        .registry_resize_tick(checked_usize(new_size)?)
        .map_err(map_durable_storage_error)?;
    Ok(0)
}

#[inline]
fn handle_tezos_durable_registry_copy_database<DS: RuntimeDurableStorage>(
    durable_storage: &mut DS,
    src_index: u64,
    dst_index: u64,
) -> Result<u64, SbiError> {
    durable_storage
        .registry_copy_database(checked_usize(src_index)?, checked_usize(dst_index)?)
        .map_err(map_durable_storage_error)?;
    Ok(0)
}

#[inline]
fn handle_tezos_durable_registry_move_database<DS: RuntimeDurableStorage>(
    durable_storage: &mut DS,
    src_index: u64,
    dst_index: u64,
) -> Result<u64, SbiError> {
    durable_storage
        .registry_move_database(checked_usize(src_index)?, checked_usize(dst_index)?)
        .map_err(map_durable_storage_error)?;
    Ok(0)
}

#[inline]
fn handle_tezos_durable_registry_clear_database<DS: RuntimeDurableStorage>(
    durable_storage: &mut DS,
    index: u64,
) -> Result<u64, SbiError> {
    durable_storage
        .registry_clear_database(checked_usize(index)?)
        .map_err(map_durable_storage_error)?;
    Ok(0)
}

#[inline]
fn handle_tezos_durable_database_exists<MC, M, DS>(
    machine: &mut MachineCoreState<MC, M>,
    durable_storage: &DS,
) -> Result<u64, SbiError>
where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode,
    DS: RuntimeDurableStorage,
{
    let database_index = checked_usize(machine.hart.xregisters.read(a0))?;
    let key_addr = machine.hart.xregisters.read(a1);
    let key_len = checked_usize(machine.hart.xregisters.read(a2))?;
    let key = read_guest_bytes(machine, key_addr, key_len, MAX_PVM_MEMORY_ACCESS)?;

    durable_storage
        .database_exists(database_index, &key)
        .map(|exists| exists as u64)
        .map_err(map_durable_storage_error)
}

#[inline]
fn handle_tezos_durable_database_delete<MC, M, DS>(
    machine: &mut MachineCoreState<MC, M>,
    durable_storage: &mut DS,
) -> Result<u64, SbiError>
where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode,
    DS: RuntimeDurableStorage,
{
    let database_index = checked_usize(machine.hart.xregisters.read(a0))?;
    let key_addr = machine.hart.xregisters.read(a1);
    let key_len = checked_usize(machine.hart.xregisters.read(a2))?;
    let key = read_guest_bytes(machine, key_addr, key_len, MAX_PVM_MEMORY_ACCESS)?;

    durable_storage
        .database_delete(database_index, &key)
        .map(|deleted| deleted as u64)
        .map_err(map_durable_storage_error)
}

#[inline]
fn handle_tezos_durable_database_value_length<MC, M, DS>(
    machine: &mut MachineCoreState<MC, M>,
    durable_storage: &DS,
) -> Result<u64, SbiError>
where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode,
    DS: RuntimeDurableStorage,
{
    let database_index = checked_usize(machine.hart.xregisters.read(a0))?;
    let key_addr = machine.hart.xregisters.read(a1);
    let key_len = checked_usize(machine.hart.xregisters.read(a2))?;
    let key = read_guest_bytes(machine, key_addr, key_len, MAX_PVM_MEMORY_ACCESS)?;

    durable_storage
        .database_value_length(database_index, &key)
        .map(|len| len as u64)
        .map_err(map_durable_storage_error)
}

#[inline]
fn handle_tezos_durable_database_read<MC, M, DS>(
    machine: &mut MachineCoreState<MC, M>,
    durable_storage: &DS,
) -> Result<u64, SbiError>
where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode,
    DS: RuntimeDurableStorage,
{
    let database_index = checked_usize(machine.hart.xregisters.read(a0))?;
    let key_addr = machine.hart.xregisters.read(a1);
    let key_len = checked_usize(machine.hart.xregisters.read(a2))?;
    let offset = checked_usize(machine.hart.xregisters.read(a3))?;
    let out_addr = machine.hart.xregisters.read(a4);
    let out_len = checked_usize(machine.hart.xregisters.read(a5))?;

    if out_len > MAX_DURABLE_VALUE_IO {
        return Err(SbiError::InvalidParam);
    }

    let key = read_guest_bytes(machine, key_addr, key_len, MAX_PVM_MEMORY_ACCESS)?;
    let value = durable_storage
        .database_read(database_index, &key, offset, out_len)
        .map_err(map_durable_storage_error)?;
    machine.main_memory.write_all(out_addr, &value)?;

    Ok(value.len() as u64)
}

#[inline]
fn handle_tezos_durable_database_set<MC, M, DS>(
    machine: &mut MachineCoreState<MC, M>,
    durable_storage: &mut DS,
) -> Result<u64, SbiError>
where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode,
    DS: RuntimeDurableStorage,
{
    let database_index = checked_usize(machine.hart.xregisters.read(a0))?;
    let key_addr = machine.hart.xregisters.read(a1);
    let key_len = checked_usize(machine.hart.xregisters.read(a2))?;
    let data_addr = machine.hart.xregisters.read(a3);
    let data_len = checked_usize(machine.hart.xregisters.read(a4))?;

    if data_len > MAX_DURABLE_VALUE_IO {
        return Err(SbiError::InvalidParam);
    }

    let key = read_guest_bytes(machine, key_addr, key_len, MAX_PVM_MEMORY_ACCESS)?;
    let data = read_guest_bytes(machine, data_addr, data_len, MAX_DURABLE_VALUE_IO)?;
    durable_storage
        .database_set(database_index, &key, &data)
        .map_err(map_durable_storage_error)?;

    Ok(data_len as u64)
}

#[inline]
fn handle_tezos_durable_database_write<MC, M, DS>(
    machine: &mut MachineCoreState<MC, M>,
    durable_storage: &mut DS,
) -> Result<u64, SbiError>
where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode,
    DS: RuntimeDurableStorage,
{
    let database_index = checked_usize(machine.hart.xregisters.read(a0))?;
    let key_addr = machine.hart.xregisters.read(a1);
    let key_len = checked_usize(machine.hart.xregisters.read(a2))?;
    let offset = checked_usize(machine.hart.xregisters.read(a3))?;
    let data_addr = machine.hart.xregisters.read(a4);
    let data_len = checked_usize(machine.hart.xregisters.read(a5))?;

    if data_len > MAX_DURABLE_VALUE_IO {
        return Err(SbiError::InvalidParam);
    }

    let key = read_guest_bytes(machine, key_addr, key_len, MAX_PVM_MEMORY_ACCESS)?;
    let data = read_guest_bytes(machine, data_addr, data_len, MAX_DURABLE_VALUE_IO)?;
    durable_storage
        .database_write(database_index, &key, offset, &data)
        .map(|written| written as u64)
        .map_err(map_durable_storage_error)
}

#[inline]
fn handle_tezos_durable_database_hash<MC, M, DS>(
    machine: &mut MachineCoreState<MC, M>,
    durable_storage: &DS,
) -> Result<u64, SbiError>
where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode,
    DS: RuntimeDurableStorage,
{
    let database_index = checked_usize(machine.hart.xregisters.read(a0))?;
    let out_addr = machine.hart.xregisters.read(a1);
    let hash = durable_storage
        .database_hash(database_index)
        .map_err(map_durable_storage_error)?;
    machine.main_memory.write_all(out_addr, hash.as_ref())?;

    Ok(hash.as_ref().len() as u64)
}

/// Handle a [SBI_TEZOS_REVEAL] call.
#[inline]
fn handle_tezos_reveal<MC, M>(
    machine: &mut MachineCoreState<MC, M>,
    reveal_request: &mut RevealRequest<M>,
    status: &mut Atom<PvmStatus, M>,
) where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode,
{
    let request_address = machine.hart.xregisters.read(a0);
    let request_size = machine.hart.xregisters.read(a1) as usize;
    let request_size = request_size.min(REVEAL_REQUEST_MAX_SIZE);

    let target_buffer = &mut reveal_request.bytes[..request_size];
    if machine
        .main_memory
        .read_all(request_address, target_buffer)
        .is_err()
    {
        return sbi_return_error(&mut machine.hart.xregisters, SbiError::InvalidAddress);
    }

    reveal_request.size.write(request_size as u64);
    status.write(PvmStatus::WaitingForReveal);
}

/// Handle unsupported SBI calls.
#[inline(always)]
fn handle_not_supported<M>(xregisters: &mut XRegisters<M>)
where
    M: AtomMode,
{
    // SBI requires us to indicate that we don't support this function by returning
    // `ERR_NOT_SUPPORTED`.
    sbi_return_error(xregisters, SbiError::NotSupported);
}

/// Handle a [SBI_TEZOS_KECCAK256_ENQUEUE] call.
#[inline]
fn handle_tezos_keccak256_enqueue<MC, M>(
    machine: &mut MachineCoreState<MC, M>,
    tezos: &mut Tezos<M>,
) -> Result<u64, SbiError>
where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode + CryptoMode<KeccakAlgorithm>,
{
    let msg_addr = machine.hart.xregisters.read(a0);
    let msg_len = machine.hart.xregisters.read(a1) as usize;
    let bytes = read_guest_bytes(machine, msg_addr, msg_len, MAX_PVM_MEMORY_ACCESS)?;
    tezos.keccak_job.push(KeccakRequest::new(&bytes));
    Ok(0)
}

/// Handle a [SBI_TEZOS_KECCAK256_DEQUEUE] call.
///
/// Writes the 32-byte result to the output pointer.  Returns `SbiError::Failed`
/// if the queue is empty.
#[inline]
fn handle_tezos_keccak256_dequeue<MC, M>(
    machine: &mut MachineCoreState<MC, M>,
    tezos: &mut Tezos<M>,
) -> Result<u64, SbiError>
where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode + CryptoMode<KeccakAlgorithm>,
{
    let out_addr = machine.hart.xregisters.read(a0);
    let hash = tezos.keccak_job.pop().ok_or(SbiError::Failed)?;
    machine.main_memory.write(out_addr, hash)?;
    Ok(hash.len() as u64)
}

/// Handle a [SBI_TEZOS_SECP256K1_ENQUEUE] call.
///
/// Reads `(public_key_ptr, signature_ptr, message_hash_ptr)` from registers and
/// stores the request in the provable secp256k1 queue, dispatching to the
/// background worker in Normal mode.
#[inline]
fn handle_tezos_secp256k1_enqueue<MC, M>(
    machine: &mut MachineCoreState<MC, M>,
    tezos: &mut Tezos<M>,
) -> Result<u64, SbiError>
where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode + CryptoMode<Secp256k1Algorithm>,
{
    let pk_addr = machine.hart.xregisters.read(a0);
    let sig_addr = machine.hart.xregisters.read(a1);
    let hash_addr = machine.hart.xregisters.read(a2);

    let request = Secp256k1Request {
        public_key: machine.main_memory.read(pk_addr)?,
        signature: machine.main_memory.read(sig_addr)?,
        message_hash: machine.main_memory.read(hash_addr)?,
    };
    tezos.secp256k1_job.push(request);
    Ok(0)
}

/// Handle a [SBI_TEZOS_SECP256K1_DEQUEUE] call.
///
/// Returns `1` if the signature was valid, `0` if invalid.
/// Returns `SbiError::Failed` if the queue is empty.
#[inline]
fn handle_tezos_secp256k1_dequeue<MC, M>(
    _machine: &mut MachineCoreState<MC, M>,
    tezos: &mut Tezos<M>,
) -> Result<u64, SbiError>
where
    MC: MemoryConfig,
    M: AtomMode + DataSpaceMode + VectorMode + CryptoMode<Secp256k1Algorithm>,
{
    let valid = tezos.secp256k1_job.pop().ok_or(SbiError::Failed)?;
    Ok(valid as u64)
}

/// Handle a Tezos SBI call.
pub(super) fn handle_tezos<MC, M, DS>(
    machine: &mut MachineCoreState<MC, M>,
    durable_storage: &mut DS,
    tezos: &mut Tezos<M>,
) where
    MC: MemoryConfig,
    DS: RuntimeDurableStorage,
    M: AtomMode + DataSpaceMode + VectorMode + PvmCryptoMode,
{
    // TODO: RV-777: remove below and instead have each system call return a `ProgramCounterUpdate`
    let pc = machine.hart.pc.read().wrapping_add(4);
    machine.hart.pc.write(pc);

    let sbi_function = machine.hart.xregisters.read(a6);
    match sbi_function {
        SBI_TEZOS_INBOX_NEXT => handle_tezos_inbox_next(&mut tezos.status),
        SBI_TEZOS_WRITE_OUTPUT => {
            handle_tezos_write_output(machine, &mut tezos.outbox, &tezos.level)
        }
        SBI_TEZOS_ED25519_SIGN => sbi_wrap(machine, handle_tezos_ed25519_sign),
        SBI_TEZOS_ED25519_VERIFY => sbi_wrap(machine, handle_tezos_ed25519_verify),
        SBI_TEZOS_BLAKE2B_HASH256 => sbi_wrap(machine, handle_tezos_blake2b_hash256),
        SBI_TEZOS_SECP256K1_VERIFY => sbi_wrap(machine, handle_tezos_secp256k1_verify),
        SBI_TEZOS_KECCAK256_HASH => sbi_wrap(machine, handle_tezos_keccak256_hash),
        SBI_TEZOS_KECCAK256_ENQUEUE => {
            sbi_wrap(machine, |machine| handle_tezos_keccak256_enqueue(machine, tezos))
        }
        SBI_TEZOS_KECCAK256_DEQUEUE => {
            sbi_wrap(machine, |machine| handle_tezos_keccak256_dequeue(machine, tezos))
        }
        SBI_TEZOS_SECP256K1_ENQUEUE => {
            sbi_wrap(machine, |machine| handle_tezos_secp256k1_enqueue(machine, tezos))
        }
        SBI_TEZOS_SECP256K1_DEQUEUE => {
            sbi_wrap(machine, |machine| handle_tezos_secp256k1_dequeue(machine, tezos))
        }
        SBI_TEZOS_REVEAL => {
            handle_tezos_reveal(machine, &mut tezos.reveal_request, &mut tezos.status)
        }
        SBI_TEZOS_DURABLE_REGISTRY_LEN => sbi_wrap(machine, |_| {
            handle_tezos_durable_registry_len(durable_storage)
        }),
        SBI_TEZOS_DURABLE_REGISTRY_RESIZE_TICK => sbi_wrap(machine, |machine| {
            handle_tezos_durable_registry_resize_tick(
                durable_storage,
                machine.hart.xregisters.read(a0),
            )
        }),
        SBI_TEZOS_DURABLE_REGISTRY_COPY_DATABASE => sbi_wrap(machine, |machine| {
            handle_tezos_durable_registry_copy_database(
                durable_storage,
                machine.hart.xregisters.read(a0),
                machine.hart.xregisters.read(a1),
            )
        }),
        SBI_TEZOS_DURABLE_REGISTRY_MOVE_DATABASE => sbi_wrap(machine, |machine| {
            handle_tezos_durable_registry_move_database(
                durable_storage,
                machine.hart.xregisters.read(a0),
                machine.hart.xregisters.read(a1),
            )
        }),
        SBI_TEZOS_DURABLE_REGISTRY_CLEAR_DATABASE => sbi_wrap(machine, |machine| {
            handle_tezos_durable_registry_clear_database(
                durable_storage,
                machine.hart.xregisters.read(a0),
            )
        }),
        SBI_TEZOS_DURABLE_DATABASE_EXISTS => sbi_wrap(machine, |machine| {
            handle_tezos_durable_database_exists(machine, durable_storage)
        }),
        SBI_TEZOS_DURABLE_DATABASE_DELETE => sbi_wrap(machine, |machine| {
            handle_tezos_durable_database_delete(machine, durable_storage)
        }),
        SBI_TEZOS_DURABLE_DATABASE_VALUE_LENGTH => sbi_wrap(machine, |machine| {
            handle_tezos_durable_database_value_length(machine, durable_storage)
        }),
        SBI_TEZOS_DURABLE_DATABASE_READ => sbi_wrap(machine, |machine| {
            handle_tezos_durable_database_read(machine, durable_storage)
        }),
        SBI_TEZOS_DURABLE_DATABASE_SET => sbi_wrap(machine, |machine| {
            handle_tezos_durable_database_set(machine, durable_storage)
        }),
        SBI_TEZOS_DURABLE_DATABASE_WRITE => sbi_wrap(machine, |machine| {
            handle_tezos_durable_database_write(machine, durable_storage)
        }),
        SBI_TEZOS_DURABLE_DATABASE_HASH => sbi_wrap(machine, |machine| {
            handle_tezos_durable_database_hash(machine, durable_storage)
        }),
        _ => handle_not_supported(&mut machine.hart.xregisters),
    }
}
