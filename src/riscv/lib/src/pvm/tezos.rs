// SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Tezos-specific host functions for the PVM

use std::cmp::min;

use ed25519_dalek::Signature;
use ed25519_dalek::Signer;
use ed25519_dalek::SigningKey;
use ed25519_dalek::VerifyingKey;
use tezos_smart_rollup_constants::core::MAX_INPUT_MESSAGE_SIZE;
use tezos_smart_rollup_constants::riscv::REVEAL_DATA_MAX_SIZE;
use tezos_smart_rollup_constants::riscv::REVEAL_REQUEST_MAX_SIZE;
use tezos_smart_rollup_constants::riscv::SBI_TEZOS_BLAKE2B_HASH256;
use tezos_smart_rollup_constants::riscv::SBI_TEZOS_ED25519_SIGN;
use tezos_smart_rollup_constants::riscv::SBI_TEZOS_ED25519_VERIFY;
use tezos_smart_rollup_constants::riscv::SBI_TEZOS_INBOX_NEXT;
use tezos_smart_rollup_constants::riscv::SBI_TEZOS_REVEAL;
use tezos_smart_rollup_constants::riscv::SbiError;

use super::PvmStatus;
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
use crate::machine_state::registers::a6;
use crate::state_backend::Cell;
use crate::state_backend::ManagerReadWrite;
use crate::state_backend::ManagerWrite;

/// Write the SBI error code as the return value.
#[inline]
fn sbi_return_error<M: ManagerWrite>(xregisters: &mut XRegisters<M>, code: SbiError) {
    xregisters.write(a0, code as i64 as u64);
}

/// Write an arbitrary value as single return value.
#[inline]
fn sbi_return1<M: ManagerWrite>(xregisters: &mut XRegisters<M>, value: XValue) {
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
    M: ManagerWrite,
    F: FnOnce(&mut MachineCoreState<MC, M>) -> Result<XValue, SbiError>,
{
    match inner(machine) {
        Ok(value) => sbi_return1(&mut machine.hart.xregisters, value),
        Err(error) => sbi_return_error(&mut machine.hart.xregisters, error),
    }
}

/// Provide input information to the machine. Returns `false` in case the
/// machine wasn't expecting any input, otherwise returns `true`.
pub fn provide_input<MC, M>(
    status: &mut Cell<PvmStatus, M>,
    machine: &mut MachineCoreState<MC, M>,
    level: u32,
    counter: u32,
    payload: &[u8],
) -> bool
where
    MC: MemoryConfig,
    M: ManagerReadWrite,
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
    status: &mut Cell<PvmStatus, M>,
    machine: &mut MachineCoreState<MC, M>,
    reveal_data: &[u8],
) -> bool
where
    MC: MemoryConfig,
    M: ManagerReadWrite,
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
fn handle_tezos_inbox_next<M>(status: &mut Cell<PvmStatus, M>)
where
    M: ManagerWrite,
{
    // Prepare the EE state for an input tick.
    status.write(PvmStatus::WaitingForInput);
}

/// Produce a Ed25519 signature.
#[inline]
fn handle_tezos_ed25519_sign<MC, M>(machine: &mut MachineCoreState<MC, M>) -> Result<u64, SbiError>
where
    MC: MemoryConfig,
    M: ManagerReadWrite,
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
    M: ManagerReadWrite,
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
    M: ManagerReadWrite,
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

/// Handle a [SBI_TEZOS_REVEAL] call.
#[inline]
fn handle_tezos_reveal<MC, M>(
    machine: &mut MachineCoreState<MC, M>,
    reveal_request: &mut RevealRequest<M>,
    status: &mut Cell<PvmStatus, M>,
) where
    MC: MemoryConfig,
    M: ManagerReadWrite,
{
    let request_address = machine.hart.xregisters.read(a0);
    let request_size = machine.hart.xregisters.read(a1);

    let mut buffer = vec![0u8; min(request_size as usize, REVEAL_REQUEST_MAX_SIZE)];

    if machine
        .main_memory
        .read_all(request_address, &mut buffer)
        .is_err()
    {
        return sbi_return_error(&mut machine.hart.xregisters, SbiError::InvalidAddress);
    }

    // TODO: RV-425 Cross-page memory accesses are not translated correctly
    reveal_request.bytes.write_all(0, &buffer);
    reveal_request.size.write(request_size);
    status.write(PvmStatus::WaitingForReveal);
}

/// Handle unsupported SBI calls.
#[inline(always)]
fn handle_not_supported<M>(xregisters: &mut XRegisters<M>)
where
    M: ManagerWrite,
{
    // SBI requires us to indicate that we don't support this function by returning
    // `ERR_NOT_SUPPORTED`.
    sbi_return_error(xregisters, SbiError::NotSupported);
}

/// Handle a Tezos SBI call.
pub(super) fn handle_tezos<MC, M>(
    machine: &mut MachineCoreState<MC, M>,
    status: &mut Cell<PvmStatus, M>,
    reveal_request: &mut RevealRequest<M>,
) where
    MC: MemoryConfig,
    M: ManagerReadWrite,
{
    let sbi_function = machine.hart.xregisters.read(a6);
    match sbi_function {
        SBI_TEZOS_INBOX_NEXT => handle_tezos_inbox_next(status),
        SBI_TEZOS_ED25519_SIGN => sbi_wrap(machine, handle_tezos_ed25519_sign),
        SBI_TEZOS_ED25519_VERIFY => sbi_wrap(machine, handle_tezos_ed25519_verify),
        SBI_TEZOS_BLAKE2B_HASH256 => sbi_wrap(machine, handle_tezos_blake2b_hash256),
        SBI_TEZOS_REVEAL => handle_tezos_reveal(machine, reveal_request, status),
        _ => handle_not_supported(&mut machine.hart.xregisters),
    }
}
