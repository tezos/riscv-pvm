(*****************************************************************************)
(*                                                                           *)
(* SPDX-License-Identifier: MIT                                              *)
(* Copyright (c) 2024-2025 TriliTech <contact@trili.tech>                    *)
(* Copyright (c) 2024-2025 Nomadic Labs <contact@nomadic-labs.com>           *)
(*                                                                           *)
(*****************************************************************************)

module ProofImpl : Common.ProofSigs with type proof = int32 = struct
  type proof = int32

  let api_verify_proof _proof = assert false

  let api_proof_start_state _proof = assert false

  let api_proof_stop_state _proof = assert false

  let api_produce_proof _input _proof = assert false

  let api_serialise_proof _proof = assert false

  let api_deserialise_proof _bytes = assert false
end

module RNImpl = Make_backend.MakeBackend (ProofImpl)

type reveals = RNImpl.reveals

type write_debug = RNImpl.write_debug

type hash = RNImpl.hash

type state = RNImpl.state

type status = RNImpl.status

type input = RNImpl.input

type input_request = RNImpl.input_request

type proof = RNImpl.proof

type output_proof = Octez_riscv_api.output_proof

type output_info = Common.output_info

type output = Common.output

module Mutable_state = RNImpl.Mutable_state

let compute_step_many = RNImpl.compute_step_many

let compute_step = RNImpl.compute_step

let compute_step_with_debug = RNImpl.compute_step_with_debug

let get_tick = RNImpl.get_tick

let get_status = RNImpl.get_status

let get_message_counter = RNImpl.get_message_counter

let string_of_status = RNImpl.string_of_status

let install_boot_sector = RNImpl.install_boot_sector

let get_current_level = RNImpl.get_current_level

let state_hash = RNImpl.state_hash

let set_input = RNImpl.set_input

let proof_start_state = RNImpl.proof_start_state

let proof_stop_state = RNImpl.proof_stop_state

let verify_proof = RNImpl.verify_proof

let produce_proof = RNImpl.produce_proof

let serialise_proof = RNImpl.serialise_proof

let deserialise_proof = RNImpl.deserialise_proof

let output_info_of_output_proof = RNImpl.output_info_of_output_proof

let state_of_output_proof = RNImpl.state_of_output_proof

let verify_output_proof = RNImpl.verify_output_proof

let serialise_output_proof = RNImpl.serialise_output_proof

let deserialise_output_proof = RNImpl.deserialise_output_proof

let get_reveal_request = RNImpl.get_reveal_request

let insert_failure = RNImpl.insert_failure
