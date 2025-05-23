(*****************************************************************************)
(*                                                                           *)
(* SPDX-License-Identifier: MIT                                              *)
(* Copyright (c) 2024-2025 TriliTech <contact@trili.tech>                    *)
(* Copyright (c) 2024-2025 Nomadic Labs <contact@nomadic-labs.com>           *)
(*                                                                           *)
(*****************************************************************************)

module ProofImpl : Common.ProofSigs with type proof = Octez_riscv_api.proof =
struct
  module Api = Octez_riscv_api

  type proof = Api.proof

  let api_verify_proof = Api.octez_riscv_verify_proof

  let api_proof_start_state = Api.octez_riscv_proof_start_state

  let api_proof_stop_state = Api.octez_riscv_proof_stop_state

  let api_produce_proof = Api.octez_riscv_produce_proof

  let api_serialise_proof = Api.octez_riscv_serialise_proof

  let api_deserialise_proof = Api.octez_riscv_deserialise_proof
end

module ProtoImpl = Make_backend.MakeBackend (ProofImpl)

type reveals = ProtoImpl.reveals

type write_debug = ProtoImpl.write_debug

type hash = ProtoImpl.hash

type state = ProtoImpl.state

type status = ProtoImpl.status

type input = ProtoImpl.input =
  | Inbox_message of int32 * int64 * string
  | Reveal of string

type input_request = ProtoImpl.input_request =
  | No_input_required
  | Initial
  | First_after of int32 * int64
  | Needs_reveal of string

type proof = ProtoImpl.proof

type output_proof = ProtoImpl.output_proof

type output_info = ProtoImpl.output_info = {
  message_index : Z.t;
  outbox_level : Bounded.Non_negative_int32.t;
}

type output = ProtoImpl.output = {info : output_info; encoded_message : string}

module Mutable_state = ProtoImpl.Mutable_state

let compute_step_many = ProtoImpl.compute_step_many

let compute_step = ProtoImpl.compute_step

let compute_step_with_debug = ProtoImpl.compute_step_with_debug

let get_tick = ProtoImpl.get_tick

let get_status = ProtoImpl.get_status

let get_message_counter = ProtoImpl.get_message_counter

let string_of_status = ProtoImpl.string_of_status

let install_boot_sector = ProtoImpl.install_boot_sector

let get_current_level = ProtoImpl.get_current_level

let state_hash = ProtoImpl.state_hash

let set_input = ProtoImpl.set_input

let proof_start_state = ProtoImpl.proof_start_state

let proof_stop_state = ProtoImpl.proof_stop_state

let verify_proof = ProtoImpl.verify_proof

let produce_proof = ProtoImpl.produce_proof

let serialise_proof = ProtoImpl.serialise_proof

let deserialise_proof = ProtoImpl.deserialise_proof

let output_info_of_output_proof = ProtoImpl.output_info_of_output_proof

let state_of_output_proof = ProtoImpl.state_of_output_proof

let verify_output_proof = ProtoImpl.verify_output_proof

let serialise_output_proof = ProtoImpl.serialise_output_proof

let deserialise_output_proof = ProtoImpl.deserialise_output_proof

let get_reveal_request = ProtoImpl.get_reveal_request

let insert_failure = ProtoImpl.insert_failure
