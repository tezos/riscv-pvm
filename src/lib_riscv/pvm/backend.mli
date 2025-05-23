(*****************************************************************************)
(*                                                                           *)
(* SPDX-License-Identifier: MIT                                              *)
(* Copyright (c) 2024-2025 TriliTech <contact@trili.tech>                    *)
(* Copyright (c) 2024 Nomadic Labs <contact@nomadic-labs.com>                *)
(*                                                                           *)
(*****************************************************************************)

type reveals

type write_debug = Common.write_debug

type hash = Common.hash

type state = Common.state

type status = Common.status

(* TODO RV-615: Improve the `input` type exposed in protocol environment *)
(* Mirrors Api.input but requires manual conversion *)
type input = Common.input =
  | Inbox_message of int32 * int64 * string
  | Reveal of string

(* Mirrors Api.input_request but requires manual conversion *)
type input_request = Common.input_request =
  | No_input_required
  | Initial
  | First_after of int32 * int64
  | Needs_reveal of string

type proof = Octez_riscv_api.proof

type output_proof = Common.output_proof

type output_info = Common.output_info = {
  message_index : Z.t;
  outbox_level : Bounded.Non_negative_int32.t;
}

type output = Common.output = {info : output_info; encoded_message : string}

module Mutable_state : Common.Mutable_state

val compute_step_many :
  ?reveal_builtins:reveals ->
  ?write_debug:write_debug ->
  ?stop_at_snapshot:bool ->
  max_steps:int64 ->
  state ->
  (state * int64) Lwt.t

val compute_step : state -> state Lwt.t

val compute_step_with_debug : ?write_debug:write_debug -> state -> state Lwt.t

val get_tick : state -> Z.t Lwt.t

val get_status : state -> status Lwt.t

val get_message_counter : state -> int64 Lwt.t

val string_of_status : status -> string

val install_boot_sector : state -> string -> state Lwt.t

val get_current_level : state -> int32 option Lwt.t

val state_hash : state -> hash

val set_input : state -> input -> state Lwt.t

val proof_start_state : proof -> hash

val proof_stop_state : proof -> hash

val verify_proof : input option -> proof -> input_request option

val produce_proof : input option -> state -> proof option

val serialise_proof : proof -> bytes

val deserialise_proof : bytes -> (proof, string) result

val output_info_of_output_proof : output_proof -> output_info

val state_of_output_proof : output_proof -> hash

val verify_output_proof : output_proof -> output option

val serialise_output_proof : output_proof -> bytes

val deserialise_output_proof : bytes -> (output_proof, string) result

val get_reveal_request : state -> string Lwt.t

val insert_failure : state -> state Lwt.t
