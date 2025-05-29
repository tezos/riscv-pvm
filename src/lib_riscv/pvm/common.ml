type reveals

type hash = Tezos_crypto.Hashed.Smart_rollup_state_hash.t

type write_debug = string -> unit Lwt.t

type state = Octez_riscv_api.state

type status = Octez_riscv_api.status

type output_proof = Octez_riscv_api.output_proof

type mut_state = Octez_riscv_api.mut_state

type input_request =
  | No_input_required
  | Initial
  | First_after of int32 * int64
  | Needs_reveal of string

type input = Inbox_message of int32 * int64 * string | Reveal of string

type output_info = {
  message_index : Z.t;
  outbox_level : Bounded.Non_negative_int32.t;
}

type output = {info : output_info; encoded_message : string}

module type ProofSigs = sig
  type proof

  val api_verify_proof :
    Octez_riscv_api.input option ->
    proof ->
    Octez_riscv_api.input_request option

  val api_proof_start_state : proof -> bytes

  val api_proof_stop_state : proof -> bytes

  val api_produce_proof :
    Octez_riscv_api.input option -> Octez_riscv_api.state -> proof option

  val api_serialise_proof : proof -> bytes

  val api_deserialise_proof : bytes -> (proof, string) result
end

module type Mutable_state = sig
  type t = mut_state

  val from_imm : state -> t

  val to_imm : t -> state

  val compute_step_many :
    ?reveal_builtins:reveals ->
    ?write_debug:write_debug ->
    ?stop_at_snapshot:bool ->
    max_steps:int64 ->
    t ->
    int64 Lwt.t

  val get_tick : t -> Z.t Lwt.t

  val get_status : t -> status Lwt.t

  val get_message_counter : t -> int64 Lwt.t

  val get_current_level : t -> int32 option Lwt.t

  val state_hash : t -> hash

  val set_input : t -> input -> unit Lwt.t

  val get_reveal_request : t -> string Lwt.t

  val insert_failure : t -> unit Lwt.t
end
