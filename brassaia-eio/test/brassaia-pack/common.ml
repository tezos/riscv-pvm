(*
 * Copyright (c) 2018-2022 Tarides <contact@tarides.com>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *)

module Brassaia = Brassaia_eio.Brassaia
module Brassaia_pack = Brassaia_eio_pack.Brassaia_pack
module Brassaia_pack_unix = Brassaia_eio_pack_io.Brassaia_pack_unix
open Brassaia.Export_for_backends
module Int63 = Optint.Int63

let get = function Some x -> x | None -> Alcotest.fail "None"

let sha1 x = Brassaia.Hash.SHA1.hash (fun f -> f x)

let sha1_contents x = sha1 ("B" ^ x)

let rm_dir root =
  if Sys.file_exists root then (
    let cmd = Printf.sprintf "rm -rf %s" root in
    [%logs.info "exec: %s\n%!" cmd] ;
    let _ = Sys.command cmd in
    ())

let index_log_size = Some 1_000

let () = Random.self_init ()

let random_char () = char_of_int (Random.int 256)

let random_string n = String.init n (fun _i -> random_char ())

let random_letter () = char_of_int (Char.code 'a' + Random.int 26)

let random_letters n = String.init n (fun _i -> random_letter ())

module Conf = Brassaia_tezos.Conf

module Schema = struct
  open Brassaia
  module Contents = Contents.String_v2
  module Branch = Branch.String
  module Hash = Hash.SHA1
  module Node = Node.Generic_key.Make_v2 (Hash)
  module Commit = Commit.Generic_key.Make_v2 (Hash)
  module Info = Info.Default
end

module Contents = struct
  include Brassaia.Contents.String

  let kind _ = Brassaia_pack.Pack_value.Kind.Contents

  type Brassaia_pack.Pack_value.kinded += Contents of t

  let to_kinded t = Contents t

  let of_kinded = function Contents c -> c | _ -> assert false

  module H = Brassaia.Hash.Typed (Brassaia.Hash.SHA1) (Brassaia.Contents.String)

  let hash = H.hash

  let magic = 'B'

  let weight _ = Brassaia_pack.Pack_value.Immediate 1

  let encode_triple = Brassaia.Type.(unstage (encode_bin (triple H.t char t)))

  let decode_triple = Brassaia.Type.(unstage (decode_bin (triple H.t char t)))

  let length_header = Fun.const (Some `Varint)

  let encode_bin ~dict:_ ~offset_of_key:_ k x = encode_triple (k, magic, x)

  let decode_bin ~dict:_ ~key_of_offset:_ ~key_of_hash:_ x pos_ref =
    let _, _, v = decode_triple x pos_ref in
    v

  let decode_bin_length =
    match Brassaia.Type.(Size.of_encoding (triple H.t char t)) with
    | Dynamic f -> f
    | _ -> assert false

  let encoding = Data_encoding.string
end

module I = Brassaia_pack_unix.Index
module Index = Brassaia_pack_unix.Index.Make (Schema.Hash)
module Key = Brassaia_pack_unix.Pack_key.Make (Schema.Hash)
module Io = Brassaia_pack_unix.Io.Unix
module Io_errors = Brassaia_pack_unix.Io_errors
module File_manager = Brassaia_pack_unix.File_manager.Make (Index)
module Dict = File_manager.Dict
module Dispatcher = Brassaia_pack_unix.Dispatcher.Make (File_manager)
module Pack =
  Brassaia_pack_unix.Pack_store.Make (File_manager) (Dispatcher) (Schema.Hash)
    (Contents)
module Branch =
  Brassaia_pack_unix.Atomic_write.Make_persistent
    (Brassaia.Branch.String)
    (Brassaia_pack.Atomic_write.Value.Of_hash (Schema.Hash))
module Lower = Brassaia_pack_unix.Lower

module Make_context (Config : sig
  val root : string
end) =
struct
  let fresh_name =
    let c = ref 0 in
    fun object_type ->
      incr c ;

      let name = Filename.concat Config.root ("pack_" ^ string_of_int !c) in
      [%logs.info "Constructing %s context object: %s" object_type name] ;
      name

  let mkdir_dash_p dirname =
    let rec aux dir =
      if Sys.file_exists dir && Sys.is_directory dir then ()
      else (
        if Sys.file_exists dir then Unix.unlink dir ;
        aux (Filename.dirname dir) ;
        Unix.mkdir dir 0o755)
    in
    aux dirname

  type d = {name : string; file_manager : File_manager.t; dict : Dict.t}

  (* TODO : test the indexing_strategy minimal. *)
  let config ~readonly ~fresh name =
    Brassaia_pack.Conf.init
      ~fresh
      ~readonly
      ~indexing_strategy:Brassaia_pack.Indexing_strategy.always
      ~lru_size:0
      name

  (* TODO : remove duplication with brassaia_pack/ext.ml *)
  let get_file_manager config =
    let readonly = Brassaia_pack.Conf.readonly config in
    if readonly then File_manager.open_ro config |> Io_errors.raise_if_error
    else
      let fresh = Brassaia_pack.Conf.fresh config in
      if fresh then (
        let root = Brassaia_pack.Conf.root config in
        mkdir_dash_p root ;
        File_manager.create_rw ~overwrite:true config
        |> Io_errors.raise_if_error)
      else File_manager.open_rw config |> Io_errors.raise_if_error

  let get_dict ?name ~readonly ~fresh () =
    let name = Option.value name ~default:(fresh_name "dict") in
    let file_manager = config ~readonly ~fresh name |> get_file_manager in
    let dict = File_manager.dict file_manager in
    {name; dict; file_manager}

  let close_dict d =
    File_manager.close d.file_manager |> Io_errors.raise_if_error

  type t = {
    name : string;
    file_manager : File_manager.t;
    index : Index.t;
    pack : read Pack.t;
    dict : Pack.dict;
  }

  let create ~readonly ~fresh name =
    let f = ref (fun () -> ()) in
    let config = config ~readonly ~fresh name in
    let file_manager = get_file_manager config in
    let dispatcher = Dispatcher.init file_manager |> Io_errors.raise_if_error in
    (* open the index created by the file_manager. *)
    let index = File_manager.index file_manager in
    let dict = File_manager.dict file_manager in
    let lru = Some (Brassaia_pack_unix.Lru.create config) in
    let pack = Pack.init ~config ~file_manager ~dict ~dispatcher ~lru in
    (f := fun () -> File_manager.flush file_manager |> Io_errors.raise_if_error) ;
    {name; index; pack; dict; file_manager}

  let get_rw_pack () =
    let name = fresh_name "" in
    create ~readonly:false ~fresh:true name

  let get_ro_pack name = create ~readonly:true ~fresh:false name

  let reopen_rw name = create ~readonly:false ~fresh:false name

  let close_pack t =
    let _ = File_manager.flush t.file_manager in
    File_manager.close t.file_manager |> Io_errors.raise_if_error
end

module Alcotest = struct
  include Helpers.Alcotest

  let int63 = testable Int63.pp Int63.equal

  let check_raises_pack_error msg pass f =
    match f () with
    | _ ->
        Alcotest.failf
          "Fail %s: expected function to raise, but it returned instead."
          msg
    | exception exn -> (
        match exn with
        | Brassaia_pack_unix.Errors.Pack_error e -> (
            match pass e with
            | true -> ()
            | false ->
                Alcotest.failf
                  "Fail %s: function raised unexpected exception %s"
                  msg
                  (Printexc.to_string exn))
        | _ ->
            Alcotest.failf
              "Fail %s: expected function to raise Pack_error, but it raised \
               %s instead"
              msg
              (Printexc.to_string exn))

  (** TODO: upstream this to Alcotest *)
  let check_raises msg exn (type a) (f : unit -> a) =
    try
      let (_ : a) = f () in
      Alcotest.failf
        "Fail %s: expected function to raise %s, but it returned instead."
        msg
        (Printexc.to_string exn)
    with
    | e when e = exn -> ()
    | e ->
        Alcotest.failf
          "Fail %s: expected function to raise %s, but it raised %s instead."
          msg
          (Printexc.to_string exn)
          (Printexc.to_string e)

  let testable_repr t =
    Alcotest.testable (Brassaia.Type.pp t) Brassaia.Type.(unstage (equal t))

  let check_repr t = Alcotest.check (testable_repr t)

  let kind = testable_repr Brassaia_pack.Pack_value.Kind.t

  let hash = testable_repr Schema.Hash.t

  let quick_tc name f = test_case_eio name `Quick f
end

module Filename = struct
  include Filename

  (** Extraction from OCaml for pre-4.10 compatibility *)

  let generic_quote quotequote s =
    let l = String.length s in
    let b = Buffer.create (l + 20) in
    Buffer.add_char b '\'' ;
    for i = 0 to l - 1 do
      if s.[i] = '\'' then Buffer.add_string b quotequote
      else Buffer.add_char b s.[i]
    done ;
    Buffer.add_char b '\'' ;
    Buffer.contents b

  let quote_command =
    match Sys.os_type with
    | "Win32" ->
        let quote s =
          let l = String.length s in
          let b = Buffer.create (l + 20) in
          Buffer.add_char b '\"' ;
          let rec loop i =
            if i = l then Buffer.add_char b '\"'
            else
              match s.[i] with
              | '\"' -> loop_bs 0 i
              | '\\' -> loop_bs 0 i
              | c ->
                  Buffer.add_char b c ;
                  loop (i + 1)
          and loop_bs n i =
            if i = l then (
              Buffer.add_char b '\"' ;
              add_bs n)
            else
              match s.[i] with
              | '\"' ->
                  add_bs ((2 * n) + 1) ;
                  Buffer.add_char b '\"' ;
                  loop (i + 1)
              | '\\' -> loop_bs (n + 1) (i + 1)
              | _ ->
                  add_bs n ;
                  loop i
          and add_bs n =
            for _j = 1 to n do
              Buffer.add_char b '\\'
            done
          in
          loop 0 ;
          Buffer.contents b
        in
        let quote_cmd s =
          let b = Buffer.create (String.length s + 20) in
          String.iter
            (fun c ->
              match c with
              | '(' | ')' | '!' | '^' | '%' | '\"' | '<' | '>' | '&' | '|' ->
                  Buffer.add_char b '^' ;
                  Buffer.add_char b c
              | _ -> Buffer.add_char b c)
            s ;
          Buffer.contents b
        in
        let quote_cmd_filename f =
          if String.contains f '\"' || String.contains f '%' then
            failwith ("Filename.quote_command: bad file name " ^ f)
          else if String.contains f ' ' then "\"" ^ f ^ "\""
          else f
        in
        fun cmd ?stdin ?stdout ?stderr args ->
          String.concat
            ""
            [
              "\"";
              quote_cmd_filename cmd;
              " ";
              quote_cmd (String.concat " " (List.map quote args));
              (match stdin with
              | None -> ""
              | Some f -> " <" ^ quote_cmd_filename f);
              (match stdout with
              | None -> ""
              | Some f -> " >" ^ quote_cmd_filename f);
              (match stderr with
              | None -> ""
              | Some f ->
                  if stderr = stdout then " 2>&1"
                  else " 2>" ^ quote_cmd_filename f);
              "\"";
            ]
    | _ -> (
        let quote = generic_quote "'\\''" in
        fun cmd ?stdin ?stdout ?stderr args ->
          String.concat " " (List.map quote (cmd :: args))
          ^ (match stdin with None -> "" | Some f -> " <" ^ quote f)
          ^ (match stdout with None -> "" | Some f -> " >" ^ quote f)
          ^
          match stderr with
          | None -> ""
          | Some f -> if stderr = stdout then " 2>&1" else " 2>" ^ quote f)
end

(** Exec a command, and return [Ok ()] or [Error n] if return code is n <> 0 *)
let exec_cmd cmd =
  [%logs.info "exec: %s" cmd] ;
  match Sys.command cmd with 0 -> Ok () | n -> Error n

let rec repeat = function
  | 0 -> fun _f x -> x
  | n -> fun f x -> f (repeat (n - 1) f x)

(** The current working directory depends on whether the test binary is directly
    run or is triggered with [dune exec], [dune runtest]. We normalise by
    switching to the project root first. *)
let goto_project_root () =
  let cwd = Fpath.v (Sys.getcwd ()) in
  match cwd |> Fpath.segs |> List.rev with
  | "brassaia-pack" :: "test" :: "default" :: "_build" :: _ ->
      let root = cwd |> repeat 4 Fpath.parent in
      Unix.chdir (Fpath.to_string root)
  | _ -> ()

let rec unlink_path path =
  match Brassaia_pack_unix.Io.Unix.classify_path path with
  | `No_such_file_or_directory -> ()
  | `Directory ->
      Sys.readdir path
      |> Array.map (fun p -> Filename.concat path p)
      |> Array.iter unlink_path ;
      Unix.rmdir path
  | _ -> Unix.unlink path

let create_lower_root =
  let counter = ref 0 in
  let ( let$ ) res f = f @@ Result.get_ok res in
  fun ?(mkdir = true) () ->
    let lower_root = "brassia_eio_test_lower_" ^ string_of_int !counter in
    incr counter ;
    let lower_path = Filename.concat "_build" lower_root in
    unlink_path lower_path ;
    let$ _ =
      if mkdir then Brassaia_pack_unix.Io.Unix.mkdir lower_path
      else Result.ok ()
    in
    lower_path

let setup_test_env ~root_archive ~root_local_build =
  goto_project_root () ;
  rm_dir root_local_build ;
  let cmd =
    Filename.quote_command "cp" ["-R"; "-p"; root_archive; root_local_build]
  in
  exec_cmd cmd |> function
  | Ok () -> ()
  | Error n ->
      Fmt.failwith
        "Failed to set up the test environment: command `%s' exited with \
         non-zero exit code %d"
        cmd
        n
