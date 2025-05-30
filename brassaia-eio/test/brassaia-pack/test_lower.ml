(*
 * Copyright (c) 2023 Tarides <contact@tarides.com>
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

open! Import
open Common

let src = Logs.Src.create "tests.lower" ~doc:"Test lower"

let mkroot path = Filename.concat "_build" ("brassia-eio-" ^ path)

module Log = (val Logs.src_log src : Logs.LOG)

module Io = Brassaia_pack_unix.Io.Unix

let ( let$ ) res f = f @@ Result.get_ok res

module Direct_tc = struct
  module Control = Brassaia_pack_unix.Control_file.Volume
  module Lower = Brassaia_pack_unix.Lower
  module Sparse_file = Brassaia_pack_unix.Sparse_file

  let create_control volume_path payload =
    let path = Brassaia_pack.Layout.V5.Volume.control ~root:volume_path in
    Control.create_rw ~path ~tmp_path:None ~overwrite:true payload

  let test_empty () =
    let lower_root = create_lower_root () in
    let$ lower = Lower.open_volumes ~readonly:false ~volume_num:0 lower_root in
    Alcotest.(check int) "0 volumes" 0 (Lower.volume_num lower) ;
    let _ = Lower.close lower in
    ()

  let test_volume_num () =
    let lower_root = create_lower_root () in
    let result = Lower.open_volumes ~readonly:false ~volume_num:1 lower_root in
    match result with
    | Error (`Volume_missing _) -> ()
    | _ -> Alcotest.fail "volume_num too high should return an error"

  let test_add_volume () =
    let lower_root = create_lower_root () in
    let$ lower = Lower.open_volumes ~readonly:false ~volume_num:0 lower_root in
    let$ _ = Lower.add_volume lower in
    Alcotest.(check int) "1 volume" 1 (Lower.volume_num lower) ;
    let$ _ = Lower.reload ~volume_num:1 lower in
    Alcotest.(check int) "1 volume after reload" 1 (Lower.volume_num lower) ;
    let _ = Lower.close lower in
    ()

  let test_add_volume_ro () =
    let lower_root = create_lower_root () in
    let$ lower = Lower.open_volumes ~readonly:true ~volume_num:0 lower_root in
    let result = Lower.add_volume lower in
    let () =
      match result with
      | Error `Ro_not_allowed -> ()
      | _ -> Alcotest.fail "cannot add volume to ro lower"
    in
    let _ = Lower.close lower in
    ()

  let test_add_multiple_empty () =
    let lower_root = create_lower_root () in
    let$ lower = Lower.open_volumes ~readonly:false ~volume_num:0 lower_root in
    let$ _ = Lower.add_volume lower in
    let result = Lower.add_volume lower |> Result.get_error in
    let () =
      match result with
      | `Multiple_empty_volumes -> ()
      | _ -> Alcotest.fail "cannot add multiple empty volumes"
    in
    let _ = Lower.close lower in
    ()

  let test_find_volume () =
    let lower_root = create_lower_root () in
    let$ lower = Lower.open_volumes ~readonly:false ~volume_num:0 lower_root in
    let$ volume = Lower.add_volume lower in
    let payload =
      Brassaia_pack_unix.Control_file.Payload.Volume.Latest.
        {
          start_offset = Int63.zero;
          end_offset = Int63.of_int 42;
          mapping_end_poff = Int63.zero;
          checksum = Int63.zero;
        }
    in
    let _ = create_control (Lower.Volume.path volume) payload in
    let volume = Lower.find_volume ~off:(Int63.of_int 21) lower in
    Alcotest.(check bool)
      "volume not found before reload"
      false
      (Option.is_some volume) ;
    let$ _ = Lower.reload ~volume_num:1 lower in
    let volume = Lower.find_volume ~off:(Int63.of_int 21) lower in
    Alcotest.(check bool) "found volume" true (Option.is_some volume) ;
    let _ = Lower.close lower in
    ()

  let test_read_exn () =
    let lower_root = create_lower_root () in
    let$ lower = Lower.open_volumes ~readonly:false ~volume_num:0 lower_root in
    let$ volume = Lower.add_volume lower in
    (* Manually create mapping, data, and control file for volume.

       Then test that reloading and read_exn work as expected. *)
    let volume_path = Lower.Volume.path volume in
    let mapping_path =
      Brassaia_pack.Layout.V5.Volume.mapping ~root:volume_path
    in
    let data_path = Brassaia_pack.Layout.V5.Volume.data ~root:volume_path in
    let test_str = "hello" in
    let len = String.length test_str in
    let$ sparse =
      Sparse_file.Ao.open_ao
        ~mapping_size:Int63.zero
        ~mapping:mapping_path
        ~data:data_path
    in
    let seq = List.to_seq [test_str] in
    Sparse_file.Ao.append_seq_exn sparse ~off:Int63.zero seq ;
    let end_offset = Sparse_file.Ao.end_off sparse in
    let$ _ = Sparse_file.Ao.flush sparse in
    let$ _ = Sparse_file.Ao.close sparse in
    let$ mapping_end_poff = Io.size_of_path mapping_path in
    let payload =
      Brassaia_pack_unix.Control_file.Payload.Volume.Latest.
        {
          start_offset = Int63.zero;
          end_offset;
          mapping_end_poff;
          checksum = Int63.zero;
        }
    in
    let _ = create_control (Lower.Volume.path volume) payload in
    let$ _ = Lower.reload ~volume_num:1 lower in
    let buf = Bytes.create len in
    let _ = Lower.read_exn ~off:Int63.zero ~len lower buf in
    Alcotest.(check string)
      "check volume read"
      test_str
      (Bytes.unsafe_to_string buf) ;
    let _ = Lower.close lower in
    ()
end

module Store_tc = struct
  module Store = struct
    module Maker = Brassaia_pack_unix.Maker (Conf)
    include Maker.Make (Schema)
  end

  let test_dir = "_build"

  let fresh_roots =
    let c = ref 0 in
    fun ?(make_root = true) () ->
      incr c ;
      let name =
        Filename.concat test_dir ("test_lower_store_" ^ string_of_int !c)
      in
      Common.rm_dir name ;
      let$ _ = if make_root then Io.mkdir name else Ok () in
      let lower = Filename.concat name "lower" in
      Common.rm_dir lower ;
      (name, lower)

  let config ?(readonly = false) ?(fresh = false) ?lower_root root =
    Brassaia_pack.(
      config
        ~readonly
        ~indexing_strategy:Indexing_strategy.minimal
        ~fresh
        ~lower_root
        root)

  let init ?(readonly = false) ?(fresh = true) ?(include_lower = true) () =
    let root, lower_root = fresh_roots () in
    let lower_root = if include_lower then Some lower_root else None in
    config ~readonly ~fresh ?lower_root root |> Store.Repo.init

  let count_volumes repo =
    let open Store.Internal in
    file_manager repo |> File_manager.lower
    |> Option.map Lower.volume_num
    |> Option.value ~default:0

  let volume_path repo offset =
    let open Store.Internal in
    let lower = file_manager repo |> File_manager.lower in
    let volume =
      match lower with
      | None -> Alcotest.fail "expected lower"
      | Some l -> Lower.find_volume ~off:offset l
    in
    match volume with
    | None -> Alcotest.fail "expected volume"
    | Some v -> Lower.Volume.path v

  let generation repo =
    let open Store.Internal in
    let ({status; _} : Brassaia_pack_unix.Control_file.Payload.Upper.Latest.t) =
      file_manager repo |> File_manager.control |> File_manager.Control.payload
    in
    match status with
    | Gced {generation; _} -> generation
    | _ -> Alcotest.fail "expected gced status"

  (* Reads all objects from the repo by iterating its index and folding its commit trees. *)
  let read_everything repo =
    let fm = Store.Internal.file_manager repo in
    let index = Store.Internal.File_manager.index fm in
    let commits = ref [] in
    let () =
      Store.Internal.Index.iter
        (fun hash (_offset, _len, kind) ->
          match kind with
          | Brassaia_pack.Pack_value.Kind.Commit_v2 ->
              commits := hash :: !commits
          | _ -> ())
        index
    in
    List.map
      (fun hash ->
        [%log.debug "read %a" Brassaia.Type.(pp Store.Hash.t) hash] ;
        match Store.Commit.of_hash repo hash with
        | None -> Alcotest.fail "failed to read commit"
        | Some commit -> Store.Tree.fold (Store.Commit.tree commit) ())
      !commits

  let test_create () =
    let repo = init () in
    (* A newly created store with a lower should have an empty volume. *)
    let volume_num = count_volumes repo in
    Alcotest.(check int) "volume_num is 1" 1 volume_num ;
    Store.Repo.close repo

  let test_create_nested () =
    let root, lower_root = fresh_roots ~make_root:false () in
    let repo = config ~fresh:true ~lower_root root |> Store.Repo.init in
    let volume_num = count_volumes repo in
    Alcotest.(check int) "volume_num is 1" 1 volume_num ;
    Store.Repo.close repo

  let test_open_rw_lower () =
    let root, lower_root = fresh_roots ~make_root:false () in
    let repo = config ~fresh:true root |> Store.Repo.init in
    let () = Store.Repo.close repo in
    let repo = config ~fresh:false ~lower_root root |> Store.Repo.init in
    let volume_num = count_volumes repo in
    Alcotest.(check int) "volume_num is 1" 1 volume_num ;
    Store.Repo.close repo

  let test_add_volume_during_gc () =
    let repo = init () in
    let main = Store.main repo in
    let () =
      Store.set_exn
        ~info:(fun () -> Store.Info.init ~author:"tester" Int64.zero)
        main
        ["a"]
        "a"
    in
    let c = Store.Head.get main in
    let _ = Store.Gc.start_exn repo (Store.Commit.key c) in
    let () =
      Alcotest.check_raises
        "add volume during gc"
        (Brassaia_pack_unix.Errors.Pack_error `Add_volume_forbidden_during_gc)
        (fun () -> Store.add_volume repo |> Lwt.return)
    in
    Store.Repo.close repo

  let test_add_volume_wo_lower () =
    let repo = init ~include_lower:false () in
    let () =
      Alcotest.check_raises
        "add volume w/o lower"
        (Brassaia_pack_unix.Errors.Pack_error `Add_volume_requires_lower)
        (fun () -> Store.add_volume repo |> Lwt.return)
    in
    Store.Repo.close repo

  let test_add_volume_reopen () =
    let root, lower_root = fresh_roots () in
    let repo = Store.Repo.init (config ~fresh:true ~lower_root root) in
    let main = Store.main repo in
    let info () = Store.Info.init ~author:"test" Int64.zero in
    let () = Store.set_exn ~info main ["a"] "a" in
    let c1 = Store.Head.get main in
    let _ = Store.Gc.start_exn repo (Store.Commit.key c1) in
    let _ = Store.Gc.finalise_exn ~wait:true repo in
    let () = Store.add_volume repo in
    Alcotest.(check int) "two volumes" 2 (count_volumes repo) ;
    let _ = Store.Repo.close repo in
    let repo = Store.Repo.init (config ~fresh:false ~lower_root root) in
    Alcotest.(check int) "two volumes after re-open" 2 (count_volumes repo) ;
    Store.Repo.close repo

  let test_migrate () =
    let root, lower_root = fresh_roots () in
    (* Create without a lower *)
    let repo = Store.Repo.init (config ~fresh:true root) in
    Alcotest.(check int) "volume_num is 0" 0 (count_volumes repo) ;
    let main = Store.main repo in
    let info () = Store.Info.init ~author:"test" Int64.zero in
    let () = Store.set_exn ~info main ["a"] "a" in
    let () = Store.Repo.close repo in
    (* Reopen with a lower to trigger the migration *)
    let repo = Store.Repo.init (config ~lower_root root) in
    Alcotest.(check int) "volume_num is 1" 1 (count_volumes repo) ;
    let main = Store.main repo in
    let a = Store.get main ["a"] in
    Alcotest.(check string) "migrated commit" "a" a ;
    Alcotest.(check bool)
      "no latest GC commit"
      true
      (Option.is_none (Store.Gc.latest_gc_target repo)) ;
    let () = Store.set_exn ~info main ["a"] "b" in
    let () = Store.Repo.close repo in
    (* Reopen with the same lower and check reads *)
    let repo = Store.Repo.init (config ~lower_root root) in
    Alcotest.(check int) "volume_num is 1" 1 (count_volumes repo) ;
    let main = Store.main repo in
    let b = Store.get main ["a"] in
    Alcotest.(check string) "upper commit" "b" b ;
    let main_commit = Store.Head.get main in
    let parent_key = List.hd @@ Store.Commit.parents main_commit in
    let parent = Store.Commit.of_key repo parent_key in
    let previous_tree = Store.Commit.tree @@ Option.get parent in
    let a_opt = Store.Tree.find previous_tree ["a"] in
    Alcotest.(check (option string)) "upper to lower" (Some "a") a_opt ;
    let _ = read_everything repo in
    Store.Repo.close repo

  (* Tests that dead header is handled appropriately *)
  let test_migrate_v2 () =
    let ( / ) = Filename.concat in
    let root_archive =
      "brassaia-eio" / "test" / "brassaia-pack" / "data"
      / "version_2_to_3_always"
    in
    let root = mkroot "test_lower_migrate_v2" in
    setup_test_env ~root_archive ~root_local_build:root ;
    let lower_root = root / "lower" in
    (* Open store and trigger migration. This should succeed. *)
    let repo = Store.Repo.init (config ~fresh:false ~lower_root root) in
    let _ = read_everything repo in
    Store.Repo.close repo

  let test_migrate_v3 () =
    (* minimal indexing *)
    let ( / ) = Filename.concat in
    let root_archive =
      "brassaia-eio" / "test" / "brassaia-pack" / "data" / "version_3_minimal"
    in
    let root = mkroot "test_lower_migrate_v3_minimal" in
    setup_test_env ~root_archive ~root_local_build:root ;
    let lower_root = root / "lower" in
    (* Open store and trigger migration. This should succeed. *)
    let repo = Store.Repo.init (config ~fresh:false ~lower_root root) in
    let _ = read_everything repo in
    let _ = Store.Repo.close repo in

    (* always indexing *)
    let ( / ) = Filename.concat in
    let root_archive =
      "brassaia-eio" / "test" / "brassaia-pack" / "data" / "version_3_always"
    in
    let root = mkroot "test_lower_migrate_v3_always" in
    setup_test_env ~root_archive ~root_local_build:root ;
    let lower_root = root / "lower" in
    (* Open store and trigger migration. This should succeed. *)
    let repo = Store.Repo.init (config ~fresh:false ~lower_root root) in
    let _ = read_everything repo in
    Store.Repo.close repo

  let test_migrate_then_gc () =
    let root, lower_root = fresh_roots () in
    (* Create without a lower *)
    let repo = Store.Repo.init (config ~fresh:true root) in
    Alcotest.(check int) "volume_num is 0" 0 (count_volumes repo) ;
    let main = Store.main repo in
    let info () = Store.Info.init ~author:"test" Int64.zero in
    let () = Store.set_exn ~info main ["a"] "a" in
    let () = Store.Repo.close repo in
    (* Reopen with a lower to trigger the migration *)
    let repo = Store.Repo.init (config ~lower_root root) in
    Alcotest.(check int) "volume_num is 1" 1 (count_volumes repo) ;
    (* Add two commits *)
    let main = Store.main repo in
    let () = Store.set_exn ~info main ["b"] "b" in
    let main = Store.main repo in
    let b_commit = Store.Head.get main in
    let () = Store.set_exn ~info main ["c"] "c" in
    (* GC at [b] requires reading [a] data from the lower volume *)
    let _ = Store.Gc.start_exn repo (Store.Commit.key b_commit) in
    let _ = Store.Gc.finalise_exn ~wait:true repo in
    let _ = read_everything repo in
    Store.Repo.close repo

  let test_migrate_then_gc_in_lower () =
    let root, lower_root = fresh_roots () in
    (* Create without a lower *)
    let repo = Store.Repo.init (config ~fresh:true root) in
    Alcotest.(check int) "volume_num is 0" 0 (count_volumes repo) ;
    let main = Store.main repo in
    let info () = Store.Info.init ~author:"test" Int64.zero in
    let () = Store.set_exn ~info main ["a"] "a" in
    let a_commit = Store.Head.get main in
    let () = Store.set_exn ~info main ["b"] "b" in
    let () = Store.Repo.close repo in
    (* Reopen with a lower to trigger the migration *)
    let repo = Store.Repo.init (config ~lower_root root) in
    Alcotest.(check int) "volume_num is 1" 1 (count_volumes repo) ;
    (* [a] is now in the lower but GC should still succeed

       Important: we call GC on a commit that is not the latest in
       the lower (ie [b]) to ensure its offset is not equal to the start
       offset of the upper. *)
    let _ = Store.Gc.start_exn repo (Store.Commit.key a_commit) in
    let _ = Store.Gc.finalise_exn ~wait:true repo in
    Store.Repo.close repo

  let test_volume_data_locality () =
    let root, lower_root = fresh_roots () in
    let repo = Store.Repo.init (config ~fresh:true ~lower_root root) in
    let main = Store.main repo in
    let info () = Store.Info.init ~author:"test" Int64.zero in
    [%log.debug "add c1"] ;
    let () = Store.set_exn ~info main ["c1"] "a" in
    let c1 = Store.Head.get main in
    [%log.debug "GC c1"] ;
    let _ = Store.Gc.start_exn repo (Store.Commit.key c1) in
    let _ = Store.Gc.finalise_exn ~wait:true repo in
    let () = Store.add_volume repo in
    [%log.debug "add c2, c3, c4"] ;
    let () = Store.set_exn ~info main ["c2"] "b" in
    let () = Store.set_exn ~info main ["c3"] "c" in
    let c3 = Store.Head.get main in
    let () = Store.set_exn ~info main ["c4"] "d" in
    let () = Store.set_exn ~info main ["c5"] "e" in
    let c5 = Store.Head.get main in
    [%log.debug "GC c5"] ;
    let _ = Store.Gc.start_exn repo (Store.Commit.key c5) in
    let _ = Store.Gc.finalise_exn ~wait:true repo in
    let get_direct_key key =
      match Brassaia_pack_unix.Pack_key.inspect key with
      | Direct {offset; hash; length; volume_identifier} ->
          (offset, hash, length, volume_identifier)
      | _ -> assert false
    in
    (* NOTE: we need to lookup c3 again so that its volume
       identifier is on its key *)
    let _, hash, _, _ = get_direct_key (Store.Commit.key c3) in
    let c3 = Store.Commit.of_hash repo hash in
    let c3 = Option.get c3 in
    let _, _, _, identifier = get_direct_key (Store.Commit.key c3) in
    let identifier = Option.get identifier in
    [%log.debug "Check c3 tree items are in volume %s" identifier] ;
    let c3 = Store.Commit.of_key repo (Store.Commit.key c3) in
    let tree = Store.Commit.tree (Option.get c3) in
    let () =
      let get_volume_identifier key =
        let _, _, _, identifier = get_direct_key key in
        match identifier with
        | None -> Alcotest.fail "expected volume identifier"
        | Some v -> v
      in
      (* Check every item of c3's tree to ensure it is in the first volume. *)
      Store.Tree.fold
        ~tree:(fun _p t a ->
          let kinded_key = Store.Tree.key t in
          let key_identifier =
            match kinded_key with
            | None -> assert false
            | Some (`Contents k) -> get_volume_identifier k
            | Some (`Node k) -> get_volume_identifier k
          in
          [%log.debug "identifier: %s" key_identifier] ;
          Alcotest.(check string)
            "key is in expected volume"
            identifier
            key_identifier ;
          a)
        tree
        ()
    in
    Store.Repo.close repo

  let test_cleanup () =
    let root, lower_root = fresh_roots () in
    [%log.debug "create store with data and run GC"] ;
    let repo = Store.Repo.init (config ~fresh:true ~lower_root root) in
    let main = Store.main repo in
    let info () = Store.Info.init ~author:"test" Int64.zero in
    let () = Store.set_exn ~info main ["a"] "a" in
    let c1 = Store.Head.get main in
    let _ = Store.Gc.start_exn repo (Store.Commit.key c1) in
    let _ = Store.Gc.finalise_exn ~wait:true repo in
    let volume_root = volume_path repo Int63.zero in
    let generation = generation repo in
    let () = Store.Repo.close repo in
    [%log.debug "test volume.1.control is moved to volume.control"] ;
    let volume_cf_gen_path =
      Brassaia_pack.Layout.V5.Volume.control_gc_tmp
        ~generation
        ~root:volume_root
    in
    let volume_cf_path =
      Brassaia_pack.Layout.V5.Volume.control ~root:volume_root
    in
    let$ () = Io.move_file ~src:volume_cf_path ~dst:volume_cf_gen_path in
    let repo = Store.Repo.init (config ~fresh:false ~lower_root root) in
    let () =
      match Io.classify_path volume_cf_path with
      | `File -> [%log.debug "control file exists"]
      | _ -> Alcotest.fail "expected conrol file"
    in
    let () =
      match Io.classify_path volume_cf_gen_path with
      | `No_such_file_or_directory -> [%log.debug "gc control file unlinked"]
      | _ -> Alcotest.fail "expected conrol gen file to not exist"
    in
    Store.Repo.close repo
end

module Store = struct
  include Store_tc

  let tests =
    Alcotest.
      [
        quick_tc_eio "create store" test_create;
        quick_tc_eio "create nested" test_create_nested;
        quick_tc_eio "open rw with lower" test_open_rw_lower;
        quick_tc_eio "add volume with no lower" test_add_volume_wo_lower;
        quick_tc_eio "add volume during gc" test_add_volume_during_gc;
        quick_tc_eio "control file updated after add" test_add_volume_reopen;
        quick_tc_eio "add volume and reopen" test_add_volume_reopen;
        quick_tc_eio "create without lower then migrate" test_migrate;
        quick_tc_eio "migrate v2" test_migrate_v2;
        quick_tc_eio "migrate v3" test_migrate_v3;
        quick_tc_eio "migrate then gc" test_migrate_then_gc;
        quick_tc_eio "migrate then gc in lower" test_migrate_then_gc_in_lower;
        quick_tc_eio "test data locality" test_volume_data_locality;
        quick_tc_eio "test cleanup" test_cleanup;
      ]
end

module Direct = struct
  include Direct_tc

  let tests =
    Alcotest.
      [
        quick_tc_eio "empty lower" test_empty;
        quick_tc_eio "volume_num too high" test_volume_num;
        quick_tc_eio "add volume" test_add_volume;
        quick_tc_eio "add volume ro" test_add_volume_ro;
        quick_tc_eio "add multiple empty" test_add_multiple_empty;
        quick_tc_eio "find volume" test_find_volume;
        quick_tc_eio "test read_exn" test_read_exn;
      ]
end
