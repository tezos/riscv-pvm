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

open! Import

module type Args = sig
  module Hash : Brassaia.Hash.S

  module File_Manager : File_manager.S with type Index.key = Hash.t

  module Dispatcher : Dispatcher.S with module File_Manager = File_Manager

  module Inode :
    Inode.Persistent
      with type hash := Hash.t
       and type key = Hash.t Pack_key.t
       and type file_manager = File_Manager.t
       and type dispatcher = Dispatcher.t

  module Contents_pack :
    Pack_store.S
      with type hash := Hash.t
       and type key = Hash.t Pack_key.t
       and type dispatcher = Dispatcher.t

  module Io_index : Brassaia_index.Index.Platform.S
end

module type Sigs = sig
  module Make (Args : Args) : sig
    open Args

    module Export : sig
      type t

      val init :
        Brassaia.config -> read Contents_pack.t -> read Inode.Pack.t -> t

      val run :
        ?on_disk:[`Path of string] ->
        t ->
        (Contents_pack.value -> unit) ->
        (Inode.Snapshot.inode -> unit) ->
        Hash.t Pack_key.t * Pack_value.Kind.t ->
        int

      val close :
        t ->
        ( unit,
          [> `Double_close
          | `Index_failure of string
          | `Io_misc of File_Manager.Io.misc_error
          | `Pending_flush
          | `Ro_not_allowed ] )
        result
    end

    module Import : sig
      type t

      val init :
        ?on_disk:[`Path of string | `Reuse] ->
        int ->
        read Contents_pack.t ->
        read Inode.Pack.t ->
        t

      val save_contents : t -> Contents_pack.value -> Hash.t Pack_key.t

      val save_inodes : t -> Inode.Snapshot.inode -> Hash.t Pack_key.t

      val close : t -> unit
    end
  end
end
