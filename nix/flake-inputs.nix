let
  endsWith = suffix: str: let
    found =
      builtins.substring
      (builtins.stringLength str - builtins.stringLength suffix) (builtins.stringLength suffix)
      str;
  in
    found == suffix;
in
  (
    import
    (
      let
        lock = builtins.fromJSON (builtins.readFile ../flake.lock);
      in
        fetchTarball {
          url = lock.nodes.flake-compat.locked.url or "https://github.com/edolstra/flake-compat/archive/${lock.nodes.flake-compat.locked.rev}.tar.gz";
          sha256 = lock.nodes.flake-compat.locked.narHash;
        }
    )
    {
      src =
        builtins.filterSource
        (path: type: endsWith ".nix" path || endsWith "flake.lock" path)
        ../.;
    }
  )
  .defaultNix
  .inputs
