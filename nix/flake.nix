let
  endsWith =
    suffix: str:
    let
      found = builtins.substring (
        builtins.stringLength str - builtins.stringLength suffix
      ) (builtins.stringLength suffix) str;
    in
    found == suffix;

  flakeLock = builtins.fromJSON (builtins.readFile ../flake.lock);

  flakeCompatSrc = fetchTarball {
    url =
      flakeLock.nodes.flake-compat.locked.url
        or "https://github.com/edolstra/flake-compat/archive/${flakeLock.nodes.flake-compat.locked.rev}.tar.gz";
    sha256 = flakeLock.nodes.flake-compat.locked.narHash;
  };

  filterSrc = path: type: endsWith "/nix" path || endsWith ".nix" path || endsWith "/flake.lock" path;

  flakeSrc = builtins.filterSource filterSrc ../.;
in
import flakeCompatSrc { src = flakeSrc; }
