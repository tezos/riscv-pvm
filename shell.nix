let
  inputs = import ./nix/flake-inputs.nix;

  pkgs = inputs.nixpkgs.legacyPackages.${builtins.currentSystem};
in
  pkgs.callPackage ./nix/env.nix {}
