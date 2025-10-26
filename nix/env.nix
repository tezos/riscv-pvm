{
  stdenv,
  mkShell,
  rustup,
  taplo,
  cargo-audit,
  cargo-nextest,
  ocaml-ng,
  cacert,
  curl,
  libclang,
  libiconv,
  fswatch,
  inotify-tools,
  pkgsCross,
}:
mkShell {
  name = "tezos-shell";

  LIBCLANG_PATH = "${libclang.lib}/lib";

  packages = [
    # Rust
    rustup

    # For RISC-V kernel cross-compilation
    pkgsCross.riscv64.pkgsStatic.stdenv.cc

    # Utilities
    taplo
    cargo-audit
    cargo-nextest

    # Make sure there is an OCaml compiler available
    ocaml-ng.ocamlPackages_5_2.ocaml

    # These are needed for downloads and stuff
    cacert
    curl
  ]
  ++ (
    if stdenv.isDarwin then
      [
        libiconv
        fswatch
      ]
    else
      [
        inotify-tools
      ]
  );
}
