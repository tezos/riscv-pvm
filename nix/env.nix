{
  stdenv,
  mkShell,
  rustup,
  taplo,
  cargo-audit,
  cargo-nextest,
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
