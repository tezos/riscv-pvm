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
  nodejs,
  pnpm,
  gh,
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

    # NodeJS stuff for documentation
    nodejs
    pnpm

    # hacking
    gh
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
