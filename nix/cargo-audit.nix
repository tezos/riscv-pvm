# Derived from Nixpkgs:
# https://github.com/NixOS/nixpkgs/blob/c6245e83d836d0433170a16eb185cefe0572f8b8/pkgs/by-name/ca/cargo-audit/package.nix#L36
# Patched for 0.22.0

{
  rustPlatform,
  fetchCrate,
  pkg-config,
  openssl,
  zlib,
}:

rustPlatform.buildRustPackage rec {
  pname = "cargo-audit";
  version = "0.22.0";

  src = fetchCrate {
    inherit pname version;
    hash = "sha256-Ha2yVyu9331NaqiW91NEwCTIeW+3XPiqZzmatN5KOws=";
  };

  cargoHash = "sha256-f8nrW1l7UA8sixwqXBD1jCJi9qyKC5tNl/dWwCt41Lk=";

  nativeBuildInputs = [
    pkg-config
  ];

  buildInputs = [
    openssl
    zlib
  ];

  buildFeatures = [ "fix" ];

  doCheck = false;
}
