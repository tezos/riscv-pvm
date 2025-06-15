#!/usr/bin/env bash

set -e

# Git ref to checkout
ref=""

# Parse command line arguments
while getopts "r:" OPTION; do
  case "$OPTION" in
  r)
    ref="$OPTARG"
    ;;
  *)
    echo "Invalid parameter"
    exit 1
    ;;
  esac
done

# Make sure a ref is provided, otherwise we have nothing to benchmark
if [[ -z "$ref" ]]; then
  echo "No ref provided"
  exit 1
fi

# Make sure this process and all its children run with the highest priority to avoid flakiness
# during the benchmark runs
sudo renice -20 -p $$ >/dev/null

# We need to source the Nix environment to gain access to our favourite tools
. /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh

# Check out the repository so we have the source code to benchmark
dir=$(mktemp -d)

cd "$dir"
git init --quiet .
git config --local gc.auto 0
git remote add origin git@github.com:tezos/riscv-pvm.git
git fetch --quiet --depth 1 origin "+$ref"
git checkout --quiet FETCH_HEAD

# Build prerequisites ahead of time
nix develop --command make -C src/riscv/jstz build-kernel &>/dev/null

# Generate inbox file for benchmark runs
inbox_file=$(mktemp)
nix develop --command cargo run --quiet --manifest-path src/riscv/jstz/Cargo.toml --bin inbox-bench -- generate --transfers 15 --inbox-file "$inbox_file"

# Run the benchmark
result_dir=$(mktemp -d)
result_args=()
for i in $(seq 1 10); do 
  nix develop --command cargo run --release --quiet --manifest-path src/riscv/Cargo.toml -- run --input src/riscv/jstz/target/riscv64gc-unknown-linux-musl/release/jstz --inbox-file "$inbox_file" --timings > $result_dir/$i.json
  result_args+=("--log-file=$result_dir/$i.json")
done

# Collect results and display them
nix develop --command cargo run --quiet --manifest-path src/riscv/jstz/Cargo.toml --bin inbox-bench -- results --inbox-file "$inbox_file" --expected-transfers 15 "${result_args[@]}"
