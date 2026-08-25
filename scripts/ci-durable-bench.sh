#!/usr/bin/env bash

# Registry lifecycle latencies, measured on the reference machine.
#
# Latencies have to be measured here rather than on a shared runner, or the numbers are noise. The
# space metrics do not need it — they are byte counts — so CI measures those on a shared runner.
#
# The scale and the scenarios live in durable-storage/Makefile, not here, so there is one
# definition of what "the reference run" means.

set -e

# Git ref to checkout
ref=""

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

# The benchmark prepopulates the databases under TMPDIR and removes them afterwards, but it needs
# room while it runs.
scratch=$(mktemp -d)
export TMPDIR="$scratch"

echo "### Registry lifecycle latencies"
echo
echo '```'
nix develop --command make -C durable-storage lifecycle-metrics-reference
echo '```'

# Clean up
rm -rf "$dir" "$scratch"
