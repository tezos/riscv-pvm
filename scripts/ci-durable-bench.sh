#!/usr/bin/env bash

# Registry lifecycle latencies, measured on the reference machine.
#
# Latencies have to be measured here rather than on a shared runner, or the numbers are noise. The
# space metrics do not need it — they are byte counts — so CI measures those on a shared runner.
#
# The scale and the scenarios live in durable-storage/Makefile, not here, so there is one
# definition of what "the reference run" means. Where the machine keeps its prepopulated registries
# is the opposite: it is a property of the machine, so it lives here.

set -e

# Git ref to checkout
ref=""

# Where prepopulated registries are kept between runs, and how many to keep. Override the root if
# the home directory is not on the volume with the room — a registry at the reference scale is
# hundreds of megabytes.
cache_root="${LIFECYCLE_CACHE_ROOT:-$HOME/.cache/riscv-pvm/lifecycle}"
cache_keep="${LIFECYCLE_CACHE_KEEP:-3}"

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

# Prepopulating the registry costs far more than the operations measured against it, and it
# produces the same registry every time, so the machine keeps it and later runs check it out
# instead. This is what makes the benchmark cheap enough to run on a pull request.
#
# What a registry cannot outlive is a change to how it is written or read: checking out one written
# by a different format either fails outright or measures a tree the revision under test would
# never have built. The three trees below decide that between them — the two storage crates define
# the format, the benchmark defines what is populated — so keying the cache on their content leaves
# every revision that would have written the same registry sharing one, and gives a revision that
# would not its own.
fingerprint=$(git rev-parse "HEAD:data/src" "HEAD:durable-storage/src" "HEAD:durable-storage/benches" | sha1sum | cut -c1-16)

export LIFECYCLE_REGISTRY_DIR="$cache_root/$fingerprint"
mkdir -p "$LIFECYCLE_REGISTRY_DIR"

# The workflow's concurrency group is what stops two benchmark runs sharing the machine, but the
# registry cannot rely on it: cancelling a job does not stop the command it left running here, and
# nothing stops a run started by hand. Two runs in one registry would leave it unusable for both.
exec 9>"$cache_root/lock"
flock -w 7200 9

# Newest first, keep that many, remove the rest. A run touches the registry it uses, so what goes is
# whatever nothing has asked for in the longest time.
touch "$LIFECYCLE_REGISTRY_DIR"
ls -dt "$cache_root"/*/ 2>/dev/null | tail -n +$((cache_keep + 1)) | while read -r stale; do
  rm -rf "$stale"
done

# The build and the benchmark harness still want scratch space of their own; the registry is no
# longer part of it.
scratch=$(mktemp -d)
export TMPDIR="$scratch"

echo "### Registry lifecycle latencies"
echo
echo '```'
nix develop --command make -C durable-storage lifecycle-metrics-reference
echo '```'

# Clean up
rm -rf "$dir" "$scratch"
