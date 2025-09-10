#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
#
# SPDX-License-Identifier: MIT

# Build and run the jstz TPS benchmark with the specified number of transfers

set -e

USAGE="Usage:
  REQUIRED:
    -t <num_transfers>
  OPTIONAL:
    -s: static inbox
    -p: profile with samply
    -n: run natively
    -i <num_iterations>: number of runs
    -w <num_warmup_tx>: number of warmup transactions
       Run additional transactions before the benchmarking scenario.
       This allows performance to be measured excluding warmup-time for
       various caches to be populated or JIT compilation to take place.
    -j <disable|inline>: disable jit / use inline jit"

DEFAULT_ROLLUP_ADDRESS="sr163Lv22CdE8QagCwf48PWDTquk6isQwv57"

ITERATIONS="1"
TX=""
STATIC_INBOX=""
SANDBOX_BIN="riscv-sandbox"
SANDBOX_ENABLE_FEATURES=()
PROFILING_WRAPPER=""
SAMPLY_OUT="riscv-sandbox-profile.json"
NATIVE=""
JSTZ_SANDBOX_PARAMS=("--input" "kernels/jstz/target/riscv64gc-unknown-linux-musl/release/jstz")
WARMUP_TX="0"

CURR=$(pwd)
RISCV_DIR=$(dirname "$0")/..
cd "$RISCV_DIR"

while getopts "i:t:m:w:spnj:" OPTION; do
  case "$OPTION" in
  i)
    ITERATIONS="$OPTARG"
    ;;
  t)
    TX="$OPTARG"
    ;;
  s)
    STATIC_INBOX="y"
    ;;
  p)
    SANDBOX_BIN="riscv-sandbox.prof"
    PROFILING_WRAPPER="samply record -s -o $SAMPLY_OUT"
    ;;
  n)
    NATIVE=$(make --silent -C kernels/jstz print-native-target | grep -wv make)
    ;;
  j)
    case "$OPTARG" in
    i*)
      SANDBOX_ENABLE_FEATURES+=("inline-jit")
      ;;
    d*)
      SANDBOX_ENABLE_FEATURES+=("disable-jit")
      ;;
    *)
      echo "-j <disable|inline>"
      exit 1
      ;;
    esac
    ;;
  w)
    WARMUP_TX="$OPTARG"
    ;;
  *)
    echo "$USAGE"
    exit 1
    ;;
  esac
done

if [ -z "$TX" ]; then
  echo "$USAGE"
  exit 1
fi

TOTAL_TX=$(("$TX" + "$WARMUP_TX"))

if [ -n "$NATIVE" ] && [ -z "$STATIC_INBOX" ]; then
  echo "Native compilation without static inbox unsupported"
  echo "$USAGE"
  exit 1
fi

echo "[INFO]: building sandbox"
make -C tools/sandbox "SANDBOX_ENABLE_FEATURES=${SANDBOX_ENABLE_FEATURES[*]}" "$SANDBOX_BIN" &> /dev/null
echo "[INFO]: building bench tool"
make -C kernels/jstz inbox-bench &> /dev/null

DATA_DIR=${DATA_DIR:=$(mktemp -d)}

echo "[INFO]: generating $TX transfers with $WARMUP_TX warmup tx"
INBOX_FILE="${DATA_DIR}/inbox.json"
RUN_INBOX="$INBOX_FILE"
kernels/jstz/inbox-bench generate --inbox-file "$INBOX_FILE" --transfers "$TOTAL_TX"

log_file_args=()

##########
# RISC-V #
##########
build_jstz_riscv() {
  if [ "$STATIC_INBOX" = "y" ]; then
    INBOX_FILE="$INBOX_FILE" make -C kernels/jstz build-kernel-static &> /dev/null
    RUN_INBOX="$DATA_DIR"/empty.json
    echo "[]" > "$RUN_INBOX"
  else
    make -C kernels/jstz build-kernel &> /dev/null
  fi
}

run_jstz_riscv() {
  LOG="$DATA_DIR/log.$1.log"
  $PROFILING_WRAPPER "tools/sandbox/$SANDBOX_BIN" run \
    "${JSTZ_SANDBOX_PARAMS[@]}" \
    --inbox-file "$RUN_INBOX" \
    --address "$DEFAULT_ROLLUP_ADDRESS" \
    --timings > "$LOG"
  log_file_args+=("--log-file=$LOG")
}

##########
# Native #
##########
build_jstz_native() {
  INBOX_FILE=$INBOX_FILE make -C kernels/jstz build-kernel-native &> /dev/null
}

run_jstz_native() {
  LOG="$DATA_DIR/log.$1.log"
  $PROFILING_WRAPPER kernels/jstz/target/"$NATIVE"/release/jstz \
    --timings > "$LOG" 2> /dev/null
  log_file_args+=("--log-file=$LOG")
}

#########
# Build #
#########
echo "[INFO]: building jstz"

if [ -z "$NATIVE" ]; then
  build_jstz_riscv
  echo "[INFO]: running $TX transfers (riscv) "
else
  build_jstz_native
  echo "[INFO]: running $TX transfers ($NATIVE) "
fi

#################
# Run & Collect #
#################
run_jstz() {
  echo -ne "\r\033[2K[INFO]: Run $1 / $ITERATIONS"
  if [ -z "$NATIVE" ]; then
    run_jstz_riscv "$1"
  else
    run_jstz_native "$1"
  fi

  if [ -n "$PROFILING_WRAPPER" ]; then
    echo -e "\n[INFO]: Samply data saved to: $SAMPLY_OUT"
  fi
}

collect() {
  echo -e "\033[1m"
  kernels/jstz/inbox-bench results --inbox-file "$INBOX_FILE" "${log_file_args[@]}" --expected-transfers "$TOTAL_TX" --exclude-warmup-transfers "$WARMUP_TX"
  echo -e "\033[0m"
}

for i in $(seq "$ITERATIONS"); do
  run_jstz "$i"
done

collect

# This loads the profile of the last run
if [ -n "$PROFILING_WRAPPER" ]; then
  echo "[INFO]: collecting results"
  samply load $SAMPLY_OUT
fi

cd "$CURR"
