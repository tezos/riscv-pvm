#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
# SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
#
# SPDX-License-Identifier: MIT

set -e

USAGE="Usage: [ -s: static inbox ] [ -n: run natively ]"

TX="15"
SANDBOX_BIN="riscv-sandbox"
DEFAULT_ROLLUP_ADDRESS="sr163Lv22CdE8QagCwf48PWDTquk6isQwv57"
ETHERLINK_SANDBOX_PARAMS=("--input" "etherlink/target/riscv64gc-unknown-linux-musl/release/etherlink")
INBOX_FILE="assets/etherlink-erc20-inbox.json"
STATIC_INBOX=""
NATIVE=""

CURR=$(pwd)
RISCV_DIR=$(dirname "$0")/..
cd "$RISCV_DIR/src/riscv"

while getopts "sn" OPTION; do
  case "$OPTION" in
  s)
    STATIC_INBOX="y"
    ;;
  n)
    NATIVE=$(make --silent -C jstz print-native-target | grep -wv make)
    ;;
  *)
    echo "$USAGE"
    exit 1
    ;;
  esac
done

if [ -n "$NATIVE" ] && [ -z "$STATIC_INBOX" ]; then
  echo "Native compilation without static inbox unsupported"
  echo "$USAGE"
  exit 1
fi

##########
# RISC-V #
##########
build_etherlink_riscv() {
  if [ "$STATIC_INBOX" = "y" ]; then
    INBOX_FILE="../../../$INBOX_FILE" make -C etherlink build-kernel-static &> /dev/null
  else
    make -C etherlink build-kernel &> /dev/null
  fi
}

run_etherlink_riscv() {
  LOG="$DATA_DIR/etherlink.log"
  "./$SANDBOX_BIN" run \
    "${ETHERLINK_SANDBOX_PARAMS[@]}" \
    --inbox-file "$INBOX_FILE" \
    --address "$DEFAULT_ROLLUP_ADDRESS" \
    --timings > "$LOG"
}

##########
# Native #
##########
build_etherlink_native() {
  INBOX_FILE="../../../$INBOX_FILE" make -C etherlink build-kernel-native
}

run_etherlink_native() {
  LOG="$DATA_DIR/etherlink.log"
  ./etherlink/target/"$NATIVE"/release/etherlink \
    --timings > "$LOG" 2> /dev/null
}

echo "[INFO]: Building RISC-V sandbox"
make "$SANDBOX_BIN" &> /dev/null
echo "[INFO]: Building bench tool"
make -C etherlink inbox-bench &> /dev/null

#########
# Build #
#########
echo "[INFO]: Building Etherlink kernel"

if [ -z "$NATIVE" ]; then
  build_etherlink_riscv
else
  build_etherlink_native
fi

#######
# Run #
#######
run_etherlink() {
  if [ -z "$NATIVE" ]; then
    echo "[INFO]: running $TX transfers (riscv) "
    run_etherlink_riscv
  else
    echo "[INFO]: running $TX transfers ($NATIVE) "
    run_etherlink_native
  fi
}

DATA_DIR=${DATA_DIR:=$(mktemp -d)}
run_etherlink

echo -e "\033[1m"
./etherlink/inbox-bench results \
  --inbox-file "$INBOX_FILE" \
  --log-file "$LOG" \
  --expected-transfers "$TX"
echo -e "\033[0m"

cd "$CURR"
