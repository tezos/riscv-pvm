#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024-2025 TriliTech <contact@trili.tech>
# SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
#
# SPDX-License-Identifier: MIT

set -e

TX="15"
SANDBOX_BIN="riscv-sandbox"
DEFAULT_ROLLUP_ADDRESS="sr163Lv22CdE8QagCwf48PWDTquk6isQwv57"
ETHERLINK_SANDBOX_PARAMS=("--input" "etherlink/target/riscv64gc-unknown-linux-musl/release/etherlink")
INBOX_FILE="assets/etherlink-erc20-inbox.json"

CURR=$(pwd)
RISCV_DIR=$(dirname "$0")/..
cd "$RISCV_DIR/src/riscv"

echo "[INFO]: Building RISC-V sandbox"
make "$SANDBOX_BIN" &> /dev/null
echo "[INFO]: Building bench tool"
make -C etherlink inbox-bench &> /dev/null
echo "[INFO]: Building Etherlink kernel (riscv)"
make -C etherlink build-kernel &> /dev/null

DATA_DIR=${DATA_DIR:=$(mktemp -d)}
LOG="$DATA_DIR/etherlink.log"

echo "[INFO]: Running $TX transfers (riscv) "
"./$SANDBOX_BIN" run \
  "${ETHERLINK_SANDBOX_PARAMS[@]}" \
  --inbox-file "$INBOX_FILE" \
  --address "$DEFAULT_ROLLUP_ADDRESS" \
  --timings > "$LOG"

echo -e "\033[1m"
./etherlink/inbox-bench results \
  --inbox-file "$INBOX_FILE" \
  --log-file "$LOG" \
  --expected-transfers "$TX"
echo -e "\033[0m"

cd "$CURR"
