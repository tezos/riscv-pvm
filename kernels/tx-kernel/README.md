# tx-kernel

`tx-kernel` is a simple Ethereum transaction kernel backed by durable storage.

The quickest workflow is:
1. create a persistent RocksDB context once
2. run the benchmark against that context in `riscv-sandbox`
3. run the same benchmark natively on the host for comparison

## Quick Start

Create a new persistent database with a chosen number of accounts:

```bash
make -C kernels/tx-kernel prepare-context \
  PREPOPULATED_ACCOUNTS=1024 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-db
```

This rebuilds `/tmp/tx-kernel-db` from scratch and prepopulates it with `1024` funded EOAs for the ETH-transfer benchmark.

Run the benchmark in `riscv-sandbox` against that database:

```bash
make -C kernels/tx-kernel benchmark \
  TRANSACTIONS=1000 \
  BLOCK_FREQUENCY=100 \
  PREPOPULATED_ACCOUNTS=1024 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-db
```

Run the same benchmark natively on the host against the same database:

```bash
make -C kernels/tx-kernel benchmark-native \
  TRANSACTIONS=1000 \
  BLOCK_FREQUENCY=100 \
  PREPOPULATED_ACCOUNTS=1024 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-db
```

## What The Parameters Mean

- `PREPOPULATED_ACCOUNTS`
  Number of funded EOAs inserted when the persistent database is created.

- `DURABLE_STORAGE_DIR`
  Directory containing the persistent RocksDB durable-storage state.

- `TRANSACTIONS`
  Total number of EIP-1559 ETH-transfer transactions in the benchmark run.

- `BLOCK_FREQUENCY`
  Number of transactions per block.
  For example, `TRANSACTIONS=1000` and `BLOCK_FREQUENCY=100` means `10` blocks of `100` transactions each.

## Recommended Smoke Test

Use a small database first:

```bash
make -C kernels/tx-kernel prepare-context \
  PREPOPULATED_ACCOUNTS=100 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-smoke-db

make -C kernels/tx-kernel benchmark \
  TRANSACTIONS=1000 \
  BLOCK_FREQUENCY=100 \
  PREPOPULATED_ACCOUNTS=100 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-smoke-db

make -C kernels/tx-kernel benchmark-native \
  TRANSACTIONS=1000 \
  BLOCK_FREQUENCY=100 \
  PREPOPULATED_ACCOUNTS=100 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-smoke-db
```

## Measuring Context Size Impact

To compare the effect of context size, repeat the same benchmark with different values of `PREPOPULATED_ACCOUNTS`, for example:

- `100`
- `1024`
- `100000`
- `1000000`

Example:

```bash
make -C kernels/tx-kernel prepare-context \
  PREPOPULATED_ACCOUNTS=1000000 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-db-1m

make -C kernels/tx-kernel benchmark \
  TRANSACTIONS=10000 \
  BLOCK_FREQUENCY=100 \
  PREPOPULATED_ACCOUNTS=1000000 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-db-1m

make -C kernels/tx-kernel benchmark-native \
  TRANSACTIONS=10000 \
  BLOCK_FREQUENCY=100 \
  PREPOPULATED_ACCOUNTS=1000000 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-db-1m
```
