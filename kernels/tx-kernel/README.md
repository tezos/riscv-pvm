# tx-kernel

`tx-kernel` is a simple payment-chain kernel backed by durable storage.

It comes with:
- a RISC-V kernel binary in this crate
- a native helper in [`inbox-generator`](./inbox-generator) for preparing context, generating inbox files, and running benchmarks

## Build

From the repo root:

```bash
make -C kernels/tx-kernel build
```

This builds:
- the RISC-V kernel ELF
- the native benchmark / inbox-generation helper

## Generate An Inbox File

To generate an `inbox.json` file without running the benchmark:

```bash
cargo run --manifest-path kernels/tx-kernel/inbox-generator/Cargo.toml -- \
  generate \
  --transactions 1000 \
  --block-frequency 100 \
  --accounts 1024 \
  --inbox-file /tmp/tx-kernel-inbox.json
```

Parameters:
- `--transactions`: total number of transactions to generate
- `--block-frequency`: number of transactions per block
- `--accounts`: number of deterministic benchmark accounts to draw transactions from
- `--inbox-file`: path to the generated sandbox inbox file

Example:
- `--transactions 1000 --block-frequency 100` generates `10` blocks of `100` transactions each

## Prepare A Persistent Durable-Storage Context

To create a persistent RocksDB durable-storage context with a given number of accounts:

```bash
make -C kernels/tx-kernel prepare-context \
  PREPOPULATED_ACCOUNTS=1024 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-db
```

This rebuilds the durable-storage directory and prepopulates it with `PREPOPULATED_ACCOUNTS` accounts.

Equivalent direct command:

```bash
cargo run --manifest-path kernels/tx-kernel/inbox-generator/Cargo.toml -- \
  prepare-context \
  --durable-storage-dir /tmp/tx-kernel-db \
  --accounts 1024 \
  --rebuild
```

Parameters:
- `PREPOPULATED_ACCOUNTS`: number of accounts to populate in durable storage
- `DURABLE_STORAGE_DIR`: path to the persistent RocksDB-backed durable-storage directory

Notes:
- the context includes account balances and metadata needed by the kernel
- `prepare-context` always passes `--rebuild`, so an existing directory is reset first

## Run The Benchmark

To benchmark the kernel in `riscv-sandbox`:

```bash
make -C kernels/tx-kernel benchmark \
  TRANSACTIONS=1000 \
  BLOCK_FREQUENCY=100 \
  PREPOPULATED_ACCOUNTS=1024 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-db \
  INBOX_FILE=/tmp/tx-kernel-inbox.json
```

Equivalent direct command:

```bash
cargo run --manifest-path kernels/tx-kernel/inbox-generator/Cargo.toml -- \
  benchmark \
  --transactions 1000 \
  --block-frequency 100 \
  --accounts 1024 \
  --durable-storage-dir /tmp/tx-kernel-db \
  --inbox-file /tmp/tx-kernel-inbox.json
```

Parameters:
- `TRANSACTIONS`: total number of transactions to execute
- `BLOCK_FREQUENCY`: number of transactions per block
- `PREPOPULATED_ACCOUNTS`: number of accounts expected in the prepared context
- `DURABLE_STORAGE_DIR`: durable-storage directory to reuse for the run
- `INBOX_FILE`: path where the generated inbox file is written

What the benchmark does:
- generates signed block blueprints
- writes the inbox file
- runs `riscv-sandbox` with the `tx-kernel` ELF and the durable-storage directory
- parses sandbox timing logs
- reports processed blocks, transactions, and TPS

## Smoke Test

Small end-to-end run:

```bash
make -C kernels/tx-kernel prepare-context \
  PREPOPULATED_ACCOUNTS=100 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-smoke-db

make -C kernels/tx-kernel benchmark \
  TRANSACTIONS=1000 \
  BLOCK_FREQUENCY=100 \
  PREPOPULATED_ACCOUNTS=100 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-smoke-db \
  INBOX_FILE=/tmp/tx-kernel-smoke-inbox.json
```

## Measuring Context Size Impact

To compare the effect of different durable-storage context sizes, repeat:

```bash
make -C kernels/tx-kernel prepare-context \
  PREPOPULATED_ACCOUNTS=<N> \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-db-<N>

make -C kernels/tx-kernel benchmark \
  TRANSACTIONS=10000 \
  BLOCK_FREQUENCY=100 \
  PREPOPULATED_ACCOUNTS=<N> \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-db-<N> \
  INBOX_FILE=/tmp/tx-kernel-<N>.json
```

Typical values for `<N>`:
- `100`
- `1024`
- `100000`
- `1000000`

This keeps the transaction workload fixed while varying the initial durable-storage context size.
