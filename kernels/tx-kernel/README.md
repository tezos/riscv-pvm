# tx-kernel

`tx-kernel` is a simple Ethereum transaction kernel backed by durable storage.

The quickest workflow is:
1. create a persistent RocksDB context once
2. run the benchmark against that context in `riscv-sandbox`
3. run the same benchmark natively on the host for comparison

The default scenario is `erc20`. It deploys a benchmark contract via a CREATE transaction in the
first inbox block, then drives it with EIP-1559 contract calls. This exercises the full
deploy → call path through the kernel.

## Quick Start (ERC-20, RISC-V)

Create a new persistent database with a chosen number of accounts:

```bash
make -C kernels/tx-kernel prepare-context \
  PREPOPULATED_ACCOUNTS=1024 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-db
```

This rebuilds `/tmp/tx-kernel-db` from scratch and prepopulates it with `1024` funded EOAs.

Run the ERC-20 benchmark in `riscv-sandbox`:

```bash
make -C kernels/tx-kernel benchmark \
  TRANSACTIONS=1000 \
  BLOCK_FREQUENCY=100 \
  PREPOPULATED_ACCOUNTS=1024 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-db
```

Run the same benchmark natively on the host for comparison:

```bash
make -C kernels/tx-kernel benchmark-native \
  TRANSACTIONS=1000 \
  BLOCK_FREQUENCY=100 \
  PREPOPULATED_ACCOUNTS=1024 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-db
```

## ETH Transfer Benchmark

To run the simpler ETH transfer scenario instead, pass `BENCHMARK_SCENARIO=eth-transfer` to every
`make` invocation:

```bash
make -C kernels/tx-kernel prepare-context \
  BENCHMARK_SCENARIO=eth-transfer \
  PREPOPULATED_ACCOUNTS=1024 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-eth-db

make -C kernels/tx-kernel benchmark \
  BENCHMARK_SCENARIO=eth-transfer \
  TRANSACTIONS=1000 \
  BLOCK_FREQUENCY=100 \
  PREPOPULATED_ACCOUNTS=1024 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-eth-db

make -C kernels/tx-kernel benchmark-native \
  BENCHMARK_SCENARIO=eth-transfer \
  TRANSACTIONS=1000 \
  BLOCK_FREQUENCY=100 \
  PREPOPULATED_ACCOUNTS=1024 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-eth-db
```

## What The Parameters Mean

- `PREPOPULATED_ACCOUNTS`
  Number of funded EOAs inserted when the persistent database is created.

- `DURABLE_STORAGE_DIR`
  Directory containing the persistent RocksDB durable-storage state.

- `TRANSACTIONS`
  Total number of benchmark transactions in the selected scenario.
  - `erc20`: total number of ERC-20 `transfer` calls. The first block also contains a
    contract-creation transaction (CREATE) and a mint call, so the actual transaction count
    is `TRANSACTIONS + 2`.
  - `eth-transfer`: total number of EIP-1559 ETH transfers.

- `BENCHMARK_SCENARIO`
  Benchmark workload to generate.
  - `erc20` (default)
  - `eth-transfer`

- `BLOCK_FREQUENCY`
  Number of transactions per block.
  For example, `TRANSACTIONS=1000` and `BLOCK_FREQUENCY=100` means `10` blocks of `100` transactions each.

## Recommended Smoke Test

Use a small database first to verify the full ERC-20 path (deploy + call) in both RISC-V and native:

```bash
make -C kernels/tx-kernel prepare-context \
  PREPOPULATED_ACCOUNTS=100 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-smoke-db

make -C kernels/tx-kernel benchmark \
  TRANSACTIONS=100 \
  BLOCK_FREQUENCY=50 \
  PREPOPULATED_ACCOUNTS=100 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-smoke-db

make -C kernels/tx-kernel benchmark-native \
  TRANSACTIONS=100 \
  BLOCK_FREQUENCY=50 \
  PREPOPULATED_ACCOUNTS=100 \
  DURABLE_STORAGE_DIR=/tmp/tx-kernel-smoke-db
```

The RISC-V run reports a state root at the end of each block. The native run should produce
identical state roots, confirming deterministic execution across both environments.

## Measuring Context Size Impact

To compare the effect of context size, repeat the same benchmark with different values of
`PREPOPULATED_ACCOUNTS`, for example:

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
