# evm-poc

`evm-poc` is a simple Ethereum transaction kernel backed by durable storage.

The quickest workflow is:
1. create a persistent RocksDB context once
2. run the benchmark against that context in `riscv-sandbox`
3. run the same benchmark natively on the host for comparison

The default scenario is `eth-transfer`.

The `erc20` scenario deploys a benchmark contract via a CREATE transaction in the first inbox
block, then drives it with EIP-1559 contract calls. This exercises the full deploy → call path
through the kernel.

## Quick Start (ERC-20, RISC-V)

Create a new persistent database with a chosen number of accounts:

```bash
make -C kernels/evm-poc prepare-context \
  BENCHMARK_SCENARIO=erc20 \
  NUMBER_OF_ACCOUNTS=1024 \
  DURABLE_STORAGE_DIR=/tmp/evm-poc-db
```

This rebuilds `/tmp/evm-poc-db` from scratch and prepopulates it with `1024` funded EOAs.

Run the ERC-20 benchmark in `riscv-sandbox`:

```bash
make -C kernels/evm-poc benchmark \
  BENCHMARK_SCENARIO=erc20 \
  TRANSACTIONS=1000 \
  BLOCK_FREQUENCY=100 \
  NUMBER_OF_ACCOUNTS=1024 \
  DURABLE_STORAGE_DIR=/tmp/evm-poc-db
```

Run the same benchmark natively on the host for comparison:

```bash
make -C kernels/evm-poc benchmark-native \
  BENCHMARK_SCENARIO=erc20 \
  TRANSACTIONS=1000 \
  BLOCK_FREQUENCY=100 \
  NUMBER_OF_ACCOUNTS=1024 \
  DURABLE_STORAGE_DIR=/tmp/evm-poc-db
```

## ETH Transfer Benchmark

`eth-transfer` is the default scenario, so you can run it without setting
`BENCHMARK_SCENARIO`:

```bash
make -C kernels/evm-poc prepare-context \
  NUMBER_OF_ACCOUNTS=1024 \
  DURABLE_STORAGE_DIR=/tmp/evm-poc-eth-db

make -C kernels/evm-poc benchmark \
  TRANSACTIONS=1000 \
  BLOCK_FREQUENCY=100 \
  NUMBER_OF_ACCOUNTS=1024 \
  DURABLE_STORAGE_DIR=/tmp/evm-poc-eth-db

make -C kernels/evm-poc benchmark-native \
  TRANSACTIONS=1000 \
  BLOCK_FREQUENCY=100 \
  NUMBER_OF_ACCOUNTS=1024 \
  DURABLE_STORAGE_DIR=/tmp/evm-poc-eth-db
```

## What The Parameters Mean

- `NUMBER_OF_ACCOUNTS`
  Number of funded EOAs to use. This is the primary Makefile-facing account-count variable and is
  accepted by both `prepare-context` and benchmark targets.

- `PREPOPULATED_ACCOUNTS`
  Backward-compatible alias for `NUMBER_OF_ACCOUNTS`.

- `DURABLE_STORAGE_DIR`
  Directory containing the persistent RocksDB durable-storage state.

- `TRANSACTIONS`
  Total number of benchmark transactions in the selected scenario.
  - `eth-transfer`: total number of EIP-1559 ETH transfers.
  - `erc20`: total number of ERC-20 `transfer` calls. The first block also contains a
    contract-creation transaction (CREATE) and a mint call, so the actual transaction count is
    `TRANSACTIONS + 2`.

- `BENCHMARK_SCENARIO`
  Benchmark workload to generate.
  - `eth-transfer` (default)
  - `erc20`

- `BLOCK_FREQUENCY`
  Number of transactions per block.
  For example, `TRANSACTIONS=1000` and `BLOCK_FREQUENCY=100` means `10` blocks of `100`
  transactions each.

## Recommended Smoke Test

Use a small database first to verify the full ERC-20 path (deploy + call) in both RISC-V and
native:

```bash
make -C kernels/evm-poc prepare-context \
  BENCHMARK_SCENARIO=erc20 \
  NUMBER_OF_ACCOUNTS=100 \
  DURABLE_STORAGE_DIR=/tmp/evm-poc-smoke-db

make -C kernels/evm-poc benchmark \
  BENCHMARK_SCENARIO=erc20 \
  TRANSACTIONS=100 \
  BLOCK_FREQUENCY=50 \
  NUMBER_OF_ACCOUNTS=100 \
  DURABLE_STORAGE_DIR=/tmp/evm-poc-smoke-db

make -C kernels/evm-poc benchmark-native \
  BENCHMARK_SCENARIO=erc20 \
  TRANSACTIONS=100 \
  BLOCK_FREQUENCY=50 \
  NUMBER_OF_ACCOUNTS=100 \
  DURABLE_STORAGE_DIR=/tmp/evm-poc-smoke-db
```

The RISC-V run reports a state root at the end of each block. The native run should produce
identical state roots, confirming deterministic execution across both environments.

## Measuring Context Size Impact

To compare the effect of context size, repeat the same benchmark with different values of
`NUMBER_OF_ACCOUNTS`:

- `100`
- `1024`
- `100000`
- `1000000`

Example:

```bash
make -C kernels/evm-poc prepare-context \
  NUMBER_OF_ACCOUNTS=1000000 \
  DURABLE_STORAGE_DIR=/tmp/evm-poc-db-1m

make -C kernels/evm-poc benchmark \
  TRANSACTIONS=10000 \
  BLOCK_FREQUENCY=100 \
  NUMBER_OF_ACCOUNTS=1000000 \
  DURABLE_STORAGE_DIR=/tmp/evm-poc-db-1m

make -C kernels/evm-poc benchmark-native \
  TRANSACTIONS=10000 \
  BLOCK_FREQUENCY=100 \
  NUMBER_OF_ACCOUNTS=1000000 \
  DURABLE_STORAGE_DIR=/tmp/evm-poc-db-1m
```