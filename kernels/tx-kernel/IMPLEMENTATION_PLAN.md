# tx-kernel Ethereum/REVM implementation plan

## Goals

Extend `kernels/tx-kernel` to execute Ethereum transactions using a simple REVM-backed EVM engine while preserving the kernel's existing performance-oriented crypto model:

- use standard Ethereum transactions
- support **EIP-1559** transactions
- support **contract creation**
- support **ETH transfer TPS** benchmark
- support **ERC-20 TPS** benchmark
- keep **host + RISC-V** builds working
- preserve the existing **enqueue/dequeue parallel crypto pipeline**
- align with the kernel's existing **host crypto functions**

## Non-goals

For the first implementation pass, we do **not** need full Etherlink compatibility or all Ethereum transaction types. The initial supported set should be:

- EIP-1559 transactions
- value transfers
- contract creation
- contract calls needed for ERC-20 benchmarking

Legacy/EIP-2930 support can be added later if useful.

## Key architectural decisions

### 1. Preserve outer block/chunk ingestion

Keep the current tx-kernel outer ingestion model:

- block chunks (`TXC1`)
- block blueprints / block payloads
- sequencer-signed blocks
- durable head tracking

This minimizes churn in sandbox integration and benchmark orchestration.

### 2. Replace inner transaction format with Ethereum transactions

The current custom transaction payloads will be replaced by raw Ethereum transaction bytes, initially EIP-1559 only.

Each sequenced block will contain:

- block metadata needed by the kernel
- raw encoded Ethereum transactions
- sequencer signature over the block payload/hash

### 3. Preserve async crypto pipeline

The kernel currently overlaps work by:

- enqueuing keccak
- enqueuing secp256k1 verification
- doing synchronous work
- dequeuing later

Ethereum transaction handling must keep that model.

### 4. Add async secp256k1 recovery host support

Standard Ethereum transactions require sender recovery from signature + sighash.

The current async API only supports verification. We will extend the crypto interface and corresponding host/SBI plumbing with an async recovery path:

- `secp256k1_recover_enqueue(signature, recovery_id, message_hash)`
- `secp256k1_recover_dequeue() -> recovered public key`

This is required to support standard Ethereum transactions without non-standard sender metadata.

### 5. Use a durable-storage-backed REVM database

A custom REVM DB adapter will back account/code/storage reads and writes with `ContextStore`, using a structured Ethereum world-state keyspace.

## Planned durable-storage layout

Tentative layout under the tx-kernel context:

- `/evm/meta/bootstrapped`
- `/evm/meta/head`
- `/evm/meta/chain_id`
- `/evm/meta/base_fee`
- `/evm/meta/block_gas_limit`
- `/evm/meta/spec_id`
- `/evm/accounts/<20-byte-address>/nonce`
- `/evm/accounts/<20-byte-address>/balance`
- `/evm/accounts/<20-byte-address>/code`
- `/evm/accounts/<20-byte-address>/code_hash`
- `/evm/accounts/<20-byte-address>/storage/<32-byte-slot>`

Notes:

- account fields should be stored independently to avoid rewriting large blobs
- storage slots should be individually addressable
- code should be stored as raw bytecode
- slot values should use a fixed 32-byte representation

## Transaction processing pipeline

Planned high-level block execution pipeline:

1. parse outer block payload
2. parse Ethereum tx bytes
3. enqueue all per-tx keccak work needed for sighashes
4. verify block signature synchronously while crypto work runs
5. dequeue tx sighashes
6. enqueue async secp256k1 recovery for each tx
7. dequeue recovered pubkeys
8. derive sender addresses from recovered pubkeys using async/sync keccak as appropriate
9. build validated tx environments for REVM
10. execute transactions sequentially in REVM against the durable DB
11. commit resulting state and advance block head
12. log receipts / applied counts for benchmark parsing

## Environment model

Initial EVM environment should be close to real execution but intentionally minimal:

- fixed `chain_id`
- fixed/fake `coinbase`
- monotonic block number from durable state
- deterministic timestamp progression
- configured block gas limit
- configured base fee
- fixed REVM spec id
- contract creation and contract calls enabled
- logs and receipts recorded at least enough for benchmark/result validation

## Benchmark plan

The benchmark tool in `kernels/tx-kernel/inbox-generator` will be extended with scenarios:

### Scenario A: ETH transfers

- prepopulate funded EOAs
- generate EIP-1559 transfer txs
- sequential nonces per sender
- report TPS and apply/success count

### Scenario B: ERC-20

- deploy or bootstrap a simple ERC-20 benchmark contract
- mint/initialize balances deterministically
- generate `transfer` calls as EIP-1559 txs
- report TPS and apply/success count

## Progress notes

### Completed (commits 1–9)

All planned commits through commit 9 are done and smoke-tested:

```bash
cargo run --manifest-path kernels/tx-kernel/inbox-generator/Cargo.toml -- \
  prepare-context --scenario erc20 --durable-storage-dir /tmp/tx-kernel-erc20-smoke --accounts 10 --rebuild

cargo run --manifest-path kernels/tx-kernel/inbox-generator/Cargo.toml -- \
  benchmark --native --scenario erc20 --transactions 10 --block-frequency 4 --accounts 10 \
  --durable-storage-dir /tmp/tx-kernel-erc20-smoke
```

Result: 11/11 applied. Native ERC-20 benchmark path works end-to-end.

### Known caveat

The current ERC-20 benchmark bootstraps the contract directly into durable state during `prepare-context`. It does **not** yet deploy via a transaction in the inbox. This means:

- contract-call benchmark plumbing is working
- durable-state bootstrap and scenario switching work
- native smoke path works

But the "deploy contract in the inbox, then benchmark calls" flow is not yet exercised.

### Remaining work (commit 10)

Commit 10 should complete the following before closing out the plan:

1. Add focused end-to-end tests for contract creation + call behavior.
2. Switch the ERC-20 scenario from prebootstrapped code to an actual deployment tx in the inbox.
3. Validate native + sandbox flows after the deploy-in-inbox change.

## Commit plan

The work should be split into self-contained commits so it is easy to review and to bisect.

### ~~Commit 1: Planning and scaffolding~~ ✓ DONE

`tx-kernel: add Ethereum/REVM implementation plan`

---

### ~~Commit 2: Extend crypto traits for async sender recovery~~ ✓ DONE

`tx-kernel: add async secp256k1 recovery support`

---

### ~~Commit 3: Introduce Ethereum world-state keyspace model~~ ✓ DONE

`tx-kernel: add Ethereum world-state durable storage`

---

### ~~Commit 4: Add Ethereum transaction/block payload parsing~~ ✓ DONE

`tx-kernel: parse sequenced EIP-1559 Ethereum transactions`

---

### ~~Commit 5: Build async Ethereum prevalidation pipeline~~ ✓ DONE

`tx-kernel: prevalidate Ethereum txs with async crypto pipeline`

---

### ~~Commit 6: Introduce REVM durable DB adapter~~ ✓ DONE

`tx-kernel: add REVM durable-storage backend`

---

### ~~Commit 7: Execute Ethereum txs through REVM~~ ✓ DONE

`tx-kernel: execute Ethereum transactions with REVM`

---

### ~~Commit 8: Migrate benchmark generator to Ethereum ETH-transfer scenario~~ ✓ DONE

`tx-kernel-bench: add EIP-1559 ETH transfer benchmark`

---

### ~~Commit 9: Add ERC-20 benchmark scenario~~ ✓ DONE (bootstrap path)

`tx-kernel-bench: add ERC-20 benchmark scenario`

Native ERC-20 benchmark smoke-tested end-to-end (11/11 applied). Current implementation bootstraps
the contract into durable state rather than deploying via an inbox transaction.

---

### Commit 10: End-to-end tests, deploy-in-inbox ERC-20, and cleanup

**Scope**
- add focused end-to-end tests for contract creation + call behavior
- switch ERC-20 scenario from prebootstrapped contract to a real deployment tx in the inbox
- validate native + sandbox flows after deploy-in-inbox change
- update README with final usage and scenarios
- remove any remaining dead code from the old custom tx path
- clean up logs and benchmark parsing as needed

**Expected result**
- ERC-20 scenario exercises the full deploy → call path through the inbox
- contract creation is covered by tests, not just benchmarks
- acceptance checklist fully satisfied

**Validation**
- end-to-end test: contract creation via inbox tx
- end-to-end test: ERC-20 deploy + transfer via inbox
- native ERC-20 benchmark with deploy-in-inbox flow
- sandbox ERC-20 benchmark
- host + riscv build

**Suggested `jj` flow**
- `jj new`
- add e2e tests, switch ERC-20 to deploy-in-inbox, validate sandbox
- `jj describe -m "tx-kernel: e2e tests and ERC-20 deploy-in-inbox flow"`

## Recommended `jj` workflow during implementation

For each self-contained step:

1. create a fresh change
   - `jj new`
2. implement one logical slice only
3. run the smallest relevant validation
4. describe the change clearly
   - `jj describe -m "..."`
5. if the slice is ready to publish, create the next child change
   - `jj new`
6. continue stacking changes on top

Useful commands during the stack build:

- show current stack
  - `jj log`
- amend current change by editing files and then
  - `jj describe -m "updated message"` if needed
- split work if a change gets too large
  - `jj split`
- reorder/fix dependencies if needed
  - `jj rebase`
- inspect current diff
  - `jj diff`

When ready to publish stacked work, use the repo's normal `jj`/git export flow.

## Acceptance checklist

The final stack should satisfy all of the following:

- [x] standard Ethereum EIP-1559 transactions accepted by tx-kernel
- [x] contract creation supported
- [x] contract calls supported
- [x] sender recovery uses async enqueue/dequeue host crypto path
- [x] durable world-state stored in keyspace
- [x] REVM executes txs against durable storage
- [x] host/native benchmark path works
- [ ] RISC-V/sandbox benchmark path works
- [x] ETH transfer TPS benchmark works
- [x] ERC-20 TPS benchmark works (bootstrap path; deploy-in-inbox pending)
- [ ] ERC-20 scenario uses deploy-in-inbox flow
- [ ] README updated
- [ ] tests/smoke coverage added for contract creation + call

## Next step to resume from

Commits 1–9 are complete. When returning to this work, go straight to commit 10:

1. Add an integration test that sends a contract-creation tx through the inbox and asserts the contract is callable afterward.
2. Add an integration test that deploys the ERC-20 contract via an inbox tx and then calls `transfer`.
3. Update the ERC-20 `prepare-context` path to emit a deployment tx instead of bootstrapping durable state directly.
4. Verify sandbox (`--no-native`) still passes for both scenarios.
5. Update README with final CLI usage.
