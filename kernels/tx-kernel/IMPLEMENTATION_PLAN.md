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

- deploy a simple ERC-20 contract
- mint/initialize balances deterministically
- generate `transfer` calls as EIP-1559 txs
- report TPS and apply/success count

## Commit plan

The work should be split into self-contained commits so it is easy to review and to bisect.

### Commit 1: Planning and scaffolding

**Scope**
- add this implementation plan
- add TODO markers / module stubs if useful
- no functional behavior change

**Expected result**
- repository contains a stable plan to return to

**Suggested `jj` flow**
- update this file
- `jj describe -m "tx-kernel: add Ethereum/REVM implementation plan"`

---

### Commit 2: Extend crypto traits for async sender recovery

**Scope**
- extend `riscv-tx-kernel` crypto traits with async secp256k1 recovery
- update kernel-side SBI wrappers in `src/main.rs`
- update `src/sbi_crypto.rs` host call bindings
- update native benchmark crypto implementation to support the new async recovery queue

**Expected result**
- host and native code can enqueue/dequeue secp256k1 recovery jobs
- no REVM/world-state integration yet

**Files likely touched**
- `kernels/tx-kernel/src/lib.rs`
- `kernels/tx-kernel/src/main.rs`
- `kernels/tx-kernel/src/sbi_crypto.rs`
- `kernels/tx-kernel/inbox-generator/src/main.rs`

**Validation**
- host build passes
- native benchmark path compiles

**Suggested `jj` flow**
- `jj new`
- implement recovery API end-to-end
- `jj describe -m "tx-kernel: add async secp256k1 recovery support"`

---

### Commit 3: Introduce Ethereum world-state keyspace model

**Scope**
- replace the current account model abstraction with Ethereum world-state helpers
- add durable key helpers for nonce/balance/code/storage/meta
- update bootstrap logic to initialize EVM metadata
- preserve durable context loader behavior

**Expected result**
- tx-kernel has a reusable world-state abstraction over `ContextStore`
- old custom `/acct/...` state path is no longer the primary model

**Files likely touched**
- `kernels/tx-kernel/src/lib.rs`
- possibly new modules under `kernels/tx-kernel/src/`

**Validation**
- unit tests for key encoding / storage roundtrips if feasible
- host/native compile still passes

**Suggested `jj` flow**
- `jj new`
- implement world-state storage helpers
- `jj describe -m "tx-kernel: add Ethereum world-state durable storage"`

---

### Commit 4: Add Ethereum transaction/block payload parsing

**Scope**
- define new inner block payload format carrying raw Ethereum tx bytes
- implement parser for sequenced block payloads
- add EIP-1559 decoding
- remove dependency on the old custom transfer tx parser in the execution path

**Expected result**
- kernel can parse blocks containing Ethereum transactions
- sender recovery/execution may still be stubbed or partial

**Files likely touched**
- `kernels/tx-kernel/src/lib.rs`
- possibly new parser/types modules
- `kernels/tx-kernel/inbox-generator/src/main.rs`

**Validation**
- parser unit tests for EIP-1559 transfer and contract-creation txs

**Suggested `jj` flow**
- `jj new`
- add tx/block parsing and serialization support
- `jj describe -m "tx-kernel: parse sequenced EIP-1559 Ethereum transactions"`

---

### Commit 5: Build async Ethereum prevalidation pipeline

**Scope**
- enqueue keccak sighash work for parsed Ethereum txs
- enqueue async secp256k1 recovery
- derive sender addresses
- validate decoded/recovered tx data before execution
- keep block signature verification and block-number checks

**Expected result**
- txs are prevalidated using the same enqueue/dequeue pattern as the current tx-kernel
- sender extraction is standard Ethereum recovery, not non-standard metadata

**Files likely touched**
- `kernels/tx-kernel/src/lib.rs`
- benchmark/native crypto code as needed

**Validation**
- end-to-end native tests for sender recovery and tx validation

**Suggested `jj` flow**
- `jj new`
- implement async prevalidation pipeline
- `jj describe -m "tx-kernel: prevalidate Ethereum txs with async crypto pipeline"`

---

### Commit 6: Introduce REVM durable DB adapter

**Scope**
- add `revm` and supporting dependencies
- implement REVM database access over `ContextStore`
- map account/code/storage reads and writes to the durable keyspace
- define block env / tx env builders

**Expected result**
- a reusable REVM adapter exists and compiles for host + riscv
- execution plumbing may still be minimally integrated

**Files likely touched**
- `kernels/tx-kernel/Cargo.toml`
- new REVM/db modules under `kernels/tx-kernel/src/`
- possibly inbox-generator dependencies for signing/encoding

**Validation**
- host compile
- riscv compile if possible early

**Suggested `jj` flow**
- `jj new`
- add REVM and DB adapter
- `jj describe -m "tx-kernel: add REVM durable-storage backend"`

---

### Commit 7: Execute Ethereum txs through REVM

**Scope**
- wire validated txs into REVM execution
- commit account/code/storage changes back to durable storage
- support:
  - ETH transfers
  - contract creation
  - contract calls
- preserve block finalization logging

**Expected result**
- kernel executes Ethereum txs end-to-end in native mode

**Validation**
- native end-to-end test: ETH transfer
- native end-to-end test: contract creation
- native end-to-end test: ERC-20 transfer call

**Suggested `jj` flow**
- `jj new`
- connect pipeline to REVM execution
- `jj describe -m "tx-kernel: execute Ethereum transactions with REVM"`

---

### Commit 8: Migrate benchmark generator to Ethereum ETH-transfer scenario

**Scope**
- update context preparation for world state
- generate EIP-1559 transfer txs
- preserve native and sandbox benchmark flows
- keep benchmark output parsing compatible or update it consistently

**Expected result**
- ETH-transfer TPS benchmark works in native mode and sandbox mode

**Files likely touched**
- `kernels/tx-kernel/inbox-generator/src/main.rs`
- `kernels/tx-kernel/README.md`
- `kernels/tx-kernel/Makefile` if needed

**Validation**
- prepare context
- run native ETH transfer benchmark
- run sandbox ETH transfer benchmark

**Suggested `jj` flow**
- `jj new`
- migrate benchmark generator for ETH transfers
- `jj describe -m "tx-kernel-bench: add EIP-1559 ETH transfer benchmark"`

---

### Commit 9: Add ERC-20 benchmark scenario

**Scope**
- add ERC-20 deployment/call scenario
- add deterministic contract bytecode / ABI call generation
- optionally predeploy during context preparation or deploy in the first benchmark block

**Expected result**
- ERC-20 TPS benchmark works in native mode and sandbox mode

**Validation**
- native ERC-20 benchmark
- sandbox ERC-20 benchmark

**Suggested `jj` flow**
- `jj new`
- add ERC-20 scenario
- `jj describe -m "tx-kernel-bench: add ERC-20 benchmark scenario"`

---

### Commit 10: Docs, smoke tests, and cleanup

**Scope**
- update README with new usage and scenarios
- add/adjust tests
- remove dead code from old custom tx path
- clean up logs and benchmark parsing

**Expected result**
- final user-facing workflow documented
- acceptance criteria covered

**Validation**
- host + riscv build
- native + sandbox smoke benchmarks
- transfer/deploy/call tests

**Suggested `jj` flow**
- `jj new`
- cleanup and document
- `jj describe -m "tx-kernel: document Ethereum/REVM workflow and finalize migration"`

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

- [ ] standard Ethereum EIP-1559 transactions accepted by tx-kernel
- [ ] contract creation supported
- [ ] contract calls supported
- [ ] sender recovery uses async enqueue/dequeue host crypto path
- [ ] durable world-state stored in keyspace
- [ ] REVM executes txs against durable storage
- [ ] host/native benchmark path works
- [ ] RISC-V/sandbox benchmark path works
- [ ] ETH transfer TPS benchmark works
- [ ] ERC-20 TPS benchmark works
- [ ] README updated
- [ ] tests/smoke coverage added

## First implementation step to resume from later

When returning to this work, start with:

1. inspect current crypto traits and SBI bindings
2. add async secp256k1 recovery support end-to-end
3. only after that, begin Ethereum tx parsing and REVM integration
