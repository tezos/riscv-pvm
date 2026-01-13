# Rust Dependency Update Notes

## Summary

This document describes the Rust dependency updates made to the workspace `Cargo.toml` file. The updates have been applied to the version specifications, but **the `Cargo.lock` file needs to be regenerated** with network access.

## Updated Dependencies

The following dependencies have been updated to their latest stable versions:

### Major Updates (Breaking Changes Possible)
- **rocksdb**: `0.24.0` → `0.43.0`
  - This is a significant version jump with breaking changes
  - Requires MSRV 1.81.0 (current project uses 1.88.0, so compatible)
  - Changes include column family concurrency, compression features, and API updates
  - See: https://github.com/rust-rocksdb/rust-rocksdb/releases for detailed changelog

### Minor Updates
- **cranelift**: `0.120.2` → `0.127.1`
- **cranelift-jit**: `0.120.1` → `0.127.1`
- **cranelift-module**: `0.120.1` → `0.127.1`
- **cranelift-native**: `0.120.2` → `0.127.1`
- **tokio**: `1` → `1.49` (more specific version)

### Patch Updates
- **anyhow**: `1.0` → `1.0.100`
- **bincode**: `2.0` → `2.0.1` (last stable version before unmaintained 3.0)
- **paste**: `1.0.14` → `1.0.15`
- **tracing**: `0.1.41` → `0.1.44`

## Already Up-to-Date Dependencies

The following dependencies were verified to be at their latest stable versions:
- blake3 (1.8.3)
- bytes (1.11.0)
- capstone (0.14)
- cfg-if (1.0.4)
- clap (4.5.54)
- comfy-table (7.2.2)
- criterion (0.8.1)
- derive_more (2.1.1)
- ed25519-dalek (2.2.0)
- elf (0.8.0)
- goldenfile (1.9.1)
- itertools (0.14.0)
- lazy_static (1.5.0)
- libsecp256k1 (0.7.2)
- log (0.4.29)
- memmap2 (0.9.9)
- num_enum (0.7.5)
- proptest (1.9.0)
- quanta (0.12.6)
- range-collections (0.4.6)
- rangemap (1.7.1)
- rustc_apfloat (0.2.3)
- serde (1.0.228)
- serde_json (1.0.149)
- sha2 (0.10.9)
- sha3 (0.10.8)
- strum (0.27.2)
- tempfile (3.24.0)
- thiserror (2.0.17)
- tracing-subscriber (0.3.22)
- zerocopy (0.8.33)

## Next Steps

To complete the dependency update process:

1. **Regenerate Cargo.lock** (requires network access):
   ```bash
   cargo update
   ```

2. **Build the project**:
   ```bash
   cargo build --workspace
   ```

3. **Run tests**:
   ```bash
   cargo test --workspace
   ```

4. **Address any issues**:
   - Pay special attention to rocksdb changes if you use durable-storage
   - Check for any API changes in cranelift if you use JIT compilation
   - Review compiler warnings for deprecated APIs

5. **Commit the updated Cargo.lock**:
   ```bash
   git add Cargo.lock
   git commit -m "Update Cargo.lock after dependency updates"
   ```

## Special Notes

### bincode
The bincode crate has been marked as **unmaintained** as of version 3.0. We're staying on version 2.0.1, which is the last functional version. Consider migrating to alternatives like:
- wincode (bincode-compatible)
- postcard
- bitcode
- rkyv

### rocksdb
The rocksdb update from 0.24.0 to 0.43.0 includes several breaking changes:
- Multi-threaded column family operations require the `multi-threaded-cf` feature
- Some deprecated options have been removed
- MSRV bumped to 1.81.0

### lazy_static
While lazy_static (1.5.0) is still maintained, consider using Rust's built-in alternatives for new code:
- `OnceCell` (since Rust 1.70)
- `LazyLock` (since Rust 1.77)

## Dependencies Not Updated

Some dependencies were not updated because:
- **rand**: Latest is 0.10.0-rc.6 (release candidate, not stable)
- **sha2/sha3**: Latest are 0.11.0-rc.3 (release candidates)

These will be updated once stable versions are released.
