// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Implementation of the RISC-V PVM durable storage's persistence layer.
//!
//! A durable storage is responsible for providing a crash-resistant key-value store.
//!
//! The main components of the durable storage are:
//! - **Repository**: Corresponds to a directory on disk, and acts as the storage place for the
//!   commits.
//!
//! - **Database registry**: Keeps track of individual & separate KV databases. The registry
//!   provides create, delete, copy & move operations between databases. The main motivation for
//!   this is to obtain atomicity.
//!
//! - **Cache layer**: Layer providing an in-memory cache for the KV operations. The concern is to
//!   optimise for performance.
//!
//! - **Merkle layer**: Responsible for arranging the data associated with each database in a way
//!   such that obtaining a root hash is possible & efficient.
//!
//! - **Persistence layer**: Responsible for actually persisting the basic get, set, delete
//!   operations on disk.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "Stubbed API - will be implemented in RV-793")
)]

mod persistence_layer;
mod repo;
