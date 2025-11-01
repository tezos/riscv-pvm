// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Operational modes of a proof-generating virtual machine (PVM)
//!
//! There are three operational modes for the PVM:
//!
//! - [`Normal`]
//! - [`Prove`]
//! - [`Verify`]

use std::convert::Infallible;
use std::marker::PhantomData;

/// Normal execution mode
///
/// The PVM advances its state as quickly as possible in this mode.
pub enum Normal {}

/// Proof generation mode
///
/// The purpose of this mode is to generate a proof for a single transition of the PVM state. This
/// proof must convince a verifier that the transition was performed correctly according to the
/// PVM semantics.
pub struct Prove<'normal>(PhantomData<&'normal Normal>, Infallible);

/// Proof verification mode
///
/// In this mode, the PVM verifies a proof that was generated in the [`Prove`] mode.
pub enum Verify {}
