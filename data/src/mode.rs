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

/// Operational mode of the PVM
///
/// This trait can be used to instantiate the modal representation for a [`Modal`] template type.
pub trait Mode {
    /// Select the type from the [`Modal`] template that corresponds to the current mode.
    type Select<Template: Modal + ?Sized>;
}

/// Normal execution mode
///
/// The PVM advances its state as quickly as possible in this mode.
pub enum Normal {}

impl Mode for Normal {
    type Select<Template: Modal + ?Sized> = Template::Normal;
}

/// Proof generation mode
///
/// The purpose of this mode is to generate a proof for a single transition of the PVM state. This
/// proof must convince a verifier that the transition was performed correctly according to the
/// PVM semantics.
pub struct Prove<'normal>(PhantomData<&'normal Normal>, Infallible);

impl<'normal> Mode for Prove<'normal> {
    type Select<Template: Modal + ?Sized> = Template::Prove<'normal>;
}

/// Proof verification mode
///
/// In this mode, the PVM verifies a proof that was generated in the [`Prove`] mode.
pub enum Verify {}

impl Mode for Verify {
    type Select<Template: Modal + ?Sized> = Template::Verify;
}

/// Proxy template for modal representations
///
/// Types that need to exhibit different representations depending on the PVM's operational mode
/// can use this trait via a template type to define their mode-specific internal representation.
///
/// # Example
///
/// In the following example, `MyType` needs to have different internal representations. It uses
/// `MyTemplate` as a modal template to define these representations for each mode. The type
/// `M::Select<MyTemplate>` will resolve to the appropriate representation based on the mode `M`.
///
/// This results in `MyType<Normal>` having a `MyNormalType` representation, `MyType<Prove>` having
/// a `MyProveType` representation, and `MyType<Verify>` having a `MyVerifyType` representation.
///
/// ```
/// use octez_riscv_data::mode::*;
///
/// enum MyTemplate {}
///
/// impl Modal for MyTemplate {
///     type Normal = MyNormalType;
///
///     type Prove<'normal> = MyProveType;
///
///     type Verify = MyVerifyType;
/// }
///
/// struct MyNormalType {
///     // ...
/// }
///
/// struct MyProveType {
///     // ...
/// }
///
/// struct MyVerifyType {
///     // ...
/// }
///
/// #[repr(transparent)]
/// struct MyType<M: Mode> {
///     repr: M::Select<MyTemplate>,
/// }
/// ```
pub trait Modal {
    /// Representation in [`Normal`] mode
    type Normal;

    /// Representation in [`Prove`] mode
    type Prove<'normal>;

    /// Representation in [`Verify`] mode
    type Verify;
}
