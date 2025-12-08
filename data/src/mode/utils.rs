// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Utilities for modal components

use std::ops::Deref;

/// Source for a state component, either borrowed or owned
///
/// We use this type to store the source version of the underlying proof data. Merkle tree proofs
/// usually encode the initial state. Hence we need to figure out what parts of the initial state
/// need to go into the proof.
///
/// Normally the source is borrowed from state in [`Normal`] mode. In those cases, it is enough to
/// obtain a reference to the source.
///
/// However, there are cases when no source can be borrowed, because it is owned. During the
/// proof generation, we might create a new state. This can happen when a dynamic state
/// component holds states. E.g. it can grow and therefore create new states. During proof
/// generation we don't differentiate between modes, hence you can't create a state in [`Normal`]
/// mode and then borrow it.
///
/// Wherever newly created states are allowed, the upstream state which allows this to happen is in
/// charge of preventing those states from being included in the Merkle tree proof. In the example
/// above, that would be the growable state component. It needs to differentiate between states that
/// existed before the proof generation started, and those that were created during proof.
///
/// It is possible to override states when we allow state creation. Overriding isn't the same as
/// writing the underlying values from one state to the other as it would override the state
/// that is specific to the mode. This would be caught by producing an invalid Merkle tree proof in
/// our test suite. In such cases, the claimed initial state hash of the Merkle tree would not match
/// that of the actual initial state in [`Normal`] mode.
///
/// Another benefit of allowing states to be created from an owned source is that it simplifies our
/// test suite where we want to instantiate a state using the [`Prove`] mode without a previous
/// state in [`Normal`] mode. This simplifies the API for tests and allows us to write one test and
/// test it against 3 different backends.
///
/// It is possible to simplify this type by only allowing owned states. This has some downsides,
/// as it would require all [`Normal`] states to be cloned when wrapping them into [`Prove`]. This
/// could be costly and unnecessary if the states isn't even used.
///
/// [`Normal`]: crate::mode::Normal
/// [`Prove`]: crate::mode::Prove
#[derive(Debug, Clone, derive_more::From)]
pub enum Source<'a, T> {
    Borrowed(#[from] &'a T),
    Owned(#[from] Box<T>),
}

impl<T> Deref for Source<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Source::Borrowed(value) => value,
            Source::Owned(value) => value.as_ref(),
        }
    }
}

impl<T> From<T> for Source<'_, T> {
    fn from(value: T) -> Self {
        Source::from(Box::new(value))
    }
}
