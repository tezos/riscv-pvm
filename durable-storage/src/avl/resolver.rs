// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Interface for resolving data from IDs of [`Tree`] and [`Node`] objects.
//!
//! [`Tree`]: crate::avl::tree::Tree
//! [`Node`]: crate::avl::node::Node

use std::sync::Arc;

/// Trait for resolving identifiers to values.
pub trait Resolver<Id, Value> {
    /// Resolve an identifier to a value.
    fn resolve<'a>(&self, id: &'a Id) -> &'a Value;

    /// Resolve an identifier to a mutable value.
    fn resolve_mut<'a>(&mut self, id: &'a mut Id) -> &'a mut Value;
}

/// Provide values identified by an [`Arc`].
#[derive(Clone, Debug)]
pub struct ArcResolver;

impl<T: Clone> Resolver<Arc<T>, T> for ArcResolver {
    fn resolve<'a>(&self, id: &'a Arc<T>) -> &'a T {
        id.as_ref()
    }

    fn resolve_mut<'a>(&mut self, id: &'a mut Arc<T>) -> &'a mut T {
        Arc::make_mut(id)
    }
}
