// SPDX-FileCopyrightText: 2025 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::fmt::Debug;
use std::sync::Arc;

use bytes::Bytes;

use super::Key;
use super::node::MavlNode;
use super::node::delete;
use super::node::get;
use super::node::set;

/// A key-value store tree with left and right nodes that supports traversal and value retrieval.
#[derive(Clone, Default, Debug)]
pub(super) struct Avl {
    root: Option<Arc<MavlNode>>,
}

impl Avl {
    /// Delete the node in the tree with a given key.
    pub(super) fn delete(&mut self, key: &Key) {
        delete(&mut self.root, key);
    }

    /// The data stored in a node in the tree with a given key.
    pub(super) fn get(&self, key: &Key) -> Option<&Bytes> {
        get(&self.root, key)
    }

    /// The root node of the tree.
    #[cfg(test)]
    pub(super) fn root(&self) -> &Option<Arc<MavlNode>> {
        &self.root
    }

    /// A mutable reference to the root node of the tree.
    pub(super) fn root_mut(&mut self) -> &mut Option<Arc<MavlNode>> {
        &mut self.root
    }

    /// Set the value of a node in the tree with a given key.
    pub(super) fn set(&mut self, key: &Key, data: Bytes) {
        set(&mut self.root, key, data);
    }
}
