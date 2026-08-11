// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Reclaiming the Merkle nodes no retained root holds.
//!
//! Removing commit directories cannot touch these. Node bodies are content-addressed, so every
//! version of every node is a distinct live key that nothing ever deletes, and the store grows with
//! the total number of node writes over a repository's lifetime rather than with the size of the
//! state. This is the part that deletes those keys.
//!
//! # Deciding liveness upwards
//!
//! A node is live when a retained root still reaches it. That is asked of each node from the node's
//! own side, by walking the reverse edges upwards, rather than by traversing every retained root
//! downwards. Each answer is memoised, and each edge that led to one is stamped with the sequence
//! number of the root it led to, so the next collection reads the answer off the edge instead of
//! walking again.
//!
//! A stamp is only ever written where the node is provably reachable from the stamped root. Nodes
//! are content-addressed and therefore immutable, so if a parent is reachable from a root and that
//! parent refers to a child, the child is reachable from that root too, permanently. A stale stamp
//! costs a walk and an over-generous one retains garbage; neither can drop something still live.
//!
//! The graph cannot contain a cycle: a node's hash is derived from its children's, so a node can
//! never be its own ancestor.

use std::collections::HashMap;

use octez_riscv_data::hash::Hash;

use super::Suspend;
use crate::avl::node::stored_children;
use crate::errors::OperationalError;
use crate::journal::Seq;
use crate::merkle_store::MerkleStore;
use crate::merkle_store::Stamp;

/// What a sweep of the Merkle store reclaimed.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct SweptNodes {
    /// Node bodies removed.
    pub nodes: usize,

    /// Bytes those bodies occupied, keys included.
    pub bytes: u64,

    /// Reverse edges removed with them.
    pub edges: usize,

    /// Whether the sweep stopped early because it was asked to.
    ///
    /// What it deleted is deleted; the rest is left for the next round, which starts again from
    /// what is present.
    pub suspended: bool,
}

/// Delete every node in `store` that no root in `roots` still reaches.
///
/// `roots` maps the tree hash a database was committed at to the sequence number of the most recent
/// retained registry commit that names it. `floor` is the sequence number being collected at, which
/// every retained root is at or above.
pub fn sweep(
    store: &MerkleStore,
    roots: &HashMap<Hash, Seq>,
    floor: Seq,
    suspend: &Suspend,
) -> Result<SweptNodes, OperationalError> {
    let mut liveness = Liveness {
        store,
        roots,
        floor,
        known: HashMap::new(),
    };

    // Collected first, because deciding liveness reads edges and deleting rewrites them, and an
    // iterator is not the place to be doing either.
    let mut dead = Vec::new();
    store.for_each_node(|key, len| {
        dead.push((key.to_vec(), len));
    })?;

    let mut swept = SweptNodes::default();

    for (key, len) in dead {
        // Checked per node rather than per batch: deciding one is a walk of bounded length, so this
        // is as fine-grained as stopping needs to be.
        if suspend.requested() {
            swept.suspended = true;
            return Ok(swept);
        }

        if liveness.of(&key)?.is_some() {
            continue;
        }

        // Noted before the first removal, so that a crash part-way still leaves a store that knows
        // an absent node may be one it collected.
        if swept.nodes == 0 {
            store.note_collected()?;
        }

        swept.edges += remove_node(store, &key)?;
        swept.nodes += 1;
        swept.bytes += len as u64 + Hash::DIGEST_SIZE as u64;
    }

    Ok(swept)
}

/// Delete the node stored under `key` and every edge that mentions it, reporting how many edges.
///
/// The edges into its children go with it: they record that this node referred to them, and it no
/// longer exists to. The edges to its own parents go too - every parent of a dead node is itself
/// dead, or the node would have been live through it.
fn remove_node(store: &MerkleStore, key: &[u8]) -> Result<usize, OperationalError> {
    let mut edges = 0;

    // Read before deleting: the body is the only record of what this node referred to.
    if let Ok(body) = store.get(key) {
        for child in stored_children(body.as_ref())? {
            store.delete_edge(child.as_ref(), key)?;
            edges += 1;
        }
    }

    edges += store.parents_of(key)?.len();
    store.delete_edges_from(key)?;

    store.delete(key)?;

    Ok(edges)
}

/// Read a store key back as the hash it is.
///
/// Anything of another length was not written by the Merkle layer, so it is nobody's root.
fn hash_of(key: &[u8]) -> Option<Hash> {
    <[u8; Hash::DIGEST_SIZE]>::try_from(key)
        .ok()
        .map(Hash::from)
}

/// Answers, and remembers, whether a node is still held by a retained root.
struct Liveness<'a> {
    store: &'a MerkleStore,
    roots: &'a HashMap<Hash, Seq>,
    floor: Seq,

    /// What has already been decided. `None` records a node shown to be unreachable, which is worth
    /// remembering as much as the other answer: its children ask the same question next.
    known: HashMap<Vec<u8>, Option<Seq>>,
}

impl Liveness<'_> {
    /// The most recent retained root holding the node under `key`, if any still does.
    fn of(&mut self, key: &[u8]) -> Result<Option<Seq>, OperationalError> {
        if let Some(known) = self.known.get(key) {
            return Ok(*known);
        }

        let answer = self.compute(key)?;
        self.known.insert(key.to_vec(), answer);

        Ok(answer)
    }

    /// Work out the answer for `key`, without consulting what is already known about it.
    fn compute(&mut self, key: &[u8]) -> Result<Option<Seq>, OperationalError> {
        // A retained root holds itself.
        if let Some(seq) = hash_of(key).and_then(|hash| self.roots.get(&hash)) {
            return Ok(Some(*seq));
        }

        let parents = self.store.parents_of(key)?;
        let mut held_by = None;

        for (parent, stamp) in &parents {
            // The stamp already says a retained root holds this child, so there is nothing to walk.
            if stamp.holds_at(self.floor) {
                return Ok(Some(self.floor));
            }

            if let Some(seq) = self.of(parent)? {
                held_by = Some(held_by.map_or(seq, |best: Seq| best.max(seq)));
            }
        }

        // Record what was learned on the edges that led to it, so the next collection reads the
        // answer instead of walking for it. Only edges to a live parent are stamped, which is what
        // keeps a stamp a proof rather than a guess.
        if let Some(seq) = held_by {
            for (parent, _) in &parents {
                if self.of(parent)?.is_some() {
                    self.store.stamp_edge(key, parent, Stamp::at(seq))?;
                }
            }
        }

        Ok(held_by)
    }
}

/// Roots that a sweep must not collect, and how recently each was committed.
///
/// Deliberately a plain map rather than anything cleverer: the number of retained roots is the
/// retention window, not the size of the state.
pub type RetainedRoots = HashMap<Hash, Seq>;
