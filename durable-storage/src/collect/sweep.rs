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
//!
//! A round holds one decision per node in the store, so its peak is set by how much the store has
//! grown rather than by the size of the live state - which is the shape of the store it is there
//! to shrink. Decisions are keyed by the hash itself, not by a copy of it on the heap.

use std::collections::HashMap;

use octez_riscv_data::hash::Hash;

use crate::avl::node::stored_children;
use crate::errors::Error;
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
) -> Result<SweptNodes, OperationalError> {
    let mut liveness = Liveness {
        store,
        roots,
        floor,
        decided: HashMap::new(),
    };

    // Collected first, because deciding liveness reads edges and deleting rewrites them, and an
    // iterator is not the place to be doing either.
    let mut candidates = Vec::new();
    store.for_each_node(|key, len| {
        // Anything of another length was not written by the Merkle layer, so it is nobody's node
        // and nobody's root: left where it is rather than swept.
        if let Some(key) = hash_of(key) {
            candidates.push((key, len));
        }
    })?;

    let mut swept = SweptNodes::default();

    for (key, len) in candidates {
        if liveness.of(key)?.is_some() {
            continue;
        }

        swept.edges += remove_node(store, key, &liveness)?;
        liveness.mark_swept(key);
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
fn remove_node(
    store: &MerkleStore,
    key: Hash,
    liveness: &Liveness,
) -> Result<usize, OperationalError> {
    let key = key.as_ref();

    // Read before deleting: the body is the only record of what this node referred to. A body that
    // is already gone is what a repeated round finds, and is not an error - but a read that failed
    // for any other reason is, and must not be taken for an absent body: the edges into this
    // node's children would then never be removed, and nothing revisits it to find them.
    let children = match store.get(key) {
        Ok(body) => stored_children(body.as_ref())?,
        // Absent, which is what a repeated round finds: reading a node is the only
        // invalid-argument this can raise.
        Err(Error::InvalidArgument(_)) => Vec::new(),
        Err(Error::Operational(error)) => return Err(error),
    };

    // An edge between two dead nodes is reachable from both ends, so it is counted by whichever
    // of them went first and not again by the second.
    let counted = children
        .iter()
        .filter(|child| !liveness.was_swept(child))
        .count();

    Ok(counted + store.delete_node(key, &children)?)
}

/// Read a store key back as the hash it is.
///
/// Anything of another length was not written by the Merkle layer, so it is nobody's node.
fn hash_of(key: &[u8]) -> Option<Hash> {
    <[u8; Hash::DIGEST_SIZE]>::try_from(key)
        .ok()
        .map(Hash::from)
}

/// What a round has settled about a node.
#[derive(Clone, Copy)]
enum Decision {
    /// Held by a retained root, which was last recorded at this position.
    Held(Seq),

    /// No retained root reaches it. Worth remembering as much as the other answer: its children
    /// ask the same question next.
    Dead,

    /// Dead, and already deleted by this round.
    Swept,
}

impl Decision {
    /// The root holding the node, if one still does.
    fn held_by(self) -> Option<Seq> {
        match self {
            Self::Held(seq) => Some(seq),
            Self::Dead | Self::Swept => None,
        }
    }
}

/// Answers, and remembers, whether a node is still held by a retained root.
struct Liveness<'a> {
    store: &'a MerkleStore,
    roots: &'a HashMap<Hash, Seq>,
    floor: Seq,

    /// What has already been settled, one entry per node the round has reached.
    decided: HashMap<Hash, Decision>,
}

impl Liveness<'_> {
    /// The most recent retained root holding the node under `key`, if any still does.
    fn of(&mut self, key: Hash) -> Result<Option<Seq>, OperationalError> {
        if let Some(decided) = self.decided.get(&key) {
            return Ok(decided.held_by());
        }

        let answer = self.compute(key)?;
        self.decided
            .insert(key, answer.map_or(Decision::Dead, Decision::Held));

        Ok(answer)
    }

    /// Whether this round has already deleted the node under `key`.
    fn was_swept(&self, key: &Hash) -> bool {
        matches!(self.decided.get(key), Some(Decision::Swept))
    }

    /// Record that the node under `key`, already decided dead, has been deleted.
    fn mark_swept(&mut self, key: Hash) {
        self.decided.insert(key, Decision::Swept);
    }

    /// Work out the answer for `key`, without consulting what is already known about it.
    fn compute(&mut self, key: Hash) -> Result<Option<Seq>, OperationalError> {
        // A retained root holds itself.
        if let Some(seq) = self.roots.get(&key) {
            return Ok(Some(*seq));
        }

        let parents = self.store.parents_of(key.as_ref())?;
        let mut held_by = None;

        for (parent, stamp) in &parents {
            // The stamp already says a retained root holds this child, so there is nothing to walk.
            if stamp.holds_at(self.floor) {
                return Ok(Some(self.floor));
            }

            let Some(parent) = hash_of(parent) else {
                continue;
            };

            if let Some(seq) = self.of(parent)? {
                held_by = Some(held_by.map_or(seq, |best: Seq| best.max(seq)));
            }
        }

        // Record what was learned on the edges that led to it, so the next collection reads the
        // answer instead of walking for it. Only edges to a live parent are stamped, which is what
        // keeps a stamp a proof rather than a guess.
        if let Some(seq) = held_by {
            for (parent, _) in &parents {
                let Some(parent) = hash_of(parent) else {
                    continue;
                };

                if self.of(parent)?.is_some() {
                    self.store
                        .stamp_edge(key.as_ref(), parent.as_ref(), Stamp::at(seq))?;
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
