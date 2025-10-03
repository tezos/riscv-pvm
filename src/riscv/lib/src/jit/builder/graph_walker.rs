// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Graph walker utilities

// TODO: RV-703: We're waiting for a set of components to have merged before we can integrate them
// into the JIT compiler. For the time being, this module will be test-only.
#![cfg(test)]

use std::collections::HashSet;
use std::collections::VecDeque;
use std::hash::Hash;

/// Helper for walking graph
pub struct GraphWalker<P> {
    /// Double-ended queue of node positions to visit
    works: VecDeque<P>,

    /// Set of node positions already visited
    seen: HashSet<P>,
}

impl<P: Eq + Hash + Copy> GraphWalker<P> {
    /// Construct a new graph walker starting at the given position.
    pub fn new(start: P) -> Self {
        Self {
            works: VecDeque::from([start]),
            seen: HashSet::new(),
        }
    }

    /// Retrieve the next position to walk to, if any.
    pub fn next(&mut self) -> Option<GraphCursor<'_, P>> {
        let idx = self.works.pop_front()?;
        self.seen.insert(idx);

        let graph_cursor = GraphCursor { walker: self, idx };
        Some(graph_cursor)
    }
}

/// Cursor pointing to a node in a graph
pub struct GraphCursor<'a, P> {
    /// Reference to the walker state
    ///
    /// We require this in case we need to queue up more work. It also provides access to the set of
    /// already-seen nodes.
    walker: &'a mut GraphWalker<P>,

    /// Current node position
    idx: P,
}

impl<P: Eq + Hash + Copy> GraphCursor<'_, P> {
    /// Position of the current node.
    pub fn position(&self) -> P {
        self.idx
    }

    /// Have we walked to this node before? This does not exclude the current cursor position.
    pub fn already_seen(&self, idx: P) -> bool {
        self.walker.seen.contains(&idx)
    }

    /// Mark the current node as not finished yet. The graph walker will navigate to nodes in
    /// `work_before` first, and then return to the current node later.
    pub fn not_done_yet(self, work_before: impl IntoIterator<Item = P>) {
        self.walker.works.extend(work_before);
        self.walker.works.push_back(self.idx);
    }

    /// Mark the current node as done. The graph walker will navigate to the given nodes next.
    pub fn done(self, next: impl IntoIterator<Item = P>) {
        self.walker.works.extend(next);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::collections::HashSet;

    use proptest::collection::hash_map;
    use proptest::collection::hash_set;
    use proptest::num::usize;
    use proptest::prelude::Strategy;
    use proptest::prelude::TestCaseError;
    use proptest::proptest;

    use super::GraphWalker;

    /// Property-based testing strategy for a graph
    ///
    /// The value is a mapping from a node ID to a set of other node IDs that are immediately
    /// reachable from the key node.
    fn graph() -> impl Strategy<Value = HashMap<usize, HashSet<usize>>> {
        (1..=100usize).prop_flat_map(|node_domain| {
            let node_strat = 0..node_domain;
            let targets_strat = hash_set(node_strat.clone(), 0..=node_domain);
            hash_map(node_strat, targets_strat, 1..=node_domain)
        })
    }

    /// Test that we can walk arbitrary graphs without getting stuck in infinite loops.
    #[test]
    fn walk_non_infinite() {
        fn inner(graph: HashMap<usize, HashSet<usize>>) -> Result<(), TestCaseError> {
            for &start in graph.keys() {
                let mut walker = GraphWalker::new(start);
                while let Some(cursor) = walker.next() {
                    let pos = cursor.position();

                    let Some(targets) = graph.get(&pos) else {
                        continue;
                    };

                    let targets = targets
                        .iter()
                        .copied()
                        .filter(|&target| !cursor.already_seen(target))
                        .collect::<Vec<_>>();

                    if targets.is_empty() {
                        continue;
                    }

                    cursor.not_done_yet(targets);
                }

                // If the loop finishes, we have successfully avoided infinite loops.
            }

            Ok(())
        }

        // We have moved out the body into a separate function to avoid formatting problems with the
        // macro.
        proptest!(|(graph in graph())| {
            inner(graph)?;
        });
    }

    /// Basic test to ensure that recursion would be detectable. Detection involves checking if a
    /// node has been seen before.
    #[test]
    fn recursion_is_detectable() {
        let mut walker = GraphWalker::new(0);

        let cursor = walker.next().unwrap();
        assert_eq!(cursor.position(), 0);
        assert!(cursor.already_seen(0));
        assert!(!cursor.already_seen(1));

        cursor.not_done_yet([1]);

        let cursor = walker.next().unwrap();
        assert_eq!(cursor.position(), 1);
        assert!(cursor.already_seen(0));
    }
}
