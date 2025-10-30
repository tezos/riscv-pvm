// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Data structures for managing instruction control flow outcomes
//!
//! Outcomes represent edges in the control flow graph. They can originate from outside of the
//! control flow graph, or from another instruction (node) within the graph. Similarly, they can
//! transition to another instruction within the graph, or to outside of it.
//!
//! "Outcome", "transition", and "edge" can be used interchangeably in this context.
//!
//! This module provides data structures and utilities to maintain a mapping between
//! instruction control flow transitions and their associated information. It includes [`OutcomeMap`] for storing
//! outcomes, [`Graph`] for navigating control flow relationships, and [`OutcomeMapBuilder`] for
//! constructing these structures in a consistent manner.

// TODO: RV-703 - `OutcomeMap` is currently only used in tests and has not yet been integrated into the JIT.
#![cfg(test)]

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Range;

use crate::jit::builder::instr_map::InstrId;

/// Unique identifier for an instruction outcome
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OutcomeId(usize);

/// Instruction outcome
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome<T> {
    /// Location of the source instruction
    ///
    /// This may be outside of the control flow graph context.
    from: SourceInstrLoc,

    /// Location of the target instruction
    ///
    /// This may be outside of the control flow graph context.
    to: TargetInstrLoc,

    /// Data associated with the outcome
    data: T,
}

impl<T> Outcome<T> {
    /// Get the source location.
    pub fn from(&self) -> SourceInstrLoc {
        self.from
    }

    /// Get the target location.
    pub fn to(&self) -> TargetInstrLoc {
        self.to
    }

    /// Obtain a reference to the associated outcome data.
    pub fn data(&self) -> &T {
        &self.data
    }

    /// Obtain a mutable reference to the associated outcome data.
    pub fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

/// Collection of [outcomes] indexable by their respective [ids]
///
/// [outcomes]: Outcome
/// [ids]: OutcomeId
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutcomeMap<T> {
    data: Box<[Outcome<T>]>,
}

impl<T> OutcomeMap<T> {
    /// Obtain an iterator over all outcomes.
    pub fn iter(&self) -> impl Iterator<Item = (OutcomeId, &Outcome<T>)> {
        self.data
            .iter()
            .enumerate()
            .map(|(idx, value)| (OutcomeId(idx), value))
    }

    /// Map each outcome to a new value, producing a new [`OutcomeMap`].
    pub fn map<R>(&self, mut f: impl FnMut(OutcomeId, &Outcome<T>) -> R) -> OutcomeMap<R> {
        let data = self
            .data
            .iter()
            .enumerate()
            .map(|(idx, outcome)| {
                let data = f(OutcomeId(idx), outcome);
                Outcome {
                    from: outcome.from,
                    to: outcome.to,
                    data,
                }
            })
            .collect();
        OutcomeMap { data }
    }
}

impl<T> Index<OutcomeId> for OutcomeMap<T> {
    type Output = Outcome<T>;

    fn index(&self, id: OutcomeId) -> &Self::Output {
        &self.data[id.0]
    }
}

impl<T> IndexMut<OutcomeId> for OutcomeMap<T> {
    fn index_mut(&mut self, id: OutcomeId) -> &mut Self::Output {
        &mut self.data[id.0]
    }
}

/// Location of a source instruction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceInstrLoc {
    /// Instruction within the control flow graph context
    Internal(InstrId),

    /// Outside of the control flow graph context
    Entry,
}

/// Location of a target instruction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetInstrLoc {
    /// Instruction within the control flow graph context
    Internal(InstrId),

    /// Outside of the control flow graph context
    Exit,
}

impl TargetInstrLoc {
    /// Retrieve target instruction id, unless the target is an exit.
    pub fn as_internal(&self) -> Option<&InstrId> {
        match self {
            TargetInstrLoc::Internal(id) => Some(id),
            TargetInstrLoc::Exit => None,
        }
    }
}

/// Graph with single direction edges
///
/// These graphs can be used to represent either forward and backward control flow relationships
/// but never both at the same time.
///
/// # How it works
///
/// To look at the outcomes associated with an instruction, first we check if the instruction is
/// within the control flow graph context.
///
/// If it is, we use its ID as an index into the `indices` array to get a range into the `ids`
/// array. The `ids` array contains all outcome IDs relevant to this graph, so we can use the
/// obtained range to slice into it and get the outcome IDs associated with the instruction.
///
/// If the instruction is outside of the control flow graph context, we use the `ext_indices`
/// range to slice into the `ids` array and get the outcome IDs that have an external component.
#[derive(Debug)]
struct SingleDirectionGraph {
    /// Mapping from instruction ID (the index) to the range of outcomes in [`Self::ids`]
    indices: Box<[Range<usize>]>,

    /// Range of outcomes that are external to the control flow graph context
    ext_indices: Range<usize>,

    /// All outcome ID sequences relevant to this graph
    ids: Box<[OutcomeId]>,
}

impl SingleDirectionGraph {
    /// Instantiate a new single-direction graph.
    fn new(nodes: BTreeMap<InstrId, BTreeSet<OutcomeId>>, external: BTreeSet<OutcomeId>) -> Self {
        let max_instrs = nodes
            .last_key_value()
            .map(|(&key, _)| usize::from(key))
            .unwrap_or(0);
        let mut indices: Box<[Range<usize>]> = (0..=max_instrs).map(|_| 0..0).collect();

        let num_ids = nodes
            .values()
            .map(BTreeSet::len)
            .fold(external.len(), usize::saturating_add);
        let mut ids = Vec::with_capacity(num_ids);

        for (instr, outcomes) in nodes {
            let start = ids.len();
            ids.extend(outcomes);
            let end = ids.len();

            let index = usize::from(instr);
            indices[index] = start..end;
        }

        let external = {
            let start = ids.len();
            ids.extend(external);
            let end = ids.len();

            start..end
        };

        Self {
            indices,
            ext_indices: external,
            ids: ids.into_boxed_slice(),
        }
    }

    /// Find the outcomes connected to the given instruction
    ///
    /// These are the outcomes that either start from or lead to the given instruction, depending
    /// on whether this graph represents forward or backward control flow.
    fn outcomes(&self, instr: InstrId) -> &[OutcomeId] {
        let index = usize::from(instr);
        let range = self.indices.get(index).cloned().unwrap_or(0..0);
        &self.ids[range]
    }

    /// Get the outcomes that have an external component.
    ///
    /// Those would be outcomes that either start from or lead to outside of the control flow graph
    /// context.
    fn external_outcomes(&self) -> &[OutcomeId] {
        let range = self.ext_indices.clone();
        &self.ids[range]
    }
}

/// Bidirectional graph for navigating instruction outcomes
#[derive(Debug)]
pub struct Graph {
    /// Graph for outgoing control flow
    ///
    /// This graph represents the forward control flow relationships between instructions.
    outgoing: SingleDirectionGraph,

    /// Graph for incoming control flow
    ///
    /// This graph represents the backward control flow relationships between instructions.
    incoming: SingleDirectionGraph,
}

impl Graph {
    /// Find the outgoing outcomes for a given source instruction.
    pub fn outgoing_outcomes(&self, loc: SourceInstrLoc) -> &[OutcomeId] {
        match loc {
            SourceInstrLoc::Internal(instr) => self.outgoing.outcomes(instr),
            SourceInstrLoc::Entry => self.outgoing.external_outcomes(),
        }
    }

    /// Find the incoming outcomes for a given target instruction.
    ///
    /// These are the outcomes that lead to the specified instruction.
    pub fn incoming_outcomes(&self, loc: TargetInstrLoc) -> &[OutcomeId] {
        match loc {
            TargetInstrLoc::Internal(instr) => self.incoming.outcomes(instr),
            TargetInstrLoc::Exit => self.incoming.external_outcomes(),
        }
    }
}

/// Builder for [`OutcomeMap`] and [`Graph`]
///
/// This builder facilitates the construction of an [`OutcomeMap`] and its associated [`Graph`]
/// in a consistent manner. It ensures that outcomes are correctly indexed and that the graph
/// accurately reflects the control flow relationships between instructions.
#[derive(Debug)]
pub struct OutcomeMapBuilder<T> {
    /// All outcomes in the control flow graph
    outcomes: Vec<Outcome<T>>,

    /// Mapping from an instruction to all its incoming outcomes
    incomings: BTreeMap<InstrId, BTreeSet<OutcomeId>>,

    /// Mapping from an instruction to all its outgoing outcomes
    outgoings: BTreeMap<InstrId, BTreeSet<OutcomeId>>,

    /// All outcomes that start from outside of the control flow graph context
    entries: BTreeSet<OutcomeId>,

    /// All outcomes that lead to outside of the control flow graph context
    exits: BTreeSet<OutcomeId>,
}

impl<T> OutcomeMapBuilder<T> {
    /// Instantiate a new builder.
    pub fn new(num_instrs: usize) -> Self {
        // Conservatively estimate the number of outcomes to avoid reallocations.
        let num_outcomes = num_instrs.saturating_mul(2);

        Self {
            outcomes: Vec::with_capacity(num_outcomes),
            incomings: BTreeMap::new(),
            outgoings: BTreeMap::new(),
            entries: BTreeSet::new(),
            exits: BTreeSet::new(),
        }
    }

    /// Track a new outcome in the control flow graph.
    pub fn insert(&mut self, from: SourceInstrLoc, to: TargetInstrLoc, data: T) {
        let id = OutcomeId(self.outcomes.len());
        self.outcomes.push(Outcome { data, from, to });

        match from {
            SourceInstrLoc::Internal(instr) => {
                self.incomings.entry(instr).or_default().insert(id);
            }

            SourceInstrLoc::Entry => {
                self.entries.insert(id);
            }
        }

        match to {
            TargetInstrLoc::Internal(instr) => {
                self.outgoings.entry(instr).or_default().insert(id);
            }

            TargetInstrLoc::Exit => {
                self.exits.insert(id);
            }
        }
    }

    /// Build [`OutcomeMap`] and [`Graph`].
    pub fn build(self) -> (Graph, OutcomeMap<T>) {
        let map = OutcomeMap {
            data: self.outcomes.into_boxed_slice(),
        };

        let graph = Graph {
            outgoing: SingleDirectionGraph::new(self.incomings, self.entries),
            incoming: SingleDirectionGraph::new(self.outgoings, self.exits),
        };

        (graph, map)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::jit::builder::graph_walker::GraphWalker;
    use crate::jit::builder::instr_map::InstrMapBuilder;
    use crate::jit::builder::outcome_map::OutcomeMapBuilder;
    use crate::jit::builder::outcome_map::SourceInstrLoc;
    use crate::jit::builder::outcome_map::TargetInstrLoc;

    #[test]
    fn empty() {
        let (graph, outcomes) = OutcomeMapBuilder::<()>::new(0).build();

        assert!(outcomes.iter().next().is_none());

        let other_outcomes = outcomes.map(|_, _| ());
        assert_eq!(outcomes, other_outcomes);

        let entries = graph.outgoing_outcomes(SourceInstrLoc::Entry);
        assert!(entries.is_empty());

        let exits = graph.incoming_outcomes(TargetInstrLoc::Exit);
        assert!(exits.is_empty());
    }

    #[test]
    fn mutation() {
        let mut instrs = InstrMapBuilder::new();

        instrs.insert(0, 3);
        instrs.insert(1, 2);
        instrs.insert(2, 1);

        let (addr_map, instrs) = instrs.build();

        let id0 = addr_map.translate(0).unwrap();
        let id1 = addr_map.translate(1).unwrap();
        let id2 = addr_map.translate(2).unwrap();

        let mut outcomes = OutcomeMapBuilder::new(instrs.len());

        outcomes.insert(SourceInstrLoc::Entry, TargetInstrLoc::Internal(id0), 1);
        outcomes.insert(
            SourceInstrLoc::Internal(id0),
            TargetInstrLoc::Internal(id1),
            2,
        );
        outcomes.insert(
            SourceInstrLoc::Internal(id1),
            TargetInstrLoc::Internal(id2),
            3,
        );
        outcomes.insert(SourceInstrLoc::Internal(id2), TargetInstrLoc::Exit, 4);

        let (_graph, mut outcomes) = outcomes.build();

        let ids = outcomes.iter().map(|(id, _)| id).collect::<Vec<_>>();
        let mut results = BTreeMap::new();

        for id in ids {
            let lhs = match outcomes[id].from() {
                SourceInstrLoc::Internal(instr_id) => instrs[instr_id],
                SourceInstrLoc::Entry => 0,
            };

            let rhs = match outcomes[id].to() {
                TargetInstrLoc::Internal(instr_id) => instrs[instr_id],
                TargetInstrLoc::Exit => 0,
            };

            let val = *outcomes[id].data();
            let new_val = val + lhs * rhs;

            results.insert(id, new_val);

            *outcomes[id].data_mut() = new_val;
        }

        for (id, val) in results {
            assert_eq!(*outcomes[id].data(), val);
        }
    }

    #[test]
    fn forward_traversal() {
        let mut instrs = InstrMapBuilder::new();

        instrs.insert(0, ());
        instrs.insert(1, ());
        instrs.insert(2, ());

        let (addr_map, instr_map) = instrs.build();

        let id0 = addr_map.translate(0).unwrap();
        let id1 = addr_map.translate(1).unwrap();
        let id2 = addr_map.translate(2).unwrap();

        let mut outcomes = OutcomeMapBuilder::new(instr_map.len());

        outcomes.insert(
            SourceInstrLoc::Entry,
            TargetInstrLoc::Internal(id0),
            "hello",
        );
        outcomes.insert(
            SourceInstrLoc::Internal(id0),
            TargetInstrLoc::Internal(id1),
            " ",
        );
        outcomes.insert(
            SourceInstrLoc::Internal(id1),
            TargetInstrLoc::Internal(id2),
            "world",
        );
        outcomes.insert(SourceInstrLoc::Internal(id2), TargetInstrLoc::Exit, "!");

        let (graph, outcomes) = outcomes.build();

        let mut result = String::new();

        let mut walker = GraphWalker::new(SourceInstrLoc::Entry);
        while let Some(cursor) = walker.next() {
            let pos = cursor.position();

            let out_ids = graph.outgoing_outcomes(pos);
            for &out_id in out_ids {
                let &outcome = outcomes[out_id].data();
                result.push_str(outcome);
            }

            cursor.done(out_ids.iter().filter_map(|&out| match outcomes[out].to() {
                TargetInstrLoc::Internal(instr) => Some(SourceInstrLoc::Internal(instr)),
                TargetInstrLoc::Exit => None,
            }));
        }

        assert_eq!(result, "hello world!");
    }

    #[test]
    fn backward_traversal() {
        let mut instrs = InstrMapBuilder::new();

        instrs.insert(0, ());
        instrs.insert(1, ());
        instrs.insert(2, ());

        let (addr_map, instr_map) = instrs.build();

        let id0 = addr_map.translate(0).unwrap();
        let id1 = addr_map.translate(1).unwrap();
        let id2 = addr_map.translate(2).unwrap();

        let mut outcomes = OutcomeMapBuilder::new(instr_map.len());

        outcomes.insert(SourceInstrLoc::Internal(id0), TargetInstrLoc::Exit, "hello");
        outcomes.insert(
            SourceInstrLoc::Internal(id1),
            TargetInstrLoc::Internal(id0),
            " ",
        );
        outcomes.insert(
            SourceInstrLoc::Internal(id2),
            TargetInstrLoc::Internal(id1),
            "world",
        );
        outcomes.insert(SourceInstrLoc::Entry, TargetInstrLoc::Internal(id2), "!");

        let (graph, outcomes) = outcomes.build();

        let mut result = String::new();

        let mut walker = GraphWalker::new(TargetInstrLoc::Exit);
        while let Some(cursor) = walker.next() {
            let pos = cursor.position();

            let out_ids = graph.incoming_outcomes(pos);
            for &out_id in out_ids {
                let &outcome = outcomes[out_id].data();
                result.push_str(outcome);
            }

            cursor.done(
                out_ids
                    .iter()
                    .filter_map(|&out| match outcomes[out].from() {
                        SourceInstrLoc::Internal(instr) => Some(TargetInstrLoc::Internal(instr)),
                        SourceInstrLoc::Entry => None,
                    }),
            );
        }

        assert_eq!(result, "hello world!");
    }
}
