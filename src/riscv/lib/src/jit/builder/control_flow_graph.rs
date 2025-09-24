// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Analysis of control flow graphs for JIT compilation
//!
//! This module provides functionality to analyse control flow graphs of instructions. This
//! information is relevant in determining the minimal set of step counter updates required to
//! accurately track the number of executed instructions.

// TODO: RV-703: Integrate this module with the JIT compiler. The reason this is test-only for now
// is because we require several components to be in place for the integration to make sense. So we
// introduce and test each component extensively in isolation first, then integrate them all at
// once later.
#![cfg(test)]

use crate::jit::builder::graph_walker::GraphWalker;
use crate::jit::builder::instr_map::InstrId;
use crate::jit::builder::instr_map::InstrMap;
use crate::jit::builder::instr_map::InstrMapBuilder;
use crate::jit::builder::outcome_map::Graph;
use crate::jit::builder::outcome_map::OutcomeMap;
use crate::jit::builder::outcome_map::OutcomeMapBuilder;
use crate::jit::builder::outcome_map::SourceInstrLoc;
use crate::jit::builder::outcome_map::TargetInstrLoc;
use crate::machine_state::memory::Address;

/// Destination of a directed edge
#[derive(Debug)]
pub enum Target {
    /// Target is an instruction at a known address
    ///
    /// The address may or may not be part of the current instruction sequence. If it is, the
    /// control flow analysis will treat it as another node. If it isn't, the analysis will treat it
    /// as an exit from the current context.
    Known(Address),

    /// Target is outside of the known context
    ///
    /// When using this, the control flow analysis will assume that the target is not part of the
    /// current instruction sequence. This is used for jumps to addresses outside of the current
    /// context, for example. It can also be used for exception-raising outcomes.
    Unknown,
}

/// Information about a directed edge in the control flow graph
///
/// Edges represent outcomes of instructions. For example, an instruction could raise an exception,
/// jump to another address, or continue to the next instruction. Each of these would be represented
/// as a different edge in the control flow graph.
#[derive(Debug)]
pub struct DirectedEdgeInfo<Info> {
    /// Destination of the edge
    pub target: Target,

    /// Extra information about the edge, but not relevant to the analysis
    pub info: Info,
}

/// Information about a node in the control flow graph
///
/// Nodes represent instructions. They contain information about the instruction itself, as well as
/// the outgoing edges from that instruction.
#[derive(Debug)]
pub struct NodeInfo<T> {
    /// Location of the instruction (program counter)
    pub location: Address,

    /// Is this instruction used as an entrypoint in the instruction sequence?
    ///
    /// Entrypoints have implicit incoming edges from outside of the known context.
    pub is_entrypoint: bool,

    /// Edges originating from this node
    pub outgoing: Box<[DirectedEdgeInfo<T>]>,
}

/// Information about a step counter update
#[derive(Debug)]
pub struct StepCounterUpdate<'info, T> {
    /// Number of steps to increment the step counter without considering the source instruction
    /// step impact
    ///
    /// For example, a successful outcome would increase the step counter by `base_diff + 1`.
    /// Unsuccessful outcomes increase by just `base_diff`.
    base_diff: usize,

    /// Edge where the step counter needs updating
    edge: &'info DirectedEdgeInfo<T>,
}

impl<'info, T> StepCounterUpdate<'info, T> {
    /// Number of steps to increment the step counter on a successful outcome.
    pub fn success_delta(&self) -> usize {
        self.base_diff + 1
    }

    /// Number of steps to increment the step counter on an exception outcome.
    pub fn exception_delta(&self) -> usize {
        self.base_diff
    }

    /// Information about the edge where the step counter needs updating.
    pub fn edge(&self) -> &'info DirectedEdgeInfo<T> {
        self.edge
    }
}

/// Control flow graph of instructions
pub struct ControlFlowGraph<'info, T> {
    /// Nodes in the control flow graph, indexed by instruction ID
    nodes: InstrMap<&'info NodeInfo<T>>,

    /// Edges in the control flow graph, indexed by outcome ID
    outcomes: OutcomeMap<Option<&'info DirectedEdgeInfo<T>>>,

    /// Underlying graph structure for efficient traversal
    graph: Graph,
}

impl<'info, T> ControlFlowGraph<'info, T>
where
    T: 'info,
{
    /// Constructs a new control flow graph from instruction information.
    pub fn new(instrs: impl IntoIterator<Item = &'info NodeInfo<T>>) -> Self {
        // We need to put all the information we're given into an initial graph for more efficient
        // retrieval. `InstrMapBuilder` also provides us with a mechanism to translate addresses
        // such that we know whether they're in the current graph context or not.
        let (addrs, nodes) = {
            let mut infos_builder = InstrMapBuilder::new();

            for instr in instrs {
                infos_builder.insert(instr.location, instr);
            }

            infos_builder.build()
        };

        let (graph, outcomes) = {
            let mut builder = OutcomeMapBuilder::new(nodes.len());

            for (idx, node) in nodes.iter() {
                if node.is_entrypoint {
                    // Entry points are presented as having an edge from outside the context.
                    builder.insert(SourceInstrLoc::Entry, TargetInstrLoc::Internal(idx), None);
                }

                let from_loc = SourceInstrLoc::Internal(idx);

                for edge in node.outgoing.iter() {
                    let to_loc = match edge.target {
                        Target::Known(addr) => addrs
                            .translate(addr)
                            // If the address is known, we create an internal edge.
                            .map(TargetInstrLoc::Internal)
                            // If the address is not known, we create an exit edge.
                            .unwrap_or(TargetInstrLoc::Exit),

                        Target::Unknown => {
                            // If the target is not known to the current context, we treat it as an
                            // exit. Commonly unknown targets used to represent exceptions. The JIT
                            // function will exit in this case.
                            TargetInstrLoc::Exit
                        }
                    };

                    builder.insert(from_loc, to_loc, Some(edge));
                }
            }

            builder.build()
        };

        Self {
            nodes,
            outcomes,
            graph,
        }
    }

    /// Classify the incoming edges of a node. The goal is to find the previous node if it is
    /// unambiguous.
    fn classify_incoming(&self, idx: InstrId) -> EdgeClass {
        let [one] = self.graph.incoming_outcomes(TargetInstrLoc::Internal(idx)) else {
            return EdgeClass::AmbiguousOrNone;
        };

        match self.outcomes[*one].from() {
            SourceInstrLoc::Internal(source) => EdgeClass::Unambiguous(source),
            SourceInstrLoc::Entry => EdgeClass::AmbiguousOrNone,
        }
    }

    /// Find the step deltas for each node in the graph.
    ///
    /// A step delta is the number of steps that have been performed since the last step counter
    /// update. 0 means the step counter was updated through an incoming edge of a node.
    fn find_step_deltas(&self) -> InstrMap<usize> {
        let mut step_deltas = self.nodes.map(|_, _| None);

        for (idx, _) in self.nodes.iter() {
            let mut walker = GraphWalker::new(idx);
            while let Some(cursor) = walker.next() {
                let pos = cursor.position();

                if step_deltas[pos].is_some() {
                    // If the step delta is already set there is nothing for us to do.
                    continue;
                }

                let EdgeClass::Unambiguous(source) = self.classify_incoming(pos) else {
                    // Join points require their incoming edges to flush the step counter, thereby
                    // setting delta back to 0. Unreachable nodes get set to 0 as well, as their
                    // step delta is irrelevant.
                    step_deltas[pos] = Some(0usize);
                    continue;
                };

                if let Some(delta) = step_deltas[source] {
                    // If the previous node already has a step delta, we can just use that to
                    // determine the current node's step delta.
                    step_deltas[pos] = Some(delta + 1);
                    continue;
                }

                if cursor.already_seen(source) {
                    // We have encountered a cycle. That means it is impossible to resolve an
                    // accurate delta. We simply require the incoming edge to flush the step
                    // counter. This makes the step delta 0.
                    step_deltas[pos] = Some(0usize);
                    continue;
                }

                cursor.not_done_yet([source]);
            }
        }

        step_deltas.map(|_idx, delta| delta.expect("All nodes should have a step delta by now"))
    }

    /// Find all the edges where the step counter needs updating.
    pub fn find_step_counter_updates(&self) -> OutcomeMap<Option<StepCounterUpdate<'_, T>>> {
        let step_deltas = self.find_step_deltas();

        self.outcomes.map(|_idx, outcome| {
            let Some(edge) = outcome.data() else {
                // There is nothing to do for edges that don't have corresponding data on caller
                // side.
                return None;
            };

            let SourceInstrLoc::Internal(source) = outcome.from() else {
                // There is nothing to do for edges from outside of the graph context. These would
                // target entrypoints anyway.
                return None;
            };
            let source_delta = step_deltas[source];

            match outcome.to() {
                TargetInstrLoc::Internal(target) => {
                    let target_delta = step_deltas[target];

                    // When the target delta is less or equal, that means the step counter is expected
                    // to be updated through edges to the target.
                    // The equal case is important for loops of instructions onto themselves. Those are
                    // step delta 0 -> 0 transitions.
                    if target_delta <= source_delta {
                        return Some(StepCounterUpdate {
                            base_diff: source_delta - target_delta,
                            edge,
                        });
                    }

                    None
                }

                TargetInstrLoc::Exit => {
                    // Exit transitions need to ensure the step counter is updated to reflect the
                    // number of executed instructions in the current context.
                    Some(StepCounterUpdate {
                        base_diff: step_deltas[source],
                        edge,
                    })
                }
            }
        })
    }
}

/// Classification of edges from a node
enum EdgeClass {
    /// Has exactly one explicit edge and no implicit edges
    Unambiguous(InstrId),

    /// Has implicit edges or not exactly 1 explicit edge
    AmbiguousOrNone,
}

// By convention, test modules are at the end of the file.
//
// We don't put the conditional compilaton attribute here to avoid issues with Clippy. Instead, we
// put it into the module file itself. Clippy doesn't look at the module declaration in the parent
// file to determine whether the module is test-only or not. That means some non-test lints would
// trigger in the tests module.
mod tests;
