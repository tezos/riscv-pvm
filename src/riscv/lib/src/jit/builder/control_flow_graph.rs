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
pub struct DirectedEdgeInfo<Info, Target = self::Target> {
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
    nodes: InstrMap<Node<'info, T>>,
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
        let (addrs, infos) = {
            let mut infos_builder = InstrMapBuilder::new();

            for instr in instrs {
                infos_builder.insert(instr.location, instr);
            }
            infos_builder.build()
        };

        // The various graph analysis mechanisms require both forward and backward edges for
        // traversal. This allows them to be expressed more intuitively.
        let mut backward_edges = infos.map(|_, _| Vec::new());
        let forward_edges = infos.map(|source_idx, source_info| {
            let mut exit_edges = Vec::new();
            let success_edges = source_info
                .outgoing
                .iter()
                .filter_map(|edge_info| {
                    let Target::Known(target_addr) = edge_info.target else {
                        // Edge target is outside of the known context. It will exit the JIT
                        // function.
                        exit_edges.push(edge_info);
                        return None;
                    };

                    // Is the target address part of the graph context? If not, it'll become an
                    // exit edge.
                    let Some(target_idx) = addrs.translate(target_addr) else {
                        exit_edges.push(edge_info);
                        return None;
                    };

                    // It's more efficient to build the backward edges while we're iterating forward
                    // edges, rather than iterating the entire graph multiple times.
                    backward_edges[target_idx].push(DirectedEdgeInfo {
                        target: source_idx,
                        info: edge_info,
                    });

                    Some(DirectedEdgeInfo {
                        target: target_idx,
                        info: edge_info,
                    })
                })
                .collect();

            (exit_edges, success_edges)
        });

        // Combine all the information into a more convenient structure for analysis.
        let nodes = backward_edges.zip2_into_with(
            forward_edges,
            |idx, backward_edges, (exit_edges, forward_edges)| Node {
                backward_edges: backward_edges.into_boxed_slice(),
                forward_edges,
                exit_edges: exit_edges.into_boxed_slice(),
                info: infos[idx],
            },
        );
        Self { nodes }
    }

    /// Classify the backward facing edges of a node.
    fn classify_backward(&self, idx: InstrId) -> EdgeClass<'_, 'info, T> {
        let node = &self.nodes[idx];
        match node.backward_edges.len() {
            0 => EdgeClass::Orphan,
            1 if !node.info.is_entrypoint => EdgeClass::SingleEdge(&node.backward_edges[0]),
            _ => EdgeClass::MultipleEdges,
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

                let EdgeClass::SingleEdge(backward) = self.classify_backward(pos) else {
                    // Join points require their incoming edges to flush the step counter, thereby
                    // setting delta back to 0. Unreachable nodes get set to 0 as well, as their
                    // step delta is irrelevant.
                    step_deltas[pos] = Some(0usize);
                    continue;
                };

                if let Some(delta) = step_deltas[backward.target] {
                    // If the previous node already has a step delta, we can just use that to
                    // determine the current node's step delta.
                    step_deltas[pos] = Some(delta + 1);
                    continue;
                }

                if cursor.already_seen(backward.target) {
                    // We have encountered a cycle. That means it is impossible to resolve an
                    // accurate delta. We simply require the incoming edge to flush the step
                    // counter. This makes the step delta 0.
                    step_deltas[pos] = Some(0usize);
                    continue;
                }

                cursor.not_done_yet([backward.target]);
            }
        }

        step_deltas.map(|_idx, delta| delta.expect("All nodes should have a step delta by now"))
    }

    /// Find all the edges where the step counter needs updating.
    pub fn find_step_counter_updates(&self) -> Box<[StepCounterUpdate<'info, T>]> {
        let step_deltas = self.find_step_deltas();

        // We pre-allocate space for the updates somewhat optimistically to avoid frequent
        // re-allocations at the cost of some potential overallocation.
        let mut updates = Vec::with_capacity(self.nodes.len().saturating_mul(2));

        for (idx, node) in self.nodes.iter() {
            let source_delta = step_deltas[idx];

            // First, let's consider all forward edges to other nodes in the graph context. Those
            // are eligible to update the step counter.
            for edge in node.forward_edges.iter() {
                let target_delta = step_deltas[edge.target];

                // When the target delta is less or equal, that means the step counter is expected
                // to be updated through edges to the target.
                // The equal case is important for loops of instructions onto themselves. Those are
                // step delta 0 -> 0 transitions.
                if target_delta <= source_delta {
                    updates.push(StepCounterUpdate {
                        base_diff: source_delta - target_delta,
                        edge: edge.info,
                    });
                }
            }

            // We mustn't forget exit edges as those are important points for updating the step
            // counter as well. Forgetting these would mean that exiting a sequence of instructions
            // would not update the step counter correctly.
            for exit_edge in self.nodes[idx].exit_edges.iter() {
                updates.push(StepCounterUpdate {
                    base_diff: step_deltas[idx],
                    edge: exit_edge,
                });
            }
        }

        updates.into_boxed_slice()
    }
}

/// Directed edge where the target has been resolved to a node within the graph context
type ResolvedDirectedEdge<'info, T> = self::DirectedEdgeInfo<&'info DirectedEdgeInfo<T>, InstrId>;

/// Node in the control flow graph
///
/// This is usually information relating to an instruction.
struct Node<'info, T> {
    /// Edges to this node that originate from another node in this graph context
    backward_edges: Box<[ResolvedDirectedEdge<'info, T>]>,

    /// Edges from this node that target another node in the graph context
    forward_edges: Box<[ResolvedDirectedEdge<'info, T>]>,

    /// Exiting edges from this node that target outside of the graph context
    exit_edges: Box<[&'info DirectedEdgeInfo<T>]>,

    /// Information about this node
    info: &'info NodeInfo<T>,
}

/// Classification of edges from a node
enum EdgeClass<'edge, 'info, E> {
    /// No attached edges (explicit or implicit)
    Orphan,

    // Exactly one explicit edge
    SingleEdge(&'edge ResolvedDirectedEdge<'info, E>),

    // Multiple explicit edges, or one explicit edge and one implicit edge (entrypoint)
    MultipleEdges,
}
