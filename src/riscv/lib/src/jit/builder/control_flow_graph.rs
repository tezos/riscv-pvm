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

use std::collections::VecDeque;

use itertools::Itertools;

use crate::jit::builder::instr_map::InstrId;
use crate::jit::builder::instr_map::InstrMap;
use crate::jit::builder::instr_map::InstrMapBuilder;
use crate::jit::builder::instruction::OutcomeProbability;
use crate::jit::builder::outcome_map::ExitKind;
use crate::jit::builder::outcome_map::Graph;
use crate::jit::builder::outcome_map::OutcomeId;
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

    /// Probability of this outcome being taken.
    pub probability: OutcomeProbability,

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

    /// If the program exits after this outcome, the step counter must be further increased by this
    /// amount (in addition to `base_diff`-related increments),
    /// as subsequent step-counter updates will not fire.
    exit_delta: usize,

    /// Edge where the step counter needs updating
    edge: &'info DirectedEdgeInfo<T>,
}

impl<'info, T> StepCounterUpdate<'info, T> {
    /// Number of steps to increment the step counter on a successful outcome.
    pub fn success_delta(&self) -> usize {
        // wrapping_add is required as base_diff can be negative.
        self.base_diff.wrapping_add(1)
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

/// The location of a budget-check. Any budget-check point allows for
/// an exit of the sequence.
#[derive(Debug)]
pub struct BudgetCheckLoc<'info, T> {
    // TODO: RV-812: fallback to interpreter at budget-check locations if
    // budget check fails.
    edge: &'info DirectedEdgeInfo<T>,
}

impl<'info, T> BudgetCheckLoc<'info, T> {
    /// Information about the edge where the budget check occurs.
    pub fn edge(&self) -> &'info DirectedEdgeInfo<T> {
        self.edge
    }
}

/// Information about a step budget check.
///
/// A budget is the maximum number of steps that can be taken from this point
/// before having the opportunity to exit again. This would be either by reaching
/// another budget-check point (which would exit if the budget is greater than
/// the remaining steps), or by reaching a direct exit point.
#[derive(Debug)]
pub struct BudgetCheck<'info, T> {
    /// Number of steps that may be run from this point, before another budget check takes place.
    budget: usize,
    /// The uncommitted step-count upon taking this edge.
    /// If the program exits after this outcome, the step counter must be further increased by this
    /// amount, as subsequent step-counter updates will not fire.
    exit_delta: usize,
    /// Edge where the budget check occurs.
    edge: &'info DirectedEdgeInfo<T>,
}

impl<'info, T> BudgetCheck<'info, T> {
    /// Returns the budget associated with this budget check.
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Returns the uncommitted step-count upon taking this edge.
    pub fn exit_delta(&self) -> usize {
        self.exit_delta
    }

    /// Returns the edge where the budget check occurs.
    pub fn edge(&self) -> &'info DirectedEdgeInfo<T> {
        self.edge
    }
}

/// Control flow graph of instructions
#[derive(Debug)]
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
                            .unwrap_or(TargetInstrLoc::Exit(ExitKind::Normal)),

                        Target::Unknown => {
                            // If the target is not known to the current context, we treat it as an
                            // exit. Commonly unknown targets used to represent exceptions. The JIT
                            // function will exit in this case.
                            TargetInstrLoc::Exit(ExitKind::Normal)
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

    /// Find all the edges where the step counter needs updating and the uncommitted step-count after
    /// every transition.
    pub fn find_step_counter_updates(
        &self,
    ) -> (
        OutcomeMap<Option<StepCounterUpdate<'_, T>>>,
        OutcomeMap<usize>,
    ) {
        // The delta is the number of steps performed since the last StepCounterUpdate.
        // 0 means the step counter was updated through an incoming edge.
        // None means the value has not been computed yet.
        let mut instr_deltas: InstrMap<Option<usize>> = self.nodes.map(|_, _| None);

        let mut step_updates = self.outcomes.map(|_, _| None);
        let mut exit_deltas = self.outcomes.map(|_, _| 0usize);

        let mut outcome_queue =
            VecDeque::from(self.graph.outgoing_outcomes(SourceInstrLoc::Entry).to_vec());

        while let Some(outcome_id) = outcome_queue.pop_front() {
            let source_delta = match self.outcomes[outcome_id].from() {
                SourceInstrLoc::Entry => None,
                SourceInstrLoc::Internal(source) => instr_deltas[source],
            };

            let edge = self.outcomes[outcome_id].data();

            match self.outcomes[outcome_id].to() {
                TargetInstrLoc::Exit(_) => {
                    let sc_update = StepCounterUpdate {
                        base_diff: source_delta.expect("Exit must have a source delta."),
                        exit_delta: 0,
                        edge: edge.expect("Exit edge must have edge info."),
                    };
                    *step_updates[outcome_id].data_mut() = Some(sc_update);
                }
                TargetInstrLoc::Internal(dest_id) => match instr_deltas[dest_id] {
                    Some(existing_delta) => {
                        // an unset source delta means the source is an entry point, implying
                        // a delta of 0.
                        let source_delta = source_delta.unwrap_or_default();

                        // The existing delta may be larger than the source delta. In this case,
                        // reconciling the step counter requires a negative diff.
                        let base_diff = source_delta.wrapping_sub(existing_delta);

                        let sc_update = StepCounterUpdate {
                            base_diff,
                            exit_delta: existing_delta,
                            edge: edge.expect("Internal edge must have edge info."),
                        };

                        *step_updates[outcome_id].data_mut() = Some(sc_update);
                        *exit_deltas[outcome_id].data_mut() = existing_delta;
                    }
                    None => {
                        let dest_delta = source_delta.map_or(0, |delta| delta.wrapping_add(1));
                        instr_deltas[dest_id] = Some(dest_delta);
                        *exit_deltas[outcome_id].data_mut() = dest_delta;

                        outcome_queue.extend(
                            self.graph
                                .outgoing_outcomes(SourceInstrLoc::Internal(dest_id)),
                        );
                    }
                },
            }
        }

        (step_updates, exit_deltas)
    }

    /// Get outgoing outcomes from a node, sorted by probability, with most probable first.
    fn sorted_outgoings(&self, source_node: InstrId) -> impl Iterator<Item = &OutcomeId> {
        self.graph
            .outgoing_outcomes(SourceInstrLoc::Internal(source_node))
            .iter()
            .sorted_by_key(|child_outcome| {
                self.outcomes[**child_outcome]
                    .data()
                    .expect("All outgoings from a node must have associated outcome data.")
                    .probability
            })
    }

    /// Iterative DFS to detect cycles in the control flow graph.
    /// Marks edges that close cycles as budget check locations.
    pub fn find_budget_check_locations(&self) -> OutcomeMap<Option<BudgetCheckLoc<'_, T>>> {
        let mut budget_checks = self.outcomes.map(|_, _| None);
        let mut nodes = self.nodes.map(|_, _| NodeState::Unchecked);

        let mut work_stack = self
            .graph
            .outgoing_outcomes(SourceInstrLoc::Entry)
            .iter()
            .filter_map(|outcome| {
                self.outcomes[*outcome]
                    .to()
                    .as_internal()
                    .map(|&node_id| (node_id, self.sorted_outgoings(node_id)))
            })
            .collect_vec();

        while let Some((node_id, mut child_outcomes)) = work_stack.pop() {
            nodes[node_id] = NodeState::Checking;

            let Some(child_outcome) = child_outcomes.next() else {
                // All children processed, mark node as visited.
                nodes[node_id] = NodeState::Checked;
                continue;
            };

            // Push the current node back onto the stack to process remaining children later.
            work_stack.push((node_id, child_outcomes));

            let child_node_id = match self.outcomes[*child_outcome].to() {
                TargetInstrLoc::Internal(id) => id,
                TargetInstrLoc::Exit(_) => continue,
            };

            match nodes[child_node_id] {
                NodeState::Checked => continue,
                NodeState::Checking => {
                    // we found a cycle. Mark this edge as a budget check.
                    let loc_data = budget_checks[*child_outcome].data_mut();
                    *loc_data = Some(BudgetCheckLoc {
                        edge: self.outcomes[*child_outcome]
                            .data()
                            .expect("Internal edge must have edge info."),
                    });
                }
                NodeState::Unchecked => {
                    work_stack.push((child_node_id, self.sorted_outgoings(child_node_id)));
                }
            }
        }

        budget_checks
    }

    /// Find the budget required for each node to guarantee reaching either an exit or a budget check.
    fn calculate_node_budgets(
        &self,
        bc_locations: &OutcomeMap<Option<BudgetCheckLoc<'info, T>>>,
    ) -> InstrMap<Option<usize>> {
        let mut budgets = self.nodes.map(|_, _| None);

        let mut work_stack = self
            .graph
            .outgoing_outcomes(SourceInstrLoc::Entry)
            .iter()
            .filter_map(|outcome| {
                let &node_id = self.outcomes[*outcome].to().as_internal()?;
                let work_item = (node_id, self.sorted_outgoings(node_id).collect_vec());
                Some(work_item)
            })
            .collect_vec();

        while let Some((node_id, child_outcomes)) = work_stack.pop() {
            let mut max_budget = 1;
            for &&child_outcome in &child_outcomes {
                if bc_locations[child_outcome].data().is_some() {
                    // Budget check edge, budget is 1.
                    continue;
                }

                let child_node_id = match self.outcomes[child_outcome].to() {
                    TargetInstrLoc::Internal(id) => id,
                    TargetInstrLoc::Exit(_) => {
                        continue;
                    }
                };

                match budgets[child_node_id] {
                    Some(child_budget) => {
                        max_budget = max_budget.max(1 + child_budget);
                    }
                    None => {
                        // A child node's budget has not yet been computed.
                        // Push the current node back onto the stack, followed by the child node.
                        // This ensures the child's budget is computed first before returning
                        // to process the current node.
                        //
                        // Infinite loops are avoided because cycles in the graph are broken by
                        // budget check edges. A node will not appear on the stack more than once.
                        work_stack.push((node_id, child_outcomes));
                        work_stack.push((
                            child_node_id,
                            self.sorted_outgoings(child_node_id).collect_vec(),
                        ));
                        break;
                    }
                }
            }

            budgets[node_id] = Some(max_budget);
        }

        budgets
    }

    /// Fill in budgets at the given budget check locations, returning the calculated values
    /// along with the minimum budget for entering the sequence.
    pub fn annotate_budget_checks(
        &self,
        bc_locations: &OutcomeMap<Option<BudgetCheckLoc<'info, T>>>,
        exit_deltas: &OutcomeMap<usize>,
    ) -> SequenceBudget<'info, T> {
        let budgets = self.calculate_node_budgets(bc_locations);

        // The minimum starting budget is the maximum budget across all entry nodes, which is only
        // one node in the current design.
        let min_budget = self
            .graph
            .outgoing_outcomes(SourceInstrLoc::Entry)
            .iter()
            .filter_map(|&outcome| {
                let &node_id = self.outcomes[outcome].to().as_internal()?;
                budgets[node_id]
            })
            .max()
            .expect("At least one entry point must exist.");

        let budget_checks = bc_locations.map(|outcome_id, outcome| {
            let bc_loc = outcome.data().as_ref()?;
            let exit_delta = *exit_deltas[outcome_id].data();
            let &node_id = self.outcomes[outcome_id].to().as_internal()?;
            let budget_check = BudgetCheck {
                budget: budgets[node_id]?,
                exit_delta,
                edge: bc_loc.edge(),
            };
            Some(budget_check)
        });

        SequenceBudget {
            min_budget,
            budget_checks,
        }
    }
}

/// State of a node during DFS traversal for budget check detection.
enum NodeState {
    Unchecked,
    Checking,
    Checked,
}

/// Budget information for a sequence of instructions.
pub struct SequenceBudget<'info, T> {
    min_budget: usize,
    budget_checks: OutcomeMap<Option<BudgetCheck<'info, T>>>,
}

impl<T> SequenceBudget<'_, T> {
    /// Get the minimum budget required for the entry to the sequence.
    pub fn min_budget(&self) -> usize {
        self.min_budget
    }

    /// Get the budget checks for the sequence.
    pub fn budget_checks(&self) -> &OutcomeMap<Option<BudgetCheck<'_, T>>> {
        &self.budget_checks
    }
}

// By convention, test modules are at the end of the file.
//
// We don't put the conditional compilation attribute here to avoid issues with Clippy. Instead, we
// put it into the module file itself. Clippy doesn't look at the module declaration in the parent
// file to determine whether the module is test-only or not. That means some non-test lints would
// trigger in the tests module.
mod tests;
