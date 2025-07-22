use cranelift::prelude::Block;

use crate::jit::builder::instruction::LoweredInstruction;
use crate::jit::builder::sequence::join_point_analysis::JoinPointAnalysedLI;
use crate::jit::builder::sequence::join_point_analysis::StepUpdatingOutgoing;

/// An outgoing transition from an instruction to another in the sequence with
/// budget-check assignment if applicable.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct BudgetCheckedOutgoing {
    /// The index of the next instruction in the sequence.
    index: usize,
    /// The hook to the next instruction.
    hook: Block,
    /// The required update to the step-count.
    step_update: Option<u64>,
    /// Indicates if this outgoing transition requires a budget check.
    budget_check: bool,
}

impl BudgetCheckedOutgoing {
    pub(super) fn from_step_updating_outgoing(
        outgoing: StepUpdatingOutgoing,
        budget_check: bool,
    ) -> Self {
        Self {
            index: outgoing.index(),
            hook: outgoing.hook(),
            step_update: outgoing.get_step_update(),
            budget_check,
        }
    }

    /// Get the index of the proceeding instruction.
    pub(super) fn index(&self) -> usize {
        self.index
    }

    /// Return the hook of this instruction.
    pub(super) fn hook(&self) -> Block {
        self.hook
    }

    /// Check if the outgoing transition requires a budget check.
    pub(super) fn budget_check(&self) -> bool {
        self.budget_check
    }

    /// Get the step update for this instruction.
    pub(super) fn step_update(&self) -> Option<u64> {
        self.step_update
    }

    // A mock implementation for testing purposes.
    #[cfg(test)]
    pub(super) fn new(index: usize, step_update: Option<u64>, budget_check: bool) -> Self {
        Self {
            index,
            hook: Block::from_u32(0), // Placeholder
            step_update,
            budget_check,
        }
    }
}

/// `BudgetCheckedLI` holds the information after the budget-check assignment analysis.
/// This provides information on which instructions require a budget check.
#[derive(Clone, Debug)]
pub(super) struct BudgetCheckedLI {
    /// Original LoweredInstruction
    lowered_instr: LoweredInstruction,

    /// Transitions into this instruction.
    incomings: Vec<usize>,

    /// Intra-sequence transitions from this instruction to another.
    outgoings: Vec<BudgetCheckedOutgoing>,

    /// Steps since the last join-point.
    steps_since_last_jp: u64,
}

impl From<JoinPointAnalysedLI> for BudgetCheckedLI {
    fn from(instr: JoinPointAnalysedLI) -> Self {
        Self {
            lowered_instr: instr.lowered_instr().clone(),
            incomings: instr.incomings().to_vec(),
            outgoings: Vec::new(), // Will be populated in the third pass of analysis
            steps_since_last_jp: instr.steps_since_last_jp(),
        }
    }
}

impl BudgetCheckedLI {
    /// Return the original LoweredInstruction.
    pub(super) fn lowered_instr(&self) -> &LoweredInstruction {
        &self.lowered_instr
    }

    /// Get the list of incomings for the instruction.
    pub(super) fn incomings(&self) -> &[usize] {
        &self.incomings
    }

    /// Get the list of outgoings for the instruction.
    pub(super) fn outgoings(&self) -> &[BudgetCheckedOutgoing] {
        &self.outgoings
    }

    /// Return the number of steps since the last join-point for the instruction.
    pub(super) fn steps_since_last_jp(&self) -> u64 {
        self.steps_since_last_jp
    }

    /// Identify if the instruction has any intra-sequence transitions.
    /// If not, it is terminal, so return true.
    pub(super) fn is_terminal(&self) -> bool {
        self.outgoings.is_empty()
    }

    /// Return whether the instruction is a branch point instruction.
    pub(super) fn is_branch_point(&self) -> bool {
        self.outgoings.len() > 1
    }

    /// Return whether the instruction is at the end of the sequence AND has an outgoing transition.
    pub(super) fn is_non_terminating(&self, is_final: bool) -> bool {
        self.outgoings.len() == 1 && is_final
    }

    /// Add an outgoing transition to the instruction.
    pub(super) fn add_outgoing(&mut self, outgoing: BudgetCheckedOutgoing) {
        self.outgoings.push(outgoing);
    }
}

/// Perform the budget-check analysis on the sequence of instructions.
/// This pass enriches the instruction outgoings with classification of whether they require a budget check or not.
#[allow(dead_code)]
pub(super) fn budget_check_analysis(instrs: &mut [JoinPointAnalysedLI]) -> Vec<BudgetCheckedLI> {
    let mut enriched_instrs: Vec<BudgetCheckedLI> = instrs
        .iter_mut()
        .map(|instr| BudgetCheckedLI::from(instr.clone()))
        .collect();

    for instr_index in 0..enriched_instrs.len() {
        if instrs[instr_index].is_branch_point()
            || instrs[instr_index].is_non_terminating(instr_index == enriched_instrs.len() - 1)
        {
            // This is a branch-point instruction, we need to check its outgoings.
            for outgoing in instrs[instr_index].outgoings().iter() {
                let mut destination_index = outgoing.index();

                while let Some(destination) = instrs.get(destination_index) {
                    if destination.is_branch_point()
                        || destination
                            .is_non_terminating(destination_index == enriched_instrs.len() - 1)
                    {
                        // We found another branch-point instruction, so this outgoing requires a budget check.
                        let new_outgoing = BudgetCheckedOutgoing::from_step_updating_outgoing(
                            outgoing.clone(),
                            true,
                        );

                        enriched_instrs[instr_index].add_outgoing(new_outgoing);
                        break;
                    } else if destination.is_terminal() {
                        // We reached an unconditional exit point, so this outgoing does not require a budget check.
                        let new_outgoing = BudgetCheckedOutgoing::from_step_updating_outgoing(
                            outgoing.clone(),
                            false,
                        );
                        enriched_instrs[instr_index].add_outgoing(new_outgoing);
                        break;
                    } else {
                        // We have reached a non-branch-point instruction, so we can continue to the next outgoing.
                        destination_index = instrs[destination_index].outgoings()[0].index();
                    }
                }
            }
        } else {
            // This is a non-branch-point instruction, so just map the first outgoing to a new outgoing if it exists.
            if let Some(outgoing) = instrs[instr_index].outgoings().first() {
                let new_outgoing =
                    BudgetCheckedOutgoing::from_step_updating_outgoing(outgoing.clone(), false);
                enriched_instrs[instr_index].add_outgoing(new_outgoing);
            }
        }
    }

    enriched_instrs
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashSet;

    use super::*;

    pub(crate) fn validate_budget_checks(
        result_instrs: &[BudgetCheckedLI],
        expected_outgoings: &[HashSet<BudgetCheckedOutgoing>],
    ) {
        for instr_index in 0..result_instrs.len() {
            let mut outgoing_set: HashSet<BudgetCheckedOutgoing> =
                HashSet::from_iter(result_instrs[instr_index].outgoings.iter().cloned());
            for expected_outgoing in &expected_outgoings[instr_index] {
                assert!(
                    outgoing_set.contains(expected_outgoing),
                    "Instruction {instr_index} does not contain outgoing {expected_outgoing:?}. 
                    Remaining outgoings: {outgoing_set:?}",
                );
                outgoing_set.remove(expected_outgoing);
            }
        }
    }
}
