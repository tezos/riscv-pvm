use cranelift::prelude::Block;

use crate::jit::builder::instruction::LoweredInstruction;
use crate::jit::builder::sequence::budget_check_analysis::BudgetCheckedLI;
use crate::jit::builder::sequence::budget_check_analysis::BudgetCheckedOutgoing;

/// An outgoing transition from an instruction to another in the sequence after budget assignment.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct BudgetAssignedOutgoing {
    index: usize,
    hook: Block,
    step_update: Option<u64>,
    budget: Option<u64>,
}

impl BudgetAssignedOutgoing {
    pub(super) fn from_budget_checked_outgoing(
        outgoing: BudgetCheckedOutgoing,
        budget: u64,
    ) -> Self {
        if outgoing.budget_check() {
            Self {
                index: outgoing.index(),
                hook: outgoing.hook(),
                step_update: outgoing.step_update(),
                budget: Some(budget),
            }
        } else {
            Self {
                index: outgoing.index(),
                hook: outgoing.hook(),
                step_update: outgoing.step_update(),
                budget: None, // No budget check required.
            }
        }
    }

    #[cfg(test)]
    pub(super) fn new(index: usize, step_update: Option<u64>, budget: Option<u64>) -> Self {
        Self {
            index,
            hook: Block::from_u32(0), // Placeholder
            step_update,
            budget,
        }
    }

    /// Return the index of the target instruction.
    pub(super) fn index(&self) -> usize {
        self.index
    }

    /// Return the hook for this outgoing transition.
    pub(super) fn hook(&self) -> Block {
        self.hook
    }

    /// Return the step update value for this outgoing transition, or None if
    /// no step update is required.
    pub(super) fn step_update(&self) -> Option<u64> {
        self.step_update
    }

    /// Return the budget-check value for this outgoing transition, or None if
    /// no budget-check is required.
    pub(super) fn budget(&self) -> Option<u64> {
        self.budget
    }
}

/// BudgetAssignedLI holds the information after the budget assignment analysis.
/// This provides the budget values required for each budget check.
#[derive(Clone)]
pub(super) struct BudgetAssignedLI {
    /// Original LoweredInstruction
    lowered_instr: LoweredInstruction,

    /// Transitions into this instruction.
    #[expect(
        dead_code,
        reason = "This information may be required in further iterations."
    )]
    incomings: Vec<usize>,

    /// Intra-sequence transitions from this instruction to another.
    outgoings: Vec<BudgetAssignedOutgoing>,

    /// Steps since the last join-point.
    steps_since_last_jp: u64,
}

impl From<BudgetCheckedLI> for BudgetAssignedLI {
    fn from(instr: BudgetCheckedLI) -> Self {
        Self {
            lowered_instr: instr.lowered_instr().clone(),
            incomings: instr.incomings().to_vec(),
            outgoings: Vec::new(),
            steps_since_last_jp: instr.steps_since_last_jp(),
        }
    }
}

impl BudgetAssignedLI {
    /// Return the original LoweredInstruction.
    pub(super) fn lowered_instr(&self) -> &LoweredInstruction {
        &self.lowered_instr
    }

    /// Get the list of outgoings for the instruction.
    pub(super) fn outgoings(&self) -> &[BudgetAssignedOutgoing] {
        &self.outgoings
    }

    /// Add an outgoing transition to the instruction.
    pub(super) fn add_outgoing(&mut self, outgoing: BudgetAssignedOutgoing) {
        self.outgoings.push(outgoing);
    }

    /// Return steps since the last join-point.
    pub(super) fn steps_since_last_jp(&self) -> u64 {
        self.steps_since_last_jp
    }
}

/// Holds budget-check information during analysis.
///
/// Since each instruction can have a maximum of two dependencies from which it can derive its
/// budget, we optionally store indices of 2 instructions here.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(super) struct BudgetCheck {
    /// The budget of the instruction that this budget check is for.
    budget: u64,
    /// Whether this budget check has been resolved or not.
    resolved: bool,
    /// The index of the first dependency, if applicable.
    outgoing_a: Option<usize>,
    /// The index of the second dependency, if applicable.
    outgoing_b: Option<usize>,
}

impl BudgetCheck {
    pub fn new() -> Self {
        Self {
            budget: 0,
            resolved: false,
            outgoing_a: None,
            outgoing_b: None,
        }
    }

    pub fn check_resolved(&self) -> bool {
        if self.outgoing_a.is_none() && self.outgoing_b.is_none() {
            return true;
        }
        false
    }

    pub fn set_budget(&mut self, budget: u64) {
        self.budget = budget.max(self.budget);
    }
}

/// Perform the budget assignment analysis on the sequence of instructions.
/// This pass enriches the instruction outgoings with the budget check values if required.
pub(crate) fn budget_assignment(
    instrs: &mut [BudgetCheckedLI],
) -> (Vec<BudgetCheck>, Vec<BudgetAssignedLI>) {
    let mut budgets = vec![BudgetCheck::new(); instrs.len()];

    let mut enriched_instrs = instrs
        .iter()
        .map(|instr| BudgetAssignedLI::from(instr.clone()))
        .collect::<Vec<_>>();

    if instrs.len() < 2 {
        // If there are less than 2 instructions, there are no intra-sequence transitions.
        return (budgets, enriched_instrs);
    }

    for instr_index in 0..instrs.len() {
        let budget_check = &mut budgets[instr_index];

        if instrs[instr_index].is_terminal() {
            budget_check.budget = 1; // Terminal instruction, budget is 1.
            budget_check.resolved = true;
            continue;
        }

        if instrs[instr_index].is_non_terminating(instr_index == instrs.len() - 1) {
            // We must treat a non-terminating final instruction as a special case. Without setting this
            // to being resolved, it will cause an infinite loop in the budget assignment.
            budget_check.budget = 1;
            budget_check.resolved = true;
            continue;
        }

        if !instrs[instr_index].is_branch_point() {
            // This is a non-branch-point instruction that is not terminal.
            let destination_index = instrs[instr_index].outgoings()[0].index();
            budget_check.outgoing_a = Some(destination_index);
            continue;
        }

        // This is a branch-point, so both outgoings must be checked.
        let outgoing_a = &instrs[instr_index].outgoings()[0];
        let outgoing_b = &instrs[instr_index].outgoings()[1];

        if outgoing_a.budget_check() {
            budget_check.budget = 1;
            budget_check.outgoing_a = None;
        } else {
            budget_check.outgoing_a = Some(outgoing_a.index());
        }

        if outgoing_b.budget_check() {
            budget_check.budget = 1;
            budget_check.outgoing_b = None;
        } else {
            budget_check.outgoing_b = Some(outgoing_b.index());
        }

        if budget_check.check_resolved() {
            budget_check.resolved = true;
        }
    }

    let mut progress_counter = 1;
    let mut all_resolved = false;

    let mut iter_count = 0;

    while !all_resolved && progress_counter > 0 {
        progress_counter = 0;
        all_resolved = true;
        iter_count += 1;

        for instr_index in 0..instrs.len() {
            if budgets[instr_index].resolved {
                continue;
            }

            if let Some(dest_a) = budgets[instr_index].outgoing_a {
                if budgets[dest_a].resolved {
                    let possible_budget = budgets[dest_a].budget + 1;
                    budgets[instr_index].set_budget(possible_budget);
                    progress_counter += 1;
                    budgets[instr_index].outgoing_a = None; // Clear the outgoing since it's resolved.
                }
            }

            if let Some(dest_b) = budgets[instr_index].outgoing_b {
                if budgets[dest_b].resolved {
                    let possible_budget = budgets[dest_b].budget + 1;
                    budgets[instr_index].set_budget(possible_budget);
                    progress_counter += 1;
                    budgets[instr_index].outgoing_b = None; // Clear the outgoing since it's resolved.
                }
            }

            if budgets[instr_index].check_resolved() {
                budgets[instr_index].resolved = true;
            } else {
                all_resolved = false;
            }
        }

        if iter_count > 1000 || progress_counter == 0 {
            // panic!(
            //     "Infinite loop detected in fourth pass analysis. iter: {}, progress_counter: {}.

            // {:#?}",
            //     iter_count, progress_counter, instrs
            // );
            break;
        }
    }

    // Now we have the budgets for each instruction, we can set the budget check values.
    for instr_index in 0..instrs.len() {
        for outgoing in instrs[instr_index].outgoings() {
            let dest_index = outgoing.index();
            let new_outgoing = BudgetAssignedOutgoing::from_budget_checked_outgoing(
                outgoing.clone(),
                budgets[dest_index].budget,
            );
            enriched_instrs[instr_index].add_outgoing(new_outgoing);
        }
    }
    (budgets, enriched_instrs)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashSet;

    use super::*;

    pub(crate) fn validate_budgets(
        result_budgets: &[BudgetCheck],
        result_instrs: &[BudgetAssignedLI],
        expected_budgets: &[u64],
        expected_outgoings: &[HashSet<BudgetAssignedOutgoing>],
    ) {
        for instr_index in 0..result_instrs.len() {
            assert_eq!(
                result_budgets[instr_index].budget, expected_budgets[instr_index],
                "Failed to match budget. Result budgets: {result_budgets:?}"
            );

            let mut outgoing_set: HashSet<BudgetAssignedOutgoing> =
                HashSet::from_iter(result_instrs[instr_index].outgoings().iter().cloned());
            for expected_outgoing in &expected_outgoings[instr_index] {
                assert!(
                    outgoing_set.contains(expected_outgoing),
                    "Instruction {instr_index} does not contain outgoing {expected_outgoing:?}. 
                    Remaining outgoings: {outgoing_set:?}"
                );
                outgoing_set.remove(expected_outgoing);
            }
        }
    }
}
