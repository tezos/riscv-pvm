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
    #[expect(dead_code, reason = "Used in further analysis stages.")]
    lowered_instr: LoweredInstruction,

    /// Transitions into this instruction.
    #[expect(dead_code, reason = "Used in further analysis stages.")]
    incomings: Vec<usize>,

    /// Intra-sequence transitions from this instruction to another.
    outgoings: Vec<BudgetCheckedOutgoing>,

    /// Steps since the last join-point.
    #[expect(dead_code, reason = "Used in further analysis stages.")]
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
    // Add an outgoing transition to the instruction.
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
    use crate::jit::JIT;
    use crate::jit::builder::sequence::edge_processing::edge_processing;
    use crate::jit::builder::sequence::edge_processing::tests::MockHookedOutgoing;
    use crate::jit::builder::sequence::edge_processing::tests::validate_edge_processing;
    use crate::jit::builder::sequence::join_point_analysis::StepUpdatingOutgoing;
    use crate::jit::builder::sequence::join_point_analysis::join_point_analysis;
    use crate::jit::builder::sequence::join_point_analysis::tests::validate_jp_analysis;
    use crate::jit::builder::sequence::tests::create_lowered_instructions;
    use crate::machine_state::instruction::Instruction;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::registers::nz;
    use crate::machine_state::registers::*;
    use crate::parser::instruction::InstrWidth;

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

    #[test]
    fn test_basic_analysis() {
        let mut jit = JIT::<M4K>::new().unwrap();

        let initial_pc = 0;
        let mut sequence_builder = jit.start(initial_pc);

        let lowered_instrs = create_lowered_instructions(&mut sequence_builder, vec![
            Instruction::new_x64_add(nz::a1, nz::a2, nz::a3, InstrWidth::Uncompressed),
            Instruction::new_x64_sub(nz::a4, nz::a5, nz::a6, InstrWidth::Uncompressed),
            Instruction::new_x64_add(nz::a7, nz::t0, nz::t1, InstrWidth::Compressed),
            Instruction::new_x64_sub(nz::t2, nz::t3, nz::t4, InstrWidth::Uncompressed),
        ]);

        let edge_processed_instrs = edge_processing::<MockHookedOutgoing>(&lowered_instrs);
        validate_edge_processing(
            &edge_processed_instrs,
            &[vec![], vec![0], vec![1], vec![2]],
            &[vec![1], vec![2], vec![3], vec![]],
            &[0, 0, 0, 1],
            &[false, false, false, false],
        );

        let mut jp_analysed_instrs =
            join_point_analysis::<MockHookedOutgoing>(&edge_processed_instrs);

        let expected_outgoings = vec![
            HashSet::from_iter(vec![StepUpdatingOutgoing::new(1, None)]),
            HashSet::from_iter(vec![StepUpdatingOutgoing::new(2, None)]),
            HashSet::from_iter(vec![StepUpdatingOutgoing::new(3, None)]),
            HashSet::new(),
        ];

        validate_jp_analysis(&jp_analysed_instrs, &[0, 1, 2, 3], &expected_outgoings);

        let budget_checked_instrs = budget_check_analysis(&mut jp_analysed_instrs);

        let expected_outgoings = vec![
            HashSet::from_iter(vec![BudgetCheckedOutgoing::new(1, None, false)]),
            HashSet::from_iter(vec![BudgetCheckedOutgoing::new(2, None, false)]),
            HashSet::from_iter(vec![BudgetCheckedOutgoing::new(3, None, false)]),
            HashSet::new(),
        ];

        validate_budget_checks(&budget_checked_instrs, &expected_outgoings);
    }

    #[test]
    fn test_analysis_with_join_point() {
        let mut jit = JIT::<M4K>::new().unwrap();

        let initial_pc = 0;
        let mut sequence_builder = jit.start(initial_pc);

        let lowered_instrs = create_lowered_instructions(&mut sequence_builder, vec![
            Instruction::new_li(nz::a1, 1000, InstrWidth::Uncompressed),
            Instruction::new_li(nz::a2, 1000, InstrWidth::Uncompressed),
            Instruction::new_branch_equal(nz::a1, nz::a2, 8, InstrWidth::Uncompressed),
            Instruction::new_jump_pc(-8, InstrWidth::Uncompressed),
            Instruction::new_x64_store(a1, a2, 0, InstrWidth::Uncompressed),
            Instruction::new_x64_add(nz::a3, nz::a4, nz::a5, InstrWidth::Uncompressed),
        ]);

        let edge_processed_instrs = edge_processing::<MockHookedOutgoing>(&lowered_instrs);
        validate_edge_processing(
            &edge_processed_instrs,
            &[vec![], vec![0, 3], vec![1], vec![2], vec![2], vec![4]],
            &[vec![1], vec![2], vec![3, 4], vec![1], vec![5], vec![]],
            &[0, 0, 0, 0, 1, 1],
            &[false, false, true, false, false, false],
        );

        let mut jp_analysed_instrs =
            join_point_analysis::<MockHookedOutgoing>(&edge_processed_instrs);

        let expected_outgoings = vec![
            HashSet::from_iter(vec![StepUpdatingOutgoing::new(1, Some(1))]),
            HashSet::from_iter(vec![StepUpdatingOutgoing::new(2, None)]),
            HashSet::from_iter(vec![
                StepUpdatingOutgoing::new(3, None),
                StepUpdatingOutgoing::new(4, None),
            ]),
            HashSet::from_iter(vec![StepUpdatingOutgoing::new(1, Some(3))]),
            HashSet::from_iter(vec![StepUpdatingOutgoing::new(5, None)]),
            HashSet::new(),
        ];
        validate_jp_analysis(
            &jp_analysed_instrs,
            &[0, 0, 1, 2, 2, 3],
            &expected_outgoings,
        );

        let budget_checked_instrs = budget_check_analysis(&mut jp_analysed_instrs);

        let expected_outgoings = vec![
            HashSet::from_iter(vec![BudgetCheckedOutgoing::new(1, Some(1), false)]),
            HashSet::from_iter(vec![BudgetCheckedOutgoing::new(2, None, false)]),
            HashSet::from_iter(vec![
                BudgetCheckedOutgoing::new(3, None, true),
                BudgetCheckedOutgoing::new(4, None, false),
            ]),
            HashSet::from_iter(vec![BudgetCheckedOutgoing::new(1, Some(3), false)]),
            HashSet::from_iter(vec![BudgetCheckedOutgoing::new(5, None, false)]),
            HashSet::new(),
        ];

        validate_budget_checks(&budget_checked_instrs, &expected_outgoings);
    }

    #[test]
    fn test_analysis_two_join_points() {
        let mut jit = JIT::<M4K>::new().unwrap();

        let initial_pc = 0;
        let mut sequence_builder = jit.start(initial_pc);

        let lowered_instrs = create_lowered_instructions(&mut sequence_builder, vec![
            Instruction::new_li(nz::a1, 1000, InstrWidth::Uncompressed),
            Instruction::new_li(nz::a2, 1000, InstrWidth::Uncompressed),
            Instruction::new_branch_equal(nz::a1, nz::a2, 6, InstrWidth::Uncompressed),
            Instruction::new_jump_pc(10, InstrWidth::Compressed),
            Instruction::new_branch_not_equal(nz::a1, nz::a2, -10, InstrWidth::Uncompressed),
            Instruction::new_x64_load_signed(a1, a2, 0, InstrWidth::Uncompressed),
            Instruction::new_mul(nz::a3, nz::a4, nz::a5, InstrWidth::Uncompressed),
            Instruction::new_add_word(nz::a6, a7, t0, InstrWidth::Uncompressed),
        ]);

        let edge_processed_instrs = edge_processing::<MockHookedOutgoing>(&lowered_instrs);
        validate_edge_processing(
            &edge_processed_instrs,
            &[
                vec![],
                vec![0, 4],
                vec![1],
                vec![2],
                vec![2],
                vec![4],
                vec![3, 5],
                vec![6],
            ],
            &[
                vec![1],
                vec![2],
                vec![3, 4],
                vec![6],
                vec![1, 5],
                vec![6],
                vec![7],
                vec![],
            ],
            &[0, 0, 0, 0, 0, 1, 0, 1],
            &[false, false, true, false, true, false, false, false],
        );

        let mut jp_analysed_instrs =
            join_point_analysis::<MockHookedOutgoing>(&edge_processed_instrs);

        let expected_outgoings = vec![
            HashSet::from_iter(vec![StepUpdatingOutgoing::new(1, Some(1))]),
            HashSet::from_iter(vec![StepUpdatingOutgoing::new(2, None)]),
            HashSet::from_iter(vec![
                StepUpdatingOutgoing::new(3, None),
                StepUpdatingOutgoing::new(4, None),
            ]),
            HashSet::from_iter(vec![StepUpdatingOutgoing::new(6, Some(3))]),
            HashSet::from_iter(vec![
                StepUpdatingOutgoing::new(1, Some(3)),
                StepUpdatingOutgoing::new(5, None),
            ]),
            HashSet::from_iter(vec![StepUpdatingOutgoing::new(6, Some(4))]),
            HashSet::from_iter(vec![StepUpdatingOutgoing::new(7, None)]),
            HashSet::new(),
        ];

        validate_jp_analysis(
            &jp_analysed_instrs,
            &[0, 0, 1, 2, 2, 3, 0, 1],
            &expected_outgoings,
        );

        let budget_checked_instrs = budget_check_analysis(&mut jp_analysed_instrs);

        let expected_outgoings = vec![
            HashSet::from_iter(vec![BudgetCheckedOutgoing::new(1, Some(1), false)]),
            HashSet::from_iter(vec![BudgetCheckedOutgoing::new(2, None, false)]),
            HashSet::from_iter(vec![
                BudgetCheckedOutgoing::new(3, None, false),
                BudgetCheckedOutgoing::new(4, None, true),
            ]),
            HashSet::from_iter(vec![BudgetCheckedOutgoing::new(6, Some(3), false)]),
            HashSet::from_iter(vec![
                BudgetCheckedOutgoing::new(1, Some(3), true),
                BudgetCheckedOutgoing::new(5, None, false),
            ]),
            HashSet::from_iter(vec![BudgetCheckedOutgoing::new(6, Some(4), false)]),
            HashSet::from_iter(vec![BudgetCheckedOutgoing::new(7, None, false)]),
            HashSet::new(),
        ];

        validate_budget_checks(&budget_checked_instrs, &expected_outgoings);
    }
}
