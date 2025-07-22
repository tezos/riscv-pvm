#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::vec;

    use crate::jit::JIT;
    use crate::jit::builder::sequence::budget_assignment::BudgetAssignedOutgoing;
    use crate::jit::builder::sequence::budget_assignment::budget_assignment;
    use crate::jit::builder::sequence::budget_assignment::tests::validate_budgets;
    use crate::jit::builder::sequence::budget_check_analysis::BudgetCheckedOutgoing;
    use crate::jit::builder::sequence::budget_check_analysis::budget_check_analysis;
    use crate::jit::builder::sequence::budget_check_analysis::tests::validate_budget_checks;
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

        let mut budget_checked_instrs = budget_check_analysis(&mut jp_analysed_instrs);

        let expected_outgoings = vec![
            HashSet::from_iter(vec![BudgetCheckedOutgoing::new(1, None, false)]),
            HashSet::from_iter(vec![BudgetCheckedOutgoing::new(2, None, false)]),
            HashSet::from_iter(vec![BudgetCheckedOutgoing::new(3, None, false)]),
            HashSet::new(),
        ];

        validate_budget_checks(&budget_checked_instrs, &expected_outgoings);
        let (budgets, budget_assigned_instrs) = budget_assignment(&mut budget_checked_instrs);

        let expected_budgets = vec![4, 3, 2, 1];

        let expected_outgoings = vec![
            HashSet::from_iter(vec![BudgetAssignedOutgoing::new(1, None, None)]),
            HashSet::from_iter(vec![BudgetAssignedOutgoing::new(2, None, None)]),
            HashSet::from_iter(vec![BudgetAssignedOutgoing::new(3, None, None)]),
            HashSet::new(),
        ];

        validate_budgets(
            &budgets,
            &budget_assigned_instrs,
            &expected_budgets,
            &expected_outgoings,
        );
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

        let mut budget_checked_instrs = budget_check_analysis(&mut jp_analysed_instrs);

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

        let (budgets, budget_assigned_instrs) = budget_assignment(&mut budget_checked_instrs);

        let expected_budgets = vec![5, 4, 3, 5, 2, 1];

        let expected_outgoings = vec![
            HashSet::from_iter(vec![BudgetAssignedOutgoing::new(1, Some(1), None)]),
            HashSet::from_iter(vec![BudgetAssignedOutgoing::new(2, None, None)]),
            HashSet::from_iter(vec![
                BudgetAssignedOutgoing::new(3, None, Some(5)),
                BudgetAssignedOutgoing::new(4, None, None),
            ]),
            HashSet::from_iter(vec![BudgetAssignedOutgoing::new(1, Some(3), None)]),
            HashSet::from_iter(vec![BudgetAssignedOutgoing::new(5, None, None)]),
            HashSet::new(),
        ];

        validate_budgets(
            &budgets,
            &budget_assigned_instrs,
            &expected_budgets,
            &expected_outgoings,
        );
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

        let mut budget_checked_instrs = budget_check_analysis(&mut jp_analysed_instrs);

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

        let (budgets, budget_assigned_instrs) = budget_assignment(&mut budget_checked_instrs);

        let expected_budgets = vec![6, 5, 4, 3, 4, 3, 2, 1];

        let expected_outgoings = vec![
            HashSet::from_iter(vec![BudgetAssignedOutgoing::new(1, Some(1), None)]),
            HashSet::from_iter(vec![BudgetAssignedOutgoing::new(2, None, None)]),
            HashSet::from_iter(vec![
                BudgetAssignedOutgoing::new(3, None, None),
                BudgetAssignedOutgoing::new(4, None, Some(4)),
            ]),
            HashSet::from_iter(vec![BudgetAssignedOutgoing::new(6, Some(3), None)]),
            HashSet::from_iter(vec![
                BudgetAssignedOutgoing::new(1, Some(3), Some(5)),
                BudgetAssignedOutgoing::new(5, None, None),
            ]),
            HashSet::from_iter(vec![BudgetAssignedOutgoing::new(6, Some(4), None)]),
            HashSet::from_iter(vec![BudgetAssignedOutgoing::new(7, None, None)]),
            HashSet::new(),
        ];

        validate_budgets(
            &budgets,
            &budget_assigned_instrs,
            &expected_budgets,
            &expected_outgoings,
        );
    }
}
