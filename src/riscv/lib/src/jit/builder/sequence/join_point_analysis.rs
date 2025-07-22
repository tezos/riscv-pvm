use cranelift::prelude::Block;

use crate::jit::builder::instruction::LoweredInstruction;
use crate::jit::builder::sequence::edge_processing::EdgeProcessedLI;
use crate::jit::builder::sequence::edge_processing::HookedOutgoing;

/// An outgoing transition from an instruction to another in the sequence with an annotated index and hook.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct StepUpdatingOutgoing {
    /// The index of the next instruction in the sequence.
    index: usize,
    /// The hook to the next block in the sequence.
    hook: Block,
    /// The required update to the step-count.
    step_update: Option<u64>,
}

impl StepUpdatingOutgoing {
    pub(super) fn from_hooked_outgoing(outgoing: impl HookedOutgoing) -> Self {
        Self {
            index: outgoing.index(),
            hook: outgoing.hook(),
            step_update: None, // Default value, will be set during join-point analysis.
        }
    }

    pub(super) fn set_step_update(&mut self, step_update: u64) {
        self.step_update = Some(step_update);
    }

    pub(super) fn index(&self) -> usize {
        self.index
    }

    #[cfg(test)]
    pub(super) fn new(index: usize, step_update: Option<u64>) -> Self {
        Self {
            index,
            hook: Block::from_u32(0), // Placeholder
            step_update,
        }
    }
}

/// JoinPointAnalysedLI holds the information after the join-point analysis stage.
/// This provides the `steps_since_last_jp` for each instruction.
#[derive(Clone)]
pub(super) struct JoinPointAnalysedLI {
    /// Original LoweredInstruction
    lowered_instr: LoweredInstruction,

    /// Transitions into this instruction.
    incomings: Vec<usize>,

    /// Intra-sequence transitions from this instruction to another.
    outgoings: Vec<StepUpdatingOutgoing>,

    /// Steps since the last join-point.
    steps_since_last_jp: u64,
}

impl<T: HookedOutgoing> From<EdgeProcessedLI<T>> for JoinPointAnalysedLI {
    fn from(instr: EdgeProcessedLI<T>) -> Self {
        Self {
            lowered_instr: instr.lowered_instr().clone(),
            incomings: instr.incomings().to_vec(),
            outgoings: instr
                .outgoings()
                .iter()
                .map(|outgoing| StepUpdatingOutgoing::from_hooked_outgoing(outgoing.clone()))
                .collect(),
            steps_since_last_jp: u64::MAX, // Default value, will be updated in join-point analysis.
        }
    }
}

impl JoinPointAnalysedLI {
    /// Return the original LoweredInstruction.
    #[expect(
        dead_code,
        reason = "Lowered instructions will be required for further stages of analysis."
    )]
    pub(super) fn lowered_instr(&self) -> &LoweredInstruction {
        &self.lowered_instr
    }

    /// Return the list of incomings for the instruction.
    pub(super) fn incomings(&self) -> &[usize] {
        &self.incomings
    }

    /// Return the list of outgoings for the instruction.
    pub(super) fn outgoings(&self) -> &[StepUpdatingOutgoing] {
        &self.outgoings
    }

    /// Return number of steps since the last join-point for the instruction.
    pub(super) fn steps_since_last_jp(&self) -> u64 {
        self.steps_since_last_jp
    }

    /// Return whether the instruction is a join-point.
    /// In this case, a starting instruction is semantically a join point, even with
    /// no intra-sequence incomings, since it will implicitly have an incoming external
    /// transition.
    pub(super) fn is_join_point(&self, instr_index: usize) -> bool {
        self.incomings.len() > 1 || self.incomings.len() == 1 && instr_index == 0
    }

    /// Set steps_since_last_jp for the instruction.
    pub(super) fn set_steps_since_last_jp(&mut self, steps: u64) {
        self.steps_since_last_jp = steps;
    }

    /// Return the number of incomings for the instruction.
    pub(super) fn incomings_count(&self) -> usize {
        self.incomings.len()
    }
}

/// Calculate the number of steps since the last join-point for the instruction at `instr_index`.
fn calc_steps_since_last_jp(instr_index: usize, instrs: &[JoinPointAnalysedLI]) -> u64 {
    // If the jp value of the incoming instruction has already been set, then return it.
    let instr = &instrs[instr_index];

    if instr.steps_since_last_jp() != u64::MAX {
        return instr.steps_since_last_jp();
    }

    let num_incomings = instr.incomings_count();

    // If the instruction has no incomings, it is the first instruction in the sequence.
    // If it has multiple incomings, it is a join point.
    // The first instruction also has an extra implicit incoming for the start of the sequence.
    if num_incomings == 0 || num_incomings > 1 || num_incomings == 1 && instr_index == 0 {
        return 0;
    }

    let incoming = instr.incomings()[0];

    1 + calc_steps_since_last_jp(incoming, instrs)
}

/// Perform the join-point analysis on the edge-processed instructions.
/// This pass enriches the instructions with the number of steps since the last join point.
#[allow(dead_code)]
pub(crate) fn join_point_analysis<T: HookedOutgoing>(
    instrs: &[EdgeProcessedLI<T>],
) -> Vec<JoinPointAnalysedLI> {
    let mut enriched_instrs: Vec<JoinPointAnalysedLI> = instrs
        .iter()
        .map(|instr| JoinPointAnalysedLI::from(instr.clone()))
        .collect();

    for instr_index in 0..instrs.len() {
        let steps_since_last_jp = calc_steps_since_last_jp(instr_index, &enriched_instrs);
        enriched_instrs[instr_index].set_steps_since_last_jp(steps_since_last_jp);

        for outgoing_index in 0..enriched_instrs[instr_index].outgoings().len() {
            let destination_index = enriched_instrs[instr_index].outgoings[outgoing_index].index();
            if enriched_instrs[destination_index].is_join_point(destination_index) {
                enriched_instrs[instr_index].outgoings[outgoing_index]
                    .set_step_update(steps_since_last_jp + 1);
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
    use crate::jit::builder::sequence::tests::create_lowered_instructions;
    use crate::machine_state::instruction::Instruction;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::registers::nz;
    use crate::machine_state::registers::*;
    use crate::parser::instruction::InstrWidth;

    pub(crate) fn validate_jp_analysis(
        result: &[JoinPointAnalysedLI],
        expected_steps: &[u64],
        expected_outgoings: &[HashSet<StepUpdatingOutgoing>],
    ) {
        for instr_index in 0..result.len() {
            assert_eq!(
                result[instr_index].steps_since_last_jp(),
                expected_steps[instr_index],
                "Failed to match steps_since_last_jp for instruction {instr_index}"
            );

            let mut outgoing_set: HashSet<StepUpdatingOutgoing> =
                HashSet::from_iter(result[instr_index].outgoings().iter().cloned());
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

        let jp_analysed_instrs = join_point_analysis::<MockHookedOutgoing>(&edge_processed_instrs);

        let expected_outgoings = vec![
            HashSet::from_iter(vec![StepUpdatingOutgoing::new(1, None)]),
            HashSet::from_iter(vec![StepUpdatingOutgoing::new(2, None)]),
            HashSet::from_iter(vec![StepUpdatingOutgoing::new(3, None)]),
            HashSet::new(),
        ];

        validate_jp_analysis(&jp_analysed_instrs, &[0, 1, 2, 3], &expected_outgoings);
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

        let jp_analysed_instrs = join_point_analysis::<MockHookedOutgoing>(&edge_processed_instrs);

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

        let jp_analysed_instrs = join_point_analysis::<MockHookedOutgoing>(&edge_processed_instrs);

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
    }
}
