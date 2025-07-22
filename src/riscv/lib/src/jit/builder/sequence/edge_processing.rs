use std::fmt::Debug;

use cranelift::prelude::Block;

use crate::jit::builder::instruction::LoweredInstruction;
use crate::jit::builder::instruction::Outcome;

/// Represents outgoing transitions that enable hooks to be added.
pub(super) trait HookedOutgoing: Clone + Debug {
    fn new(index: usize, hook: Block) -> Self;

    #[cfg(test)]
    fn index(&self) -> usize;
    #[expect(
        dead_code,
        reason = "This function will be used in later stages of development for converting into further
        enriched types."
    )]
    fn hook(&self) -> Block;
}

/// An outgoing transition from an instruction to another in the sequence with an annotated index and hook.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[expect(
    dead_code,
    reason = "This struct will be used in later stages of development for converting into further
    enriched types."
)]
pub(super) struct IndexedOutgoing {
    /// The index of the next instruction in the sequence.
    index: usize,
    /// The hook to the next block in the sequence.
    hook: Block,
}

impl HookedOutgoing for IndexedOutgoing {
    /// Create a new outgoing transition.
    fn new(index: usize, hook: Block) -> Self {
        Self { index, hook }
    }

    /// Get the index of the next instruction.
    #[cfg(test)]
    fn index(&self) -> usize {
        self.index
    }

    /// Return the hook to the next block in the sequence.
    fn hook(&self) -> Block {
        self.hook
    }
}

/// `EdgeProcessedLI` holds the information after the edge-processing analysis.
/// This provides the incoming and outgoing paths for each instruction, which instructions
/// are join-points, and which instructions are branch-points.
#[derive(Clone, Debug)]
pub(super) struct EdgeProcessedLI<T: HookedOutgoing> {
    /// Original LoweredInstruction
    lowered_instr: LoweredInstruction,

    /// Transitions into this instruction.
    incomings: Vec<usize>,

    /// Intra-sequence transitions from this instruction to another.
    outgoings: Vec<T>,
}

impl<T: HookedOutgoing> EdgeProcessedLI<T> {
    pub(super) fn from_lowered_instr(instr: LoweredInstruction) -> Self {
        Self {
            lowered_instr: instr,
            incomings: Vec::new(),
            outgoings: Vec::new(),
        }
    }

    /// Process the outcomes of the instruction to identify intra-sequence transitions.
    /// Return the indices of the instructions that are targets of these transitions.
    pub(super) fn process_outcomes(&mut self, instrs: &[LoweredInstruction]) -> Vec<usize> {
        let mut outcomes_to_remove: Vec<Outcome> = Vec::new();
        let mut target_indices: Vec<usize> = Vec::new();

        for outcome in self.lowered_instr.outcomes() {
            let result_pc;
            let out_hook;

            match outcome {
                Outcome::Next { hook } => {
                    result_pc = self.lowered_instr.next_instruction_address();
                    out_hook = *hook;
                }

                Outcome::KnownBranch { offset, hook } => {
                    result_pc = self
                        .lowered_instr
                        .program_counter()
                        .wrapping_add_signed(*offset);
                    out_hook = *hook;
                }
                _ => {
                    // Guaranteed not to be an intra-sequence transition.
                    continue;
                }
            }

            // Search the program_counter in the instructions to find the target instruction using a binary search.
            if let Ok(target_idx) =
                instrs.binary_search_by_key(&result_pc, |instr| instr.program_counter())
            {
                // Create an outgoing transition to the target instruction.
                let outgoing = T::new(target_idx, out_hook);

                // Add the outgoing and incoming transitions.
                self.outgoings.push(outgoing);
                target_indices.push(target_idx);

                // Remove the outcome from the list of outcomes for the LoweredInstruction.
                outcomes_to_remove.push(outcome.clone());
            }
        }

        for outcome in outcomes_to_remove {
            self.lowered_instr.remove_outcome(outcome);
        }

        target_indices
    }
}

/// Analyse outcomes over all instructions in a sequence to add incomings and outgoings
/// to each instruction.
#[allow(
    dead_code,
    reason = "This function will be used in later stages of development for converting into further
    enriched types."
)]
pub(crate) fn edge_processing<T: HookedOutgoing>(
    instrs: &[LoweredInstruction],
) -> Vec<EdgeProcessedLI<T>> {
    let n = instrs.len();

    let mut enriched_instrs: Vec<EdgeProcessedLI<T>> = instrs
        .iter()
        .map(|instr| EdgeProcessedLI::from_lowered_instr(instr.clone()))
        .collect();

    for enriched_instr_idx in 0..n {
        let destination_indices = enriched_instrs[enriched_instr_idx].process_outcomes(instrs);

        for destination_idx in destination_indices {
            enriched_instrs[destination_idx]
                .incomings
                .push(enriched_instr_idx);
        }
    }

    enriched_instrs
}

#[cfg(test)]
pub(crate) mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::default::ConstDefault;
    use crate::jit::JIT;
    use crate::jit::builder::sequence::tests::create_lowered_instructions;
    use crate::machine_state::instruction::Args;
    use crate::machine_state::instruction::Instruction;
    use crate::machine_state::instruction::OpCode;
    use crate::machine_state::memory::M4K;
    use crate::machine_state::registers::nz;
    use crate::machine_state::registers::*;
    use crate::parser::instruction::InstrWidth;

    impl<T: HookedOutgoing> EdgeProcessedLI<T> {
        /// Return whether the instruction is a branch point
        pub(super) fn is_branch_point(&self) -> bool {
            self.outgoings.len() > 1
        }

        /// Return the sorted indices of the incomings for this instruction.
        pub(super) fn incoming_indices(&self) -> Vec<usize> {
            let mut incomings: Vec<usize> = self.incomings.iter().map(|i| *i).collect();
            incomings.sort();
            incomings
        }

        /// Return the sorted indices of the outgoings for this instruction.
        pub(super) fn outgoing_indices(&self) -> Vec<usize> {
            let mut outgoings: Vec<usize> = self.outgoings.iter().map(|o| o.index()).collect();
            outgoings.sort();
            outgoings
        }

        /// Return the remaining number of inter-sequence transitions for this instruction.
        pub(super) fn num_remaining_outcomes(&self) -> usize {
            self.lowered_instr.outcomes().len()
        }
    }

    /// A mocked outgoing for testing purposes. The hook is not used in this context.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub(crate) struct MockHookedOutgoing {
        /// The index of the next instruction in the sequence.
        index: usize,
    }

    impl HookedOutgoing for MockHookedOutgoing {
        fn new(index: usize, _hook: Block) -> Self {
            Self { index }
        }

        fn index(&self) -> usize {
            self.index
        }

        fn hook(&self) -> Block {
            Block::from_u32(0) // Placeholder
        }
    }

    /// Validate the results of edge-processing analysis on a sequence of instructions
    /// against expected values.
    pub(crate) fn validate_edge_processing<T: HookedOutgoing>(
        result: &[EdgeProcessedLI<T>],
        expected_incomings: &[Vec<usize>],
        expected_outgoings: &[Vec<usize>],
        expected_remaining_outcomes: &[usize],
        expected_branch_points: &[bool],
    ) {
        for result_index in 0..result.len() {
            assert_eq!(
                result[result_index].incoming_indices(),
                expected_incomings[result_index],
                "Failed to match incomings for instruction {result_index}"
            );

            assert_eq!(
                result[result_index].outgoing_indices(),
                expected_outgoings[result_index],
                "Failed to match outgoings for instruction {result_index}"
            );

            assert_eq!(
                result[result_index].num_remaining_outcomes(),
                expected_remaining_outcomes[result_index],
                "Failed to match remaining outcomes for instruction {result_index}"
            );

            assert_eq!(
                result[result_index].is_branch_point(),
                expected_branch_points[result_index],
                "Failed to match branch-point status for instruction {result_index}"
            );
        }
    }

    /// The number of types of nodes we have for testing sequence graph analysis.
    const NUM_NODES: usize = 13;

    /// (Instruction, outgoings_offsets, number_of_remaining_outcomes)
    /// The set of different nodes available for testing sequence graph analysis.
    ///
    /// `outgoings_offsets` indicates the index offsets of the destinations of outgoing edges.
    /// `number_of_remaining_outcomes` indicates the number of known leftover outcomes for
    /// each node type before analysis.
    const SEQUENCE_ANALYSIS_NODES: [(Instruction, &[i64], usize); NUM_NODES] = [
        (
            Instruction {
                opcode: OpCode::Jr,
                args: Args::DEFAULT,
            },
            &[],
            1,
        ),
        (
            Instruction {
                opcode: OpCode::ECall,
                args: Args::DEFAULT,
            },
            &[],
            1,
        ),
        (
            Instruction {
                opcode: OpCode::X64Add,
                args: Args::DEFAULT,
            },
            &[1],
            0,
        ),
        (
            Instruction {
                opcode: OpCode::X64Sub,
                args: Args::DEFAULT,
            },
            &[1],
            0,
        ),
        (
            Instruction {
                opcode: OpCode::Li,
                args: Args::DEFAULT,
            },
            &[1],
            0,
        ),
        (
            Instruction {
                opcode: OpCode::X64LoadSigned,
                args: Args::DEFAULT,
            },
            &[1],
            1,
        ),
        (
            Instruction {
                opcode: OpCode::X64AtomicLoad,
                args: Args::DEFAULT,
            },
            &[1],
            2,
        ),
        (
            Instruction {
                opcode: OpCode::JumpAndLinkPC,
                args: Args {
                    imm: -8,
                    ..Args::DEFAULT
                },
            },
            &[-2],
            0,
        ),
        (
            Instruction {
                opcode: OpCode::JumpPC,
                args: Args {
                    imm: 8,
                    ..Args::DEFAULT
                },
            },
            &[2],
            0,
        ),
        (
            Instruction {
                opcode: OpCode::JumpPC,
                args: Args {
                    imm: 0,
                    ..Args::DEFAULT
                },
            },
            &[0],
            0,
        ),
        (
            Instruction {
                opcode: OpCode::BranchNotEqual,
                args: Args {
                    imm: -8,
                    ..Args::DEFAULT
                },
            },
            &[1, -2],
            0,
        ),
        (
            Instruction {
                opcode: OpCode::BranchEqual,
                args: Args {
                    imm: 8,
                    ..Args::DEFAULT
                },
            },
            &[1, 2],
            0,
        ),
        (
            Instruction {
                opcode: OpCode::BranchEqualZero,
                args: Args {
                    imm: 0,
                    ..Args::DEFAULT
                },
            },
            &[0, 1],
            0,
        ),
    ];

    fn compute_expected_edge_processing_results(
        instrs: &[LoweredInstruction],
        expected_outgoing_offsets: &[&[i64]],
        expected_remaining_outcomes: &mut [usize],
    ) -> (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<bool>) {
        let mut expected_incomings = vec![Vec::new(); instrs.len()];
        let mut expected_outgoings = vec![Vec::new(); instrs.len()];
        let mut expected_branch_points = vec![false; instrs.len()];

        for (instr_index, offsets) in expected_outgoing_offsets.iter().enumerate() {
            for offset in *offsets {
                let incoming_index = instr_index as i64 + offset;
                if (0..instrs.len() as i64).contains(&incoming_index) {
                    expected_incomings[incoming_index as usize].push(instr_index);
                    expected_outgoings[instr_index].push(incoming_index as usize);
                } else {
                    // Count as a remaining outcome
                    expected_remaining_outcomes[instr_index] += 1;
                }
            }
            expected_branch_points[instr_index] = expected_outgoings[instr_index].len() > 1;
        }

        expected_incomings
            .iter_mut()
            .for_each(|incomings| incomings.sort());
        expected_outgoings
            .iter_mut()
            .for_each(|outgoings| outgoings.sort());

        (
            expected_incomings,
            expected_outgoings,
            expected_branch_points,
        )
    }

    proptest! {
        #[test]
        fn test_edge_processing(seq in proptest::collection::vec(0..NUM_NODES, 1..20)) {
            let mut jit = JIT::<M4K>::new().unwrap();
            let initial_pc = 0;
            let mut sequence_builder = jit.start(initial_pc);

            let mut instrs = Vec::new();
            let mut expected_outgoing_offsets = Vec::new();
            let mut expected_remaining_outcomes = Vec::new();

            for node_index in seq {
                let node = &SEQUENCE_ANALYSIS_NODES[node_index];
                instrs.push(node.0.clone());
                expected_outgoing_offsets.push(node.1);
                expected_remaining_outcomes.push(node.2);
            }

            let lowered_instrs = create_lowered_instructions(&mut sequence_builder, instrs);

            let (expected_incomings, expected_outgoings, expected_branch_points) =
                compute_expected_edge_processing_results(
                    &lowered_instrs,
                    &expected_outgoing_offsets,
                    &mut expected_remaining_outcomes,
                );

            let edge_processed_instrs = edge_processing::<MockHookedOutgoing>(&lowered_instrs);
            validate_edge_processing(
                &edge_processed_instrs,
                &expected_incomings,
                &expected_outgoings,
                &expected_remaining_outcomes,
                &expected_branch_points,
            );
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
    }
}
