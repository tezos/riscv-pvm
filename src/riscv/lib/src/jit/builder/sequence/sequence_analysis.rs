//! Analysis of instruction sequences to determine control flow within the sequence.
//!
//! `Sequence Analysis` is currently only used in tests and has not yet been integrated into the JIT.
#![cfg(test)]

use std::collections::HashSet;
use std::fmt::Debug;

use cranelift::prelude::Block;

use crate::jit::builder::instr_map::AddrMap;
use crate::jit::builder::instr_map::InstrId;
use crate::jit::builder::instr_map::InstrMap;
use crate::jit::builder::instr_map::InstrMapBuilder;
use crate::jit::builder::instruction::LoweredInstruction;
use crate::jit::builder::instruction::Outcome;

/// Representation of [`Block`] in a sequence graph.
/// This can be customized to use other representations such as
/// a unit type when block information is not needed.
pub trait BlockRepr: Clone + Debug {
    fn from_block(block: Block) -> Self;
}

impl BlockRepr for Block {
    fn from_block(block: Block) -> Self {
        block
    }
}

/// Representation of [`Outcome`] in a sequence graph.
/// This can be customized to use other representations such as
/// a unit type when outcome information is not needed.
pub trait OutcomeRepr: Clone + Debug {
    fn from_outcome(outcome: Outcome) -> Self;
}

impl OutcomeRepr for Outcome {
    fn from_outcome(outcome: Outcome) -> Self {
        outcome
    }
}

/// Instruction information on the incoming edges to an instruction in a sequence graph.
#[derive(Clone, Debug)]
struct IncomingInfo {
    num_incomings: usize,
    sources: Vec<InstrId>,
}

/// Instruction information on the outgoing edges from an instruction in a sequence graph.
#[derive(Clone, Debug)]
struct OutgoingInfo<B: BlockRepr, OC: OutcomeRepr> {
    internal_dests: Vec<(InstrId, B)>,
    external_dests: Vec<OC>,
}

/// Instruction information classifying internal-outgoing edges as simple or recursive.
#[derive(Clone, Debug)]
struct RecursionInfo<B: BlockRepr> {
    simple_transitions: Vec<(InstrId, B)>,
    recursive_transitions: Vec<(InstrId, B)>,
}

/// Information built from sequence analysis for managing intra-sequence jumps.
#[derive(Clone, Debug)]
pub struct SequenceInfo<B: BlockRepr, OC: OutcomeRepr> {
    addr_map: AddrMap,
    graph: InstrMap<(IncomingInfo, OutgoingInfo<B, OC>)>,
    recursions_info: InstrMap<RecursionInfo<B>>,
}

impl<B: BlockRepr, OC: OutcomeRepr> SequenceInfo<B, OC> {
    /// Create a new [`SequenceInfo`] from a slice of lowered instructions.
    pub(super) fn from_lowered_instructions(lowered_instrs: &[LoweredInstruction]) -> Self {
        let mut instr_map_builder = InstrMapBuilder::new();
        for instr in lowered_instrs {
            instr_map_builder.insert(instr.program_counter(), instr.clone());
        }
        let (addr_map, instrs) = instr_map_builder.build();

        let create_graph = {
            (
                IncomingInfo {
                    num_incomings: 0,
                    sources: vec![],
                },
                OutgoingInfo::<B, OC> {
                    internal_dests: vec![],
                    external_dests: vec![],
                },
            )
        };

        let mut graph: InstrMap<(IncomingInfo, OutgoingInfo<B, OC>)> =
            instrs.clone().map(|_, _| create_graph.clone());

        // Add an implicit incoming from outside the sequence for the first instruction.
        graph
            .iter_mut()
            .next()
            .expect("No instructions in map.")
            .1
            .0
            .num_incomings += 1;

        resolve_outcomes::<B, OC>(&addr_map, instrs, &mut graph);

        let create_recursive_info = RecursionInfo {
            simple_transitions: vec![],
            recursive_transitions: vec![],
        };

        let recursions_info: InstrMap<RecursionInfo<B>> =
            graph.clone().map(|_, _| create_recursive_info.clone());

        Self {
            addr_map,
            graph,
            recursions_info,
        }
    }

    /// Populate the recursion data for the sequence graph.
    pub fn populate_recursion_data(&mut self) {
        let mut recursions_info = self.recursions_info.clone();
        let mut instrs_seen = HashSet::new();

        let outgoing_graph = self.graph.clone().map(|_, (_, outgoing)| outgoing.clone());
        for (instr, _) in self.graph.iter() {
            resolve_recursions::<B, OC>(
                &outgoing_graph,
                &mut recursions_info,
                &mut instrs_seen,
                instr,
            );
        }

        self.recursions_info = recursions_info;
    }
}

/// Resolve the outcomes in [`LoweredInstruction`]s to build the
/// [`IncomingInfo`] and [`OutgoingInfo`] for each instruction in the sequence graph.
fn resolve_outcomes<B: BlockRepr, OC: OutcomeRepr>(
    addr_map: &AddrMap,
    info_map: InstrMap<LoweredInstruction>,
    graph_map: &mut InstrMap<(IncomingInfo, OutgoingInfo<B, OC>)>,
) {
    for (instr_id, instr_info) in info_map.iter() {
        let instr_pc = addr_map[instr_id];

        for outcome in instr_info.outcomes() {
            let source_info = &mut graph_map[instr_id].1;

            // Filter to outcomes that can lead to internal transitions.
            // Find the target destination PC for any of these outcomes.
            let Some((target_pc, block)) = outcome.find_target(instr_pc, instr_info.width()) else {
                source_info.external_dests.push(OC::from_outcome(*outcome));
                continue;
            };

            // Translate the target PC to an instruction ID if it is in the sequence.
            let Some(dest_instr_id) = addr_map.translate(target_pc) else {
                source_info.external_dests.push(OC::from_outcome(*outcome));
                continue;
            };

            source_info
                .internal_dests
                .push((dest_instr_id, B::from_block(block)));

            let dest_info = &mut graph_map[dest_instr_id].0;
            dest_info.num_incomings += 1;
            dest_info.sources.push(instr_id);
        }
    }
}

/// Traverse the sequence graph from the current instruction to classify
/// all edges as simple or recursive in the [`RecursionInfo`].
///
/// A recursive edge is one that forms part of a loop in the graph. Only one
/// edge in the loop is classified as recursive, with the rest being simple edges.
fn resolve_recursions<B: BlockRepr, OC: OutcomeRepr>(
    graph: &InstrMap<OutgoingInfo<B, OC>>,
    recursions: &mut InstrMap<RecursionInfo<B>>,
    instrs_seen: &mut HashSet<InstrId>,
    curr_instr: InstrId,
) {
    if !instrs_seen.insert(curr_instr) {
        return;
    }

    let mut paths = vec![vec![curr_instr]];
    while let Some(path) = paths.pop() {
        let Some(source_instr_id) = path.last() else {
            panic!("A path cannot be empty.");
        };

        let source_instr = &graph[*source_instr_id];

        for dest_instr in &source_instr.internal_dests {
            if path.contains(&dest_instr.0) {
                // This instruction has been seen in the current path.
                // Hence, we have a recursion.
                recursions[*source_instr_id]
                    .recursive_transitions
                    .push(dest_instr.clone());
                // Do not explore this path further as it has recursed.
                continue;
            }

            if instrs_seen.insert(dest_instr.0) {
                // We have not seen this destination instruction before.
                // Hence, we must explore this new path.
                let mut new_path = path.clone();
                new_path.push(dest_instr.0);
                paths.push(new_path);
            }

            recursions[*source_instr_id]
                .simple_transitions
                .push(dest_instr.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::collection::vec;
    use proptest::prelude::Strategy;
    use proptest::proptest;

    use super::*;
    use crate::default::ConstDefault;
    use crate::jit::JIT;
    use crate::jit::builder::sequence::tests::create_lowered_instructions;
    use crate::machine_state::instruction::Args;
    use crate::machine_state::instruction::Instruction;
    use crate::machine_state::instruction::OpCode;
    use crate::machine_state::memory::M4K;

    impl BlockRepr for () {
        fn from_block(_: Block) -> Self {}
    }

    impl OutcomeRepr for () {
        fn from_outcome(_: Outcome) -> Self {}
    }

    impl<B: BlockRepr, OC: OutcomeRepr> SequenceInfo<B, OC> {
        fn set(
            addr_map: AddrMap,
            graph: InstrMap<(IncomingInfo, OutgoingInfo<B, OC>)>,
            recursions_info: InstrMap<RecursionInfo<B>>,
        ) -> Self {
            Self {
                addr_map,
                graph,
                recursions_info,
            }
        }
    }

    impl IncomingInfo {
        fn set(num_incomings: usize, sources: Vec<InstrId>) -> Self {
            Self {
                num_incomings,
                sources,
            }
        }
    }

    impl PartialEq for IncomingInfo {
        fn eq(&self, other: &Self) -> bool {
            let mut self_incomings = self.sources.clone();
            let mut other_incomings = other.sources.clone();
            self_incomings.sort_by_key(|id| id.id());
            other_incomings.sort_by_key(|id| id.id());
            self.num_incomings == other.num_incomings && self.sources == other.sources
        }
    }

    impl<B: BlockRepr, OC: OutcomeRepr> OutgoingInfo<B, OC> {
        fn set(internal_dests: Vec<(InstrId, B)>, external_dests: Vec<OC>) -> Self {
            Self {
                internal_dests,
                external_dests,
            }
        }
    }

    impl PartialEq for OutgoingInfo<(), ()> {
        fn eq(&self, other: &Self) -> bool {
            let mut self_internals = self.internal_dests.clone();
            let mut other_internals = other.internal_dests.clone();
            self_internals.sort_by_key(|id| id.0.id());
            other_internals.sort_by_key(|id| id.0.id());

            let mut self_externals = self.external_dests.clone();
            let mut other_externals = other.external_dests.clone();
            self_externals.sort();
            other_externals.sort();

            self_internals == other_internals && self_externals == other_externals
        }
    }

    impl<B: BlockRepr> RecursionInfo<B> {
        fn set(
            simple_transitions: Vec<(InstrId, B)>,
            recursive_transitions: Vec<(InstrId, B)>,
        ) -> Self {
            Self {
                simple_transitions,
                recursive_transitions,
            }
        }
    }

    impl PartialEq for RecursionInfo<()> {
        fn eq(&self, other: &Self) -> bool {
            let mut self_simples = self.simple_transitions.clone();
            let mut other_simples = other.simple_transitions.clone();
            self_simples.sort_by_key(|id| id.0.id());
            other_simples.sort_by_key(|id| id.0.id());

            let mut self_recursives = self.recursive_transitions.clone();
            let mut other_recursives = other.recursive_transitions.clone();
            self_recursives.sort_by_key(|id| id.0.id());
            other_recursives.sort_by_key(|id| id.0.id());

            self_simples == other_simples && self_recursives == other_recursives
        }
    }

    /// The number of instruction examples we have for testing sequence graph analysis.
    const NUM_INSTR_EXAMPLES: usize = 15;

    /// Example instructions for testing sequence graph analysis.
    const INSTR_EXAMPLES: [Instruction; NUM_INSTR_EXAMPLES] = [
        Instruction {
            // one `Outcome::UnkownBranch`
            opcode: OpCode::Jr,
            args: Args::DEFAULT,
        },
        Instruction {
            // one `Outcome::Exception`
            opcode: OpCode::ECall,
            args: Args::DEFAULT,
        },
        Instruction {
            // one `Outcome::Next`
            opcode: OpCode::X64Add,
            args: Args::DEFAULT,
        },
        Instruction {
            // one `Outcome::Next`
            opcode: OpCode::X64Sub,
            args: Args::DEFAULT,
        },
        Instruction {
            // one `Outcome::Next`
            opcode: OpCode::Li,
            args: Args::DEFAULT,
        },
        Instruction {
            // one `Outcome::Next`. one `Outcome::Exception`.
            opcode: OpCode::X64LoadSigned,
            args: Args::DEFAULT,
        },
        Instruction {
            // one `Outcome::Next`. two `Outcome::Exception`.
            opcode: OpCode::X64AtomicLoad,
            args: Args::DEFAULT,
        },
        Instruction {
            // one `Outcome::KnownBranch`.
            opcode: OpCode::JumpAndLinkPC,
            args: Args {
                imm: -8,
                ..Args::DEFAULT
            },
        },
        Instruction {
            // one `Outcome::KnownBranch`.
            opcode: OpCode::JumpPC,
            args: Args {
                imm: 8,
                ..Args::DEFAULT
            },
        },
        Instruction {
            // one `Outcome::KnownBranch`.
            opcode: OpCode::JumpPC,
            args: Args {
                imm: 0,
                ..Args::DEFAULT
            },
        },
        Instruction {
            // one `Outcome::KnownBranch`. one `Outcome::Next`.
            opcode: OpCode::BranchNotEqual,
            args: Args {
                imm: -8,
                ..Args::DEFAULT
            },
        },
        Instruction {
            // one `Outcome::KnownBranch`. one `Outcome::Next`.
            opcode: OpCode::BranchEqual,
            args: Args {
                imm: 8,
                ..Args::DEFAULT
            },
        },
        Instruction {
            // one `Outcome::KnownBranch`. one `Outcome::Next`.
            opcode: OpCode::BranchEqualZero,
            args: Args {
                imm: 0,
                ..Args::DEFAULT
            },
        },
        Instruction {
            // one `Outcome::KnownBranch`.
            opcode: OpCode::JumpPC,
            args: Args {
                imm: 12,
                ..Args::DEFAULT
            },
        },
        Instruction {
            // one `Outcome::KnownBranch`. one `Outcome::Next`.
            opcode: OpCode::BranchEqualZero,
            args: Args {
                imm: -12,
                ..Args::DEFAULT
            },
        },
    ];

    /// This test matches the first example in the design doc.
    #[test]
    fn test_graph_one() {
        let mut jit = JIT::<M4K>::new().unwrap();
        let initial_pc = 0;
        let mut sequence_builder = jit.start(initial_pc);

        let instrs = vec![
            INSTR_EXAMPLES[2],  // add
            INSTR_EXAMPLES[3],  // sub
            INSTR_EXAMPLES[11], // beq +8
            INSTR_EXAMPLES[7],  // JumpAndLinkPC -8
            INSTR_EXAMPLES[5],  // x64loadsigned
            INSTR_EXAMPLES[0],  // jr
        ];
        let lowered_instrs = create_lowered_instructions(&mut sequence_builder, instrs);

        let mut sequence_info = SequenceInfo::<(), ()>::from_lowered_instructions(&lowered_instrs);
        assert_eq!(
            sequence_info.addr_map.addresses().len(),
            lowered_instrs.len()
        );
        assert_eq!(
            sequence_info.graph.instructions().len(),
            lowered_instrs.len()
        );

        sequence_info.populate_recursion_data();
        assert_eq!(
            sequence_info.recursions_info.instructions().len(),
            lowered_instrs.len()
        );

        let expected_incomings = vec![
            IncomingInfo::set(1, vec![]),
            IncomingInfo::set(2, vec![InstrId::set(0), InstrId::set(3)]),
            IncomingInfo::set(1, vec![InstrId::set(1)]),
            IncomingInfo::set(1, vec![InstrId::set(2)]),
            IncomingInfo::set(1, vec![InstrId::set(2)]),
            IncomingInfo::set(1, vec![InstrId::set(4)]),
        ];

        let expected_outgoings: Vec<OutgoingInfo<(), ()>> = vec![
            OutgoingInfo::set(vec![(InstrId::set(1), ())], vec![]),
            OutgoingInfo::set(vec![(InstrId::set(2), ())], vec![]),
            OutgoingInfo::set(vec![(InstrId::set(3), ()), (InstrId::set(4), ())], vec![]),
            OutgoingInfo::set(vec![(InstrId::set(1), ())], vec![]),
            OutgoingInfo::set(vec![(InstrId::set(5), ())], vec![()]),
            OutgoingInfo::set(vec![], vec![()]),
        ];

        let expected_recursions: Vec<RecursionInfo<()>> = vec![
            RecursionInfo::set(vec![(InstrId::set(1), ())], vec![]),
            RecursionInfo::set(vec![(InstrId::set(2), ())], vec![]),
            RecursionInfo::set(vec![(InstrId::set(3), ()), (InstrId::set(4), ())], vec![]),
            RecursionInfo::set(vec![], vec![(InstrId::set(1), ())]),
            RecursionInfo::set(vec![(InstrId::set(5), ())], vec![]),
            RecursionInfo::set(vec![], vec![]),
        ];

        let expected_seq_info = SequenceInfo::set(
            sequence_info.addr_map.clone(),
            InstrMap::set(
                expected_incomings
                    .into_iter()
                    .zip(expected_outgoings)
                    .collect(),
            ),
            InstrMap::set(expected_recursions),
        );

        // test that the generated sequence information matches the expected information.
        for (instr_id, (incoming, outgoing)) in sequence_info.graph.iter() {
            let (expected_incoming, expected_outgoing) = &expected_seq_info.graph[instr_id];
            assert_eq!(
                incoming, expected_incoming,
                "Mismatched incoming at instr_id {instr_id:?}"
            );
            assert_eq!(
                outgoing, expected_outgoing,
                "Mismatched outgoing at instr_id {instr_id:?}"
            );
            let expected_recursion = &expected_seq_info.recursions_info[instr_id];
            let actual_recursion = &sequence_info.recursions_info[instr_id];
            assert_eq!(
                actual_recursion, expected_recursion,
                "Mismatched recursion at instr_id {instr_id:?}"
            );
        }
    }

    /// This test matches the second example in the design doc.
    #[test]
    fn test_graph_two() {
        let mut jit = JIT::<M4K>::new().unwrap();
        let initial_pc = 0;
        let mut sequence_builder = jit.start(initial_pc);

        let instrs = vec![
            INSTR_EXAMPLES[2],  // add
            INSTR_EXAMPLES[3],  // sub
            INSTR_EXAMPLES[11], // beq +8
            INSTR_EXAMPLES[13], // JumpAndLinkPC +12
            INSTR_EXAMPLES[14], // beq -12
            INSTR_EXAMPLES[5],  // x64loadsigned
            INSTR_EXAMPLES[2],  // add
            INSTR_EXAMPLES[3],  // sub
        ];
        let lowered_instrs = create_lowered_instructions(&mut sequence_builder, instrs);

        let mut sequence_info = SequenceInfo::<(), ()>::from_lowered_instructions(&lowered_instrs);
        assert_eq!(
            sequence_info.addr_map.addresses().len(),
            lowered_instrs.len()
        );
        assert_eq!(
            sequence_info.graph.instructions().len(),
            lowered_instrs.len()
        );

        sequence_info.populate_recursion_data();
        assert_eq!(
            sequence_info.recursions_info.instructions().len(),
            lowered_instrs.len()
        );

        let expected_incomings = vec![
            IncomingInfo::set(1, vec![]),
            IncomingInfo::set(2, vec![InstrId::set(0), InstrId::set(4)]),
            IncomingInfo::set(1, vec![InstrId::set(1)]),
            IncomingInfo::set(1, vec![InstrId::set(2)]),
            IncomingInfo::set(1, vec![InstrId::set(2)]),
            IncomingInfo::set(1, vec![InstrId::set(4)]),
            IncomingInfo::set(2, vec![InstrId::set(3), InstrId::set(5)]),
            IncomingInfo::set(1, vec![InstrId::set(6)]),
        ];

        let expected_outgoings: Vec<OutgoingInfo<(), ()>> = vec![
            OutgoingInfo::set(vec![(InstrId::set(1), ())], vec![]),
            OutgoingInfo::set(vec![(InstrId::set(2), ())], vec![]),
            OutgoingInfo::set(vec![(InstrId::set(3), ()), (InstrId::set(4), ())], vec![]),
            OutgoingInfo::set(vec![(InstrId::set(6), ())], vec![]),
            OutgoingInfo::set(vec![(InstrId::set(1), ()), (InstrId::set(5), ())], vec![]),
            OutgoingInfo::set(vec![(InstrId::set(6), ())], vec![()]),
            OutgoingInfo::set(vec![(InstrId::set(7), ())], vec![]),
            OutgoingInfo::set(vec![], vec![()]),
        ];

        let expected_recursions: Vec<RecursionInfo<()>> = vec![
            RecursionInfo::set(vec![(InstrId::set(1), ())], vec![]),
            RecursionInfo::set(vec![(InstrId::set(2), ())], vec![]),
            RecursionInfo::set(vec![(InstrId::set(3), ()), (InstrId::set(4), ())], vec![]),
            RecursionInfo::set(vec![(InstrId::set(6), ())], vec![]),
            RecursionInfo::set(vec![(InstrId::set(5), ())], vec![(InstrId::set(1), ())]),
            RecursionInfo::set(vec![(InstrId::set(6), ())], vec![]),
            RecursionInfo::set(vec![(InstrId::set(7), ())], vec![]),
            RecursionInfo::set(vec![], vec![]),
        ];

        let expected_seq_info = SequenceInfo::set(
            sequence_info.addr_map.clone(),
            InstrMap::set(
                expected_incomings
                    .into_iter()
                    .zip(expected_outgoings)
                    .collect(),
            ),
            InstrMap::set(expected_recursions),
        );

        // test that the generated sequence information matches the expected information.
        for (instr_id, (incoming, outgoing)) in sequence_info.graph.iter() {
            let (expected_incoming, expected_outgoing) = &expected_seq_info.graph[instr_id];
            assert_eq!(
                incoming, expected_incoming,
                "Mismatched incoming at instr_id {instr_id:?}"
            );
            assert_eq!(
                outgoing, expected_outgoing,
                "Mismatched outgoing at instr_id {instr_id:?}"
            );

            let expected_recursion = &expected_seq_info.recursions_info[instr_id];
            let actual_recursion = &sequence_info.recursions_info[instr_id];
            assert_eq!(
                actual_recursion, expected_recursion,
                "Mismatched recursion at instr_id {instr_id:?}"
            );
        }
    }

    proptest! {
        #[test]
        fn test_sequence_information_initialisation(seq in proptest::collection::vec(0..NUM_INSTR_EXAMPLES, 1..20)) {
            let mut jit = JIT::<M4K>::new().unwrap();
            let initial_pc = 0;
            let mut sequence_builder = jit.start(initial_pc);

            let mut instrs = Vec::new();
            for index in seq.iter() {
                instrs.push(INSTR_EXAMPLES[*index]);
            }
            let lowered_instrs = create_lowered_instructions(&mut sequence_builder, instrs);

            // Check resulting sequence information.
            let sequence_info = SequenceInfo::<(), ()>::from_lowered_instructions(&lowered_instrs);
            assert_eq!(sequence_info.addr_map.addresses().len(), lowered_instrs.len());
            assert_eq!(sequence_info.graph.instructions().len(), lowered_instrs.len());

            // Check that the number of outcomes matches the number of outgoings recorded.
            for (instr_id, instr_info) in sequence_info.graph.iter() {
                let num_outcomes = lowered_instrs[instr_id.id()].outcomes().len();
                let num_outgoings = instr_info.1.internal_dests.len() + instr_info.1.external_dests.len();
                assert_eq!(num_outcomes, num_outgoings);
            }

            // Check that the total number of incomings matches the total number of internal outgoings.
            let (total_incomings, total_outgoings): (usize, usize) = sequence_info.graph.iter()
                .map(|(_, (incoming, outgoing))| (incoming.num_incomings, outgoing.internal_dests.len()))
                .fold((0, 0), |(sum_incomings, sum_outgoings), (num_incomings, num_outgoings)| {
                    (sum_incomings + num_incomings, sum_outgoings + num_outgoings)
                });
            assert_eq!(total_incomings, total_outgoings + 1); // +1 for the implicit incoming to the first instruction.
        }
    }

    fn outgoing_info_strategy(size: usize) -> impl Strategy<Value = OutgoingInfo<(), ()>> {
        vec(0..size, 0..3).prop_map(|node_indices| {
            let internal_dests = node_indices
                .iter()
                .map(|idx| (InstrId::set(*idx), ()))
                .collect();
            OutgoingInfo {
                internal_dests,
                external_dests: Vec::new(),
            }
        })
    }

    fn create_outgoing_instrmap() -> impl Strategy<Value = InstrMap<OutgoingInfo<(), ()>>> {
        (1..10_usize).prop_flat_map(|size| {
            vec(outgoing_info_strategy(size), size..=size).prop_map(move |outgoings| {
                let mut instr_builder = InstrMapBuilder::new();
                for (i, outgoing) in outgoings.into_iter().enumerate() {
                    instr_builder.insert(i as u64, outgoing);
                }
                let (_, instr_map) = instr_builder.build();
                instr_map
            })
        })
    }

    proptest! {
        #[test]
        fn test_recursion(graph in create_outgoing_instrmap()) {
            let create_recursive_info = RecursionInfo::<()> {
                simple_transitions: vec![],
                recursive_transitions: vec![],
            };

            let mut recursions_info: InstrMap<RecursionInfo<()>> =
                graph.clone().map(|_, _| create_recursive_info.clone());

            let mut instrs_seen = HashSet::new();

            for (instr, _) in graph.iter() {
                resolve_recursions(
                    &graph,
                    &mut recursions_info,
                    &mut instrs_seen,
                    instr,
                );
            }

            // ensure every instruction has been seen
            for (instr_id, _) in graph.iter() {
                assert!(instrs_seen.contains(&instr_id));
            }

            // Ensure the total number of simple and recursive transitions matches the total number of internal outgoings.
            let (total_simple, total_recursive, total_outgoings): (usize, usize, usize) = graph.iter()
                .map(|(instr_id, outgoing)| {
                    let rec_info = &recursions_info[instr_id];
                    (rec_info.simple_transitions.len(), rec_info.recursive_transitions.len(), outgoing.internal_dests.len())
                })
                .fold((0, 0, 0), |(sum_simple, sum_recursive, sum_outgoings), (num_simple, num_recursive, num_outgoings)| {
                    (sum_simple + num_simple, sum_recursive + num_recursive, sum_outgoings + num_outgoings)
                });
            assert_eq!(total_simple + total_recursive, total_outgoings);

            for instr in recursions_info.iter() {
                // Ensure no overlap between simple and recursive transitions.
                for simple in &instr.1.simple_transitions {
                    assert!(!instr.1.recursive_transitions.iter().any(|rec| rec.0 == simple.0));
                }
            }
        }
    }
}
