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

    pub(crate) fn hook(&self) -> Block {
        self.hook
    }

    pub(crate) fn get_step_update(&self) -> Option<u64> {
        self.step_update
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

    /// Return whether the instruction is a branch point instruction.
    pub(super) fn is_branch_point(&self) -> bool {
        self.outgoings.len() > 1
    }

    /// Return whether the instruction is at the end of the sequence AND has an outgoing transition.
    pub(super) fn is_non_terminating(&self, is_final: bool) -> bool {
        self.outgoings.len() == 1 && is_final
    }

    /// Return whether the instruction is a terminal instruction.
    pub(super) fn is_terminal(&self) -> bool {
        self.outgoings.is_empty()
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
}
