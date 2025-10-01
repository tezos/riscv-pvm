// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Tests for the control flow graph analysis

#![cfg(test)]

use std::cell::Cell;
use std::fmt;

use proptest::prelude::Just;
use proptest::prelude::Strategy;
use proptest::prop_assert_eq;
use proptest::prop_oneof;
use proptest::test_runner::TestCaseError;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::jit::builder::control_flow_graph::ControlFlowGraph;
use crate::jit::builder::control_flow_graph::DirectedEdgeInfo;
use crate::jit::builder::control_flow_graph::NodeInfo;
use crate::jit::builder::control_flow_graph::Target;

/// Action associated with a test instruction outcome
#[derive(Clone)]
enum TestInstrPostAction {
    /// Jump relative to the current program counter
    RelativeJump(i64),

    /// Exit the program
    Exit,
}

impl TestInstrPostAction {
    /// Convert the action into a target for the control flow analysis.
    fn to_target(&self, instr_pc: u64) -> Target {
        match self {
            TestInstrPostAction::RelativeJump(offset) => {
                let target = instr_pc.wrapping_add_signed(*offset);
                Target::Known(target)
            }

            TestInstrPostAction::Exit => Target::Unknown,
        }
    }
}

/// Outcome of executing a test instruction
#[derive(Clone)]
struct TestInstrOutcome {
    /// What to do after executing the instruction
    action: TestInstrPostAction,

    /// Amount to increment the step counter by if this outcome is taken
    step_delta: Cell<usize>,

    /// Extra amount to increment the step counter if the program
    /// exits after executing this outcome.
    exit_delta: Cell<usize>,
}

impl TestInstrOutcome {
    /// Convert the outcome into a directed edge for the control flow analysis.
    fn to_edge(&self, instr_pc: u64) -> DirectedEdgeInfo<&'_ Self> {
        DirectedEdgeInfo {
            target: self.action.to_target(instr_pc),
            info: self,
        }
    }
}

impl fmt::Debug for TestInstrOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.action {
            TestInstrPostAction::RelativeJump(offset) => write!(f, "success {offset}"),
            TestInstrPostAction::Exit => write!(f, "exit"),
        }
    }
}

/// Test instruction
#[derive(Clone)]
struct TestInstr {
    outcomes: Vec<TestInstrOutcome>,
}

impl TestInstr {
    /// Construct an instruction that always goes to the next instruction.
    fn next() -> Self {
        Self {
            outcomes: vec![TestInstrOutcome {
                action: TestInstrPostAction::RelativeJump(1),
                step_delta: Cell::new(1),
                exit_delta: Cell::new(0),
            }],
        }
    }

    /// Construct an instruction that either goes to the next instruction or exits.
    fn next_or_exit() -> Self {
        Self {
            outcomes: vec![
                TestInstrOutcome {
                    action: TestInstrPostAction::RelativeJump(1),
                    step_delta: Cell::new(1),
                    exit_delta: Cell::new(0),
                },
                TestInstrOutcome {
                    action: TestInstrPostAction::Exit,
                    step_delta: Cell::new(0),
                    exit_delta: Cell::new(0),
                },
            ],
        }
    }

    /// Construct an instruction that either branches to the given offset or goes to the next
    /// instruction.
    fn branch_or_next(offset: i64) -> Self {
        Self {
            outcomes: vec![
                TestInstrOutcome {
                    action: TestInstrPostAction::RelativeJump(1),
                    step_delta: Cell::new(1),
                    exit_delta: Cell::new(0),
                },
                TestInstrOutcome {
                    action: TestInstrPostAction::RelativeJump(offset),
                    step_delta: Cell::new(1),
                    exit_delta: Cell::new(0),
                },
            ],
        }
    }

    /// Construct an instruction that always jumps to the given offset.
    fn jump(offset: i64) -> Self {
        Self {
            outcomes: vec![TestInstrOutcome {
                action: TestInstrPostAction::RelativeJump(offset),
                step_delta: Cell::new(1),
                exit_delta: Cell::new(0),
            }],
        }
    }

    /// Run the instruction, randomly selecting one of the possible outcomes.
    fn run(&self, rng: &mut impl Rng) -> &TestInstrOutcome {
        let index = rng.random_range(0..self.outcomes.len());
        &self.outcomes[index]
    }
}

impl fmt::Debug for TestInstr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let options = self
            .outcomes
            .iter()
            .map(|o| format!("{o:?}"))
            .collect::<Vec<_>>()
            .join(" | ");
        write!(f, "{options}")
    }
}

/// Property-based test strategy for a test instruction
fn instruction_strat(program_width: u64) -> impl Strategy<Value = TestInstr> {
    let lower_bound = -(program_width as i64);
    let upper_bound = program_width as i64;
    let branch_range = -10.max(lower_bound)..=10.min(upper_bound);
    let jump_range = lower_bound..=upper_bound;

    prop_oneof![
       16 => Just(TestInstr::next()),
       4 => branch_range.prop_map(TestInstr::branch_or_next),
       4 => jump_range.prop_map(TestInstr::jump),
       1 => Just(TestInstr::next_or_exit()),
    ]
}

/// Property-based test strategy for a sequence of test instructions
fn program_strat() -> impl Strategy<Value = Vec<TestInstr>> {
    (1..=100u64).prop_flat_map(|program_width| {
        proptest::collection::vec(
            instruction_strat(program_width),
            1..=(program_width as usize),
        )
    })
}

/// Evaluate a program consisting of test instructions.
///
/// # Arguments
///
/// * `seed`: Seed for the random number generator to ensure consistent execution
/// * `pc`: Initial program counter
/// * `start`: Address of the first instruction in the program
/// * `program`: Sequence of instructions to execute
/// * `max_steps`: Maximum number of steps to execute (not exact, only to stop runaway programs)
fn evaluate_program(
    seed: u64,
    mut pc: u64,
    start: u64,
    program: &[TestInstr],
    max_steps: usize,
) -> (usize, u64) {
    let mut steps = 0;
    let mut exit_steps = 0;
    let mut rng = StdRng::seed_from_u64(seed);

    // We use a conditional loop break to not mess with how step counters are updated. For example,
    // if we are exact about the number of steps being run, we might exit the sequence earlier than
    // the step counting analysis expects. JIT programs can't exit at arbitrary instructions, so
    // this matches real-world behavior with the exception that JIT functions don't exceed the
    // maximum step counter.
    while steps < max_steps {
        let index = pc.wrapping_sub(start);

        if index >= program.len() as u64 {
            break;
        }

        let instr = &program[index as usize];
        let outcome = instr.run(&mut rng);

        steps = steps
            .checked_add(outcome.step_delta.get())
            .expect("step counting should not overflow");

        exit_steps = outcome.exit_delta.get();

        match outcome.action {
            TestInstrPostAction::RelativeJump(offset) => pc = pc.wrapping_add_signed(offset),
            TestInstrPostAction::Exit => break,
        }
    }

    steps += exit_steps;

    (steps, pc)
}

/// Run a program with sparse step counting.
fn run_sparse_program(
    seed: u64,
    start: u64,
    program: &[TestInstr],
    max_steps: usize,
) -> (usize, u64) {
    // Analysis is performed on a slightly different representation of the program. So we need
    // to transform it.
    let infos = program
        .iter()
        .enumerate()
        .map(|(index, instr)| {
            let instr_pc = start.wrapping_add(index as u64);

            let outgoing = instr
                .outcomes
                .iter()
                .map(|outcome| outcome.to_edge(instr_pc))
                .collect();

            NodeInfo {
                location: instr_pc,
                is_entrypoint: index == 0,
                outgoing,
            }
        })
        .collect::<Vec<_>>();

    let graph = ControlFlowGraph::new(infos.iter());
    let step_updates = graph.find_step_counter_updates();

    // Ensure there is no step counting by default in the sparse program. The goal is to insert
    // the minimum number of step counter operations to get accurate step counting.
    for instr in program.iter() {
        for outcome in instr.outcomes.iter() {
            outcome.step_delta.set(0);
        }
    }

    for (_, outcome) in step_updates.iter() {
        let Some(update) = outcome.data() else {
            // The analysis determined there is nothing to do for this edge.
            continue;
        };

        let outcome = update.edge().info;

        let step_delta = match outcome.action {
            TestInstrPostAction::RelativeJump(_) => update.success_delta(),
            TestInstrPostAction::Exit => update.exception_delta(),
        };

        outcome.step_delta.set(step_delta);
        outcome.exit_delta.set(update.exit_delta);
    }

    evaluate_program(seed, start, start, program, max_steps)
}

/// This test generates random programs and ensures that sparse step counting (i.e. not for every
/// instruction) produces accurate step counts. We validate this by running the same program with
/// and without sparse step counting and ensuring that the results match.
#[test]
fn random_program_step_counting() {
    // The `proptest!` macro prevents formatting. Defining this function outside of the macro means
    // we get formatting. We call this function from inside the macro.
    fn inner(
        seed: u64,
        start: u64,
        program: Vec<TestInstr>,
        max_steps: usize,
    ) -> Result<(), TestCaseError> {
        let sparse_program = program.clone();
        let (sparse_steps, sparse_pc) = run_sparse_program(seed, start, &sparse_program, max_steps);

        // We can't pass `max_steps` to the non-sparse evaluation mode. Step counting on its own
        // does not protect against overrunning the maximum step counter. Generally, with sparse
        // step counting and step budget checks, we can only guarantee that step counting is
        // accurate and we don't overrun the step budget. Performing a precise number of steps
        // is not universally possible.
        let (steps, pc) = evaluate_program(seed, start, start, &program, sparse_steps);

        prop_assert_eq!(steps, sparse_steps);
        prop_assert_eq!(pc, sparse_pc);

        Ok(())
    }

    proptest::proptest! {
        |(
            seed: u64,
            start: u64,
            program in program_strat(),
            max_steps in 1..1000usize,
        )| {
            inner(seed, start, program, max_steps)?;
        }
    }
}

#[test]
fn finishes_on_entry_recursion() {
    let (steps, pc) = run_sparse_program(
        0,
        0x1000u64,
        &[TestInstr::next(), TestInstr::next(), TestInstr::jump(-2)],
        100,
    );

    assert_eq!(steps, 102);
    assert_eq!(pc, 0x1000);
}

#[test]
fn finishes_on_mid_sequence_recursion() {
    let program = [TestInstr::next(), TestInstr::next(), TestInstr::jump(-1)];

    // perform analysis and run the program.
    let (steps, pc) = run_sparse_program(0, 0x1000u64, &program, 100);

    // Ensure the step counting analysis inserted the expected step counter updates.
    // In this case, we only expect a step counter update on the jump back to the previous
    // instruction.
    assert_eq!(program[0].outcomes[0].step_delta.get(), 0);
    assert_eq!(program[1].outcomes[0].step_delta.get(), 0);
    assert_eq!(program[2].outcomes[0].step_delta.get(), 2);

    assert_eq!(steps, 101);
    assert_eq!(pc, 0x1001);
}
