// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! This file defines a framework for parallel-capable host functions.
//! The idea is that we don't exacty need the pvm to always execute deterministically on a step by step
//! basis. Instead it is sufficient for the pvm to be convergent on some state. Of course there
//! must be some canonical step by step deterministic execution as well when required (this can be
//! forced by calling `Pvm::eval_once` repeatedly)

//! So what must a non-deterministic, but convergent execution look like? We want to run `n` steps in
//! parallel for a speedup. We consider the following simplest case in this framework:
//! the case of a guarantee that running these `n` tasks in any order results in the same final
//! state. In other words no task `i` may influence the outcome of task `j`.
//!
//! Going into the implementation details, the above guarantee corresponds exactly to Rust's
//! Ownership model: if each thread gets ownership or borrows the input it reads and the output it
//! write to, then they will be unable to influence each other.
//!
//! This is reflected in the parallel iteration (by reference here) over the input vector using `rayon`
//!
//!
//! If we want to run `n=total_steps` steps in parallel, it is important to run be able run any
//! `m < total_steps=n` steps at a time (when playing a refutation game for example).
//! So until `total_steps` is completed we will not progress the `pc` and repeatedly execute the
//! same instruction. However we are not caught in an infinite loop, progress is made by
//! 1) reducing the remaining steps stored in register `a0`,
//! 2) progressing the input pointers and output pointer
//!
//! The byte array interface between the pvm and kernel through trait `Elem`, is exposed as is by
//! this macro for separation of concern of parsing. So parsing is handled by the user
//! (when defining the `map` input)

#[macro_export]
macro_rules! mk_parallel_host_fn {
    (
        $(#[$m:meta])* $name:ident,
        map: $map:expr,
        inputs: [$(($data:ident, $data_size:expr, $data_reg:expr)),* $(,)?],
        output: ($out_ty:ty, $out_size:expr, $out_reg:expr)
    ) => {

        #[inline]
        fn $name<MC, M>( machine: &mut MachineCoreState<MC, M>, step_bounds: NonZeroUsize) -> TezosCallResult
        where
            MC: MemoryConfig,
            M: ManagerReadWrite,
        {
            // We expect `a0` to contain the number of steps yet to be completed
            let total_steps = machine.hart.xregisters.read(a0);
            let curr_steps = min (total_steps, step_bounds.get() as u64);
            let steps_remaining = total_steps - curr_steps;

            // collect any errors that occur while interacting with the `MachineState`
            let sbi_result: Result<(), SbiError> = try_blocks::try_block! {

                $(
                    let mut $data = vec![[0u8; $data_size]; curr_steps as usize];
                    machine.main_memory
                        .read_all(machine.hart.xregisters.read($data_reg), &mut $data)?;
                )*

                let res: Vec<$out_ty> = ($($data),*)
                    .par_iter()
                    .map($map)
                    .collect();

                machine.main_memory
                    .write_all(machine.hart.xregisters.read($out_reg), &res)?;

                // update the state which is captured just using the registers
                // NB the registers must be marked in+out in the call

                // drop the first arg_len elements of each input
                $(machine.hart.xregisters.write($data_reg,
                    machine.hart.xregisters.read($data_reg).saturating_add($data_size as u64 * curr_steps));)*
                // move the output pointer forward
                machine.hart.xregisters.write($out_reg, machine.hart.xregisters.read($out_reg).saturating_add($out_size as u64 * curr_steps));
            };

            match sbi_result {
                Ok(()) => {
                    let pc_update = if steps_remaining == 0 {
                        ProgramCounterUpdate::Next(ECALL_WIDTH)
                    } else {
                        ProgramCounterUpdate::REMAIN
                    };
                    // write steps remaining to `a0` indirectly as the result
                    // register usage
                    TezosCallResult {status: CallStatus::Success(steps_remaining), pc_update, steps_completed: curr_steps as usize}
                }
                Err(sbi_error) => TezosCallResult::error(sbi_error)
            }
        }

        // We would like to typecheck map like so:
        // let map: impl FnOnce(( $([u8; $data_size], )* )) -> $out_ty = $map;
        // To help provide users with more helpful compile errors

        // But rust doesn't allow `impl Trait` in a let binding. So we typecheck with a dummy
        // identity function
        paste::paste! {
            #[expect(dead_code, reason = "This function exists for type-checking purposes")]
            fn [<typecheck_map_of_$name>]<F>(map: F) -> F
            where
                F: FnOnce(( $([u8; $data_size], )* )) -> $out_ty
            {
                map
            }
        }
    }
}
