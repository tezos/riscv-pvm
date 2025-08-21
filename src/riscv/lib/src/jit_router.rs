use std::collections::LinkedList;
use std::num::NonZeroU64;

use crate::jit::{JIT, JitFn};
use crate::log;
use crate::machine_state::instruction::Instruction;
use crate::machine_state::memory::{Address, MemoryConfig, PAGE_SIZE, Permissions};

pub struct JitRouter<'a, MC: MemoryConfig> {
    default_jit: &'a mut JIT<MC>,
    jits: Vec<JIT<MC>>,
}

impl<'a, MC: MemoryConfig> JitRouter<'a, MC> {
    pub fn new(jit: &'a mut JIT<MC>) -> Self {
        Self {
            default_jit: jit,
            jits: Vec::new(),
        }
    }

    pub fn compile(
        &mut self,
        instr: &[Instruction],
        program_counter: Address,
    ) -> Option<JitFn<MC>> {
        let start_page_index = program_counter / PAGE_SIZE;
        let end_page_index = (program_counter + instr.len() as u64 - 1) / PAGE_SIZE;

        for jit in self.jits.iter_mut() {
            if jit.start_page_index <= start_page_index && jit.end_page_index >= end_page_index {
                return jit.compile(instr, program_counter);
            }
        }

        self.default_jit.compile(instr, program_counter)
    }

    pub fn update_memory(&mut self, addr: u64, length: NonZeroU64, perms: Permissions) {
        let start_page_index = addr / PAGE_SIZE;
        let end_page_index = (addr + length.get() as u64 - 1) / PAGE_SIZE;

        let mut impacted_jit_indices: Vec<usize> = Vec::new();

        for (index, jit) in self.jits.iter().enumerate() {
            if jit.start_page_index <= start_page_index && jit.end_page_index >= end_page_index {
                impacted_jit_indices.push(index);
            }
        }

        if perms.can_read() && perms.can_exec() && !perms.can_exec() {
            if impacted_jit_indices.is_empty() {
                if let Ok(mut new_jit) = JIT::new() {
                    new_jit.update_memory_range(start_page_index, end_page_index);
                    self.jits.push(new_jit);
                } else {
                    log::error!("Failed to create new JIT");
                }
            } else {
                for &index in impacted_jit_indices.iter() {
                    if index == *impacted_jit_indices.first().unwrap() {
                        if let Some(jit) = self.jits.iter_mut().nth(index) {
                            jit.update_memory_range(start_page_index, end_page_index);
                        } else {
                            log::error!("Failed to find JIT at index {}", index);
                        }
                    } else {
                        self.jits.remove(index);
                    }
                }
            }
        } else {
            for &index in impacted_jit_indices.iter() {
                self.jits.remove(index);
            }
        }
    }
}
