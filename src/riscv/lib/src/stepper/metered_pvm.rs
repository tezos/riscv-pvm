// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use crate::machine_state::block_cache::BlockCacheConfig;
use crate::machine_state::memory::MemoryConfig;
use crate::pvm::hooks::NoHooks;
use crate::state_backend::metrics_backend::Metrics;
use crate::stepper::pvm::PvmStepper;

pub struct ProofMetrics {
    // For each step, hold size of the would-be proof for each step
    sizes: Vec<u64>,
}

pub struct MeteredPvm<MC: MemoryConfig, BCC: BlockCacheConfig> {
    pvm: PvmStepper<NoHooks, MC, BCC, Metrics>,
    metrics: ProofMetrics,
}
