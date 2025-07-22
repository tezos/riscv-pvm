use std::fmt::Debug;

use cranelift::prelude::Block;
use cranelift::prelude::FunctionBuilder;
use cranelift::prelude::InstBuilder;

use crate::jit::builder::instr_map::AddrMap;
use crate::jit::builder::instr_map::InstrId;
use crate::jit::builder::instr_map::InstrMap;
use crate::jit::builder::instr_map::InstrMapBuilder;
use crate::jit::builder::instruction::LoweredInstruction;
use crate::jit::builder::instruction::Outcome;
use crate::jit::builder::sequence::InstrInfo;
use crate::parser::instruction::InstrWidth;

/// Sequence Information: Contains metadata on the sequence for analysis.
#[derive(Debug, Clone)]
pub(super) struct SequenceInfo {
    addr_map: AddrMap,
    external_instrs: InstrMap<ExternalInfo>,
    graph: InstrMap<(IncomingInfo, OutgoingInfo)>,
    su_instrs: InstrMap<StepUpdateInfo>,
    bc_instrs: InstrMap<BudgetCheckInfo>,
    budget_instrs: InstrMap<BudgetInfo>,
}

fn resolve_outcomes(
    addr_map: &AddrMap,
    info_map: &InstrMap<InstrInfo>,
    graph_map: &mut InstrMap<(IncomingInfo, OutgoingInfo)>,
    external_instrs: &mut InstrMap<ExternalInfo>,
    instr_id: InstrId,
) {
    let instr_pc = addr_map[instr_id];
    let instr_info = &info_map[instr_id];

    for outcome in instr_info.outcomes() {
        let Some((target_pc, block)) = outcome.to_target(instr_pc, instr_info.width()) else {
            external_instrs[instr_id].insert_outcome(*outcome);
            continue;
        };

        let Some(dest_instr_id) = addr_map.translate(target_pc) else {
            external_instrs[instr_id].insert_outcome(*outcome);
            continue;
        };

        let source_info = &mut graph_map[instr_id].1;
        source_info.num_outgoings += 1;
        source_info.dests.push((dest_instr_id, block));

        let dest_info = &mut graph_map[dest_instr_id].0;
        dest_info.num_incomings += 1;
        dest_info.sources.push(instr_id);
    }
}

impl SequenceInfo {
    pub(super) fn new(instrs: &[LoweredInstruction]) -> Self {
        let mut instr_map_builder = InstrMapBuilder::with_capacity(instrs.len());
        for instr in instrs {
            instr_map_builder.insert(instr.program_counter(), instr);
        }
        let (addr_map, instrs) = instr_map_builder.build();

        let to_graph = {
            (
                IncomingInfo {
                    num_incomings: 0,
                    sources: vec![],
                },
                OutgoingInfo {
                    num_outgoings: 0,
                    dests: vec![],
                },
            )
        };

        let mut graph: InstrMap<(IncomingInfo, OutgoingInfo)> =
            instrs.clone().map(|_, _| to_graph.clone());

        // Add an implicit incoming from outside the sequence for the first instruction.
        graph.get_first_mut().1.0.num_incomings += 1;

        // Add an implicit outgoing to outside the sequence for the last instruction.
        graph.get_last_mut().1.1.num_outgoings += 1;

        let mut basic_instrs = instrs.clone();

        let mut external_instrs = instrs.clone().map(|_, info| ExternalInfo::from(info));

        for (source_instr_id, _) in instrs.iter() {
            resolve_outcomes(
                &addr_map,
                &mut basic_instrs,
                &mut graph,
                &mut external_instrs,
                source_instr_id,
            );
        }

        let su_instrs = graph.clone().map(|_, info| StepUpdateInfo::from(&info.0));
        let bc_instrs = graph.clone().map(|_, info| BudgetCheckInfo::from(&info.1));
        let budget_instrs = graph.clone().map(|_, _| BudgetInfo { budget: None });

        Self {
            addr_map,
            external_instrs,
            graph,
            su_instrs,
            bc_instrs,
            budget_instrs,
        }
    }

    pub(super) fn populate_step_updates(&mut self) {
        let mut su_instrs = self.su_instrs.clone();
        for instr_id in self.graph.iter() {
            calc_step_update(&self.graph, &mut su_instrs, instr_id.0);
        }
        self.su_instrs = su_instrs;
    }

    pub(super) fn populate_budget_checks(&mut self) {
        budget_check(&self.graph, &mut self.bc_instrs);
    }

    pub(super) fn populate_budgets(&mut self) {
        let mut budget_instrs = self.budget_instrs.clone();
        for instr_id in self.graph.iter() {
            budget_search(&self.bc_instrs, &mut budget_instrs, instr_id.0);
        }
        self.budget_instrs = budget_instrs;
    }

    pub(super) fn get_graph(&self) -> &InstrMap<(IncomingInfo, OutgoingInfo)> {
        &self.graph
    }

    pub(super) fn get_addresses(&self) -> &AddrMap {
        &self.addr_map
    }

    pub(super) fn get_external_instrs(&self) -> &InstrMap<ExternalInfo> {
        &self.external_instrs
    }

    pub(super) fn get_su_instrs(&self) -> &InstrMap<StepUpdateInfo> {
        &self.su_instrs
    }

    pub(super) fn get_bc_instrs(&self) -> &InstrMap<BudgetCheckInfo> {
        &self.bc_instrs
    }

    pub(super) fn get_budget_instrs(&self) -> &InstrMap<BudgetInfo> {
        &self.budget_instrs
    }
}

fn calc_step_update(
    graph: &InstrMap<(IncomingInfo, OutgoingInfo)>,
    su_instr_map: &mut InstrMap<StepUpdateInfo>,
    starting_instr_id: InstrId,
) {
    let mut instr_id_stack = vec![starting_instr_id];
    while let Some(curr_instr_id) = instr_id_stack.pop() {
        if su_instr_map[curr_instr_id].step_update.is_some() {
            continue;
        }

        let parent = graph[curr_instr_id].0.sources.first();
        let Some(parent_instr_id) = parent else {
            su_instr_map[curr_instr_id].step_update = Some(0);
            continue;
        };

        if let Some(parent_su) = su_instr_map[*parent_instr_id].step_update {
            su_instr_map[curr_instr_id].step_update = Some(parent_su + 1);
        } else {
            instr_id_stack.push(curr_instr_id);
            instr_id_stack.push(*parent_instr_id);
        }
    }
}

fn budget_check(
    graph: &InstrMap<(IncomingInfo, OutgoingInfo)>,
    bc_instrs: &mut InstrMap<BudgetCheckInfo>,
) {
    for (_, info) in bc_instrs
        .iter_mut()
        .filter(|(instr_id, _)| graph[*instr_id].1.num_outgoings >= 2)
    {
        for outgoing in &mut info.outgoings {
            let mut dest_instr_id = outgoing.0;
            let mut dest_instr = &graph[dest_instr_id];

            while dest_instr.1.num_outgoings == 1 {
                dest_instr_id = graph[dest_instr_id].1.dests.first().unwrap().0;
                dest_instr = &graph[dest_instr_id]
            }

            if dest_instr.1.num_outgoings == 0 {
                // No branch-point in the path of this outgoing edge, so no budget check required.
                outgoing.1 = false;
            } else {
                // Branch-point in the path of this outgoing edge, so budget check required.
                outgoing.1 = true;
            }
        }
    }
}

fn budget_search(
    bc_instrs: &InstrMap<BudgetCheckInfo>,
    budget_instrs: &mut InstrMap<BudgetInfo>,
    starting_instr_id: InstrId,
) {
    let mut stack = vec![starting_instr_id];
    while let Some(current) = stack.pop() {
        let current_budget = budget_instrs[current].budget;
        if current_budget.is_some() {
            continue;
        }

        let child_budgets = bc_instrs[current]
            .outgoings
            .iter()
            .filter(|(_child_instr_id, budget_check, _block)| !budget_check)
            .map(|(child_instr_id, _budget_check, _block)| {
                let budget = budget_instrs[*child_instr_id].budget;
                (child_instr_id, budget)
            });

        // The budget of the current instruction is 1 + the maximum budget among the children of the instruction.
        // If any are still undetermined, we cannot determine the current instruction's budget.
        let max_child_budget =
            child_budgets.fold(Some(0), |curr_max, (_instr_id, budget)| {
                match (curr_max, budget) {
                    (Some(curr_max), Some(budget)) => Some(curr_max.max(budget)),
                    _ => None,
                }
            });

        if let Some(budget) = max_child_budget {
            budget_instrs[current].budget = Some(budget + 1);
            continue;
        }

        stack.push(current);
        for instr_id in bc_instrs[current].outgoings.iter() {
            if !instr_id.1 {
                stack.push(instr_id.0);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct ExternalInfo {
    width: InstrWidth,
    outcomes: Vec<Outcome>,
    run_block: Block,
}

impl From<&InstrInfo> for ExternalInfo {
    fn from(info: &InstrInfo) -> Self {
        Self {
            width: info.width,
            outcomes: Vec::new(),
            run_block: info.run_block,
        }
    }
}

impl ExternalInfo {
    pub(super) fn width(&self) -> InstrWidth {
        self.width
    }

    pub(super) fn insert_outcome(&mut self, outcome: Outcome) {
        self.outcomes.push(outcome);
    }

    pub(super) fn outcomes(&self) -> &[Outcome] {
        &self.outcomes
    }

    /// Build a jump that effectively runs the instruction.
    pub(super) fn build_run(&self, builder: &mut FunctionBuilder) {
        builder.ins().jump(self.run_block, []);
    }
}

#[derive(Debug, Clone)]
pub(super) struct IncomingInfo {
    num_incomings: usize,
    sources: Vec<InstrId>,
}

impl IncomingInfo {
    pub(super) fn num_incomings(&self) -> usize {
        self.num_incomings
    }
}

#[derive(Debug, Clone)]
pub(super) struct OutgoingInfo {
    num_outgoings: usize,
    dests: Vec<(InstrId, Block)>,
}

#[derive(Debug, Clone)]
pub(super) struct StepUpdateInfo {
    step_update: Option<usize>,
}

impl From<&IncomingInfo> for StepUpdateInfo {
    fn from(info: &IncomingInfo) -> Self {
        let step_update = if info.num_incomings >= 2 {
            Some(0)
        } else {
            None
        };

        Self { step_update }
    }
}

impl StepUpdateInfo {
    pub(super) fn step_update(&self) -> Option<usize> {
        self.step_update
    }
}

#[derive(Debug, Clone)]
pub(super) struct BudgetCheckInfo {
    outgoings: Vec<(InstrId, bool, Block)>,
}

impl From<&OutgoingInfo> for BudgetCheckInfo {
    fn from(info: &OutgoingInfo) -> Self {
        Self {
            outgoings: info
                .dests
                .iter()
                .map(|(id, block)| (*id, false, block.clone()))
                .collect(),
        }
    }
}

impl BudgetCheckInfo {
    pub(super) fn outgoings(&self) -> &Vec<(InstrId, bool, Block)> {
        &self.outgoings
    }
}

#[derive(Debug, Clone)]
pub(super) struct BudgetInfo {
    budget: Option<usize>,
}

impl BudgetInfo {
    pub(super) fn budget(&self) -> Option<usize> {
        self.budget
    }
}
