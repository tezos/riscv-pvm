use std::marker::PhantomData;

use super::MachineCoreState;
use super::StepManyResult;
use super::memory::Address;
use super::memory::MemoryConfig;
use crate::state_backend::ManagerBase;
use crate::traps::EnvironException;
use crate::traps::Exception;

pub struct PageCache<MC: MemoryConfig, M: ManagerBase> {
    _pd: PhantomData<(MC, M)>,
}

impl<MC: MemoryConfig, M: ManagerBase> PageCache<MC, M> {
    pub fn new() -> Self {
        Self { _pd: PhantomData }
    }

    pub fn get_block(&mut self, _addr: Address) -> Option<BlockCall<'_, MC, M>> {
        None
    }
}

pub struct BlockCall<'a, MC: MemoryConfig, M: ManagerBase> {
    _pd: PhantomData<&'a (MC, M)>,
}

impl<'a, MC: MemoryConfig, M: ManagerBase> BlockCall<'a, MC, M> {
    pub fn run_block(
        &mut self,
        _core: &mut MachineCoreState<MC, M>,
        //block_builder: &mut B::BlockBuilder,
        _instr_pc: Address,
        _max_steps: usize,
    ) -> StepManyResult<Exception> {
        todo!()
    }
}
