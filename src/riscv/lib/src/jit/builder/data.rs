// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use cranelift::prelude::FunctionBuilder;
use cranelift::prelude::InstBuilder;
use cranelift_jit::JITModule;
use cranelift_module::DataDescription;
use cranelift_module::DataId;
use cranelift_module::Module;
use cranelift_module::ModuleResult;

use crate::jit::builder::typed::Pointer;

/// Define data in the JIT module. The contents of are taken from the provided value.
fn define_module_data<T: Copy + Sized>(module: &mut JITModule, value: &T) -> ModuleResult<DataId> {
    let contents = {
        let mut data = vec![0u8; size_of::<T>()].into_boxed_slice();
        let ptr = (value as *const T).cast::<u8>();

        unsafe {
            data.as_mut_ptr().copy_from_nonoverlapping(ptr, data.len());
        }

        data
    };

    let mut desc = DataDescription::new();
    desc.set_align(align_of::<T>() as u64);
    desc.define(contents);

    let data_id = module.declare_anonymous_data(false, false)?;
    module.define_data(data_id, &desc)?;

    Ok(data_id)
}

/// Define data in the JIT module and reference it in the current function.
pub fn define_function_data<T: Copy + Sized>(
    module: &mut JITModule,
    builder: &mut FunctionBuilder,
    value: &T,
) -> ModuleResult<Pointer<T>> {
    let data_id = define_module_data(module, value)?;
    let global_value = module.declare_data_in_func(data_id, builder.func);

    let raw_value = builder
        .ins()
        .global_value(module.target_config().pointer_type(), global_value);
    let value = unsafe { Pointer::<T>::from_raw(raw_value) };

    Ok(value)
}
