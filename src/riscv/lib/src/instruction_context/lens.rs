// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! TODO

use crate::machine_state::memory::MemoryConfig;
use crate::state_backend::ManagerBase;

pub trait LensFocus {
    type Instance<MC: MemoryConfig, M: ManagerBase>;
}

pub trait Lens {
    type Subject: LensFocus;

    type Target: LensFocus;

    fn refer<'a, MC: MemoryConfig, M: ManagerBase + 'a>(
        state: &'a <Self::Subject as LensFocus>::Instance<MC, M>,
    ) -> &'a <Self::Target as LensFocus>::Instance<MC, M>;

    fn refer_mut<'a, MC: MemoryConfig, M: ManagerBase + 'a>(
        state: &'a mut <Self::Subject as LensFocus>::Instance<MC, M>,
    ) -> &'a mut <Self::Target as LensFocus>::Instance<MC, M>;

    fn pointer_offset<MC: MemoryConfig, M: ManagerBase>() -> usize;
}

macro_rules! impl_lens {
    ($vis:vis $name:ident ( $subject:ty => $target:ty ) = $($field:ident).+) => {
        $vis enum $name {}

        impl $crate::instruction_context::lens::Lens for $name {
            type Subject = $subject;

            type Target = $target;

            #[inline]
            fn refer<
                'a,
                MC: $crate::machine_state::memory::MemoryConfig,
                M: $crate::state_backend::ManagerBase + 'a,
            >(
                state: &'a <Self::Subject as $crate::instruction_context::lens::LensFocus>::Instance<MC, M>,
            ) -> &'a <Self::Target as $crate::instruction_context::lens::LensFocus>::Instance<MC, M> {
                &state.$($field).+
            }

            #[inline]
            fn refer_mut<
                'a,
                MC: $crate::machine_state::memory::MemoryConfig,
                M: $crate::state_backend::ManagerBase + 'a,
            >(
                state: &'a mut <$subject as $crate::instruction_context::lens::LensFocus>::Instance<MC, M>
            ) -> &'a mut <Self::Target as $crate::instruction_context::lens::LensFocus>::Instance<MC, M> {
                &mut state.$($field).+
            }

            fn pointer_offset<
                MC: $crate::machine_state::memory::MemoryConfig,
                M: $crate::state_backend::ManagerBase,
            >() -> usize {
                std::mem::offset_of!(
                    <Self::Subject as $crate::instruction_context::lens::LensFocus>::Instance<MC, M>,
                    $($field).+
                )
            }
        }
    };
}

pub(crate) use impl_lens;
