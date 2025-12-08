// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Like [`super::branch`] but for various branch sizes
//!
//! Introducing more types instead of composing [`BuddyBranch2`]/[`BuddyBranch2Config`] makes type
//! checking much faster.

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::Suspended;
use octez_riscv_data::merkle_proof::SuspendedResult;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;

use super::Buddy;
use super::BuddyConfig;
use super::branch::BuddyBranch2;
use super::branch::BuddyBranch2Config;
use crate::state::NewState;
use crate::state_backend::ManagerAlloc;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerClone;
use crate::state_backend::ManagerDeserialise;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerSerialise;
use crate::state_backend::ManagerWrite;

/// Generate a new combined Buddy branch.
macro_rules! combined_buddy_branch {
    ($name:ident = $buddy1:ident * $buddy2:ident) => {
        paste::paste! {
            /// Allocated combined Buddy branch
            #[perfect_derive::perfect_derive(PartialEq, Eq)]
            pub struct [<$name Alloc>]<B: BuddyConfig, M: ManagerBase>(
                <[<$buddy1 Config>]<[<$buddy2 Config>]<B>> as BuddyConfig>::Buddy<M>
            );

            // Passthrough implementation, default derive macro can't derive this ...
            impl<B, M> Encode for [<$name Alloc>]<B, M>
            where
                B: BuddyConfig,
                M: ManagerSerialise,
                <[<$buddy1 Config>]<[<$buddy2 Config>]<B>> as BuddyConfig>::Buddy<M>: Encode,
            {

                fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
                    Encode::encode(&self.0, encoder)
                }
            }

            // Passthrough implementation, default derive macro can't derive this ...
            impl<B, M> Decode<()> for [<$name Alloc>]<B, M>
            where
                B: BuddyConfig,
                M: ManagerDeserialise,
                <[<$buddy1 Config>]<[<$buddy2 Config>]<B>> as BuddyConfig>::Buddy<M>: Decode<()>,
            {
                fn decode<D: Decoder<Context = ()>>(decoder: &mut D) -> Result<Self, DecodeError> {
                    Ok(Self(Decode::decode(decoder)?))
                }
            }

            /// Config for a combined Buddy branch
            pub struct [<$name Config>]<B>(B);

            impl<B: BuddyConfig> BuddyConfig for [<$name Config>]<B> {
                type Buddy<M: ManagerBase> = $name<B::Buddy<M>, M>;

                fn start_proof(instance: &Self::Buddy<Normal>) -> Self::Buddy<Prove<'_>> {
                    let inner = <[<$buddy1 Config>]<[<$buddy2 Config>]<B>> as BuddyConfig>::start_proof(&instance.0);
                    $name(inner)
                }

                fn buddy_from_proof<D: Deserialiser>(proof: D) -> SuspendedResult<D, Self::Buddy<Verify>> {
                    let result = <[<$buddy1 Config>]<[<$buddy2 Config>]<B>> as BuddyConfig>::buddy_from_proof(proof)?;
                    let result = result.map(|inner| $name(inner));
                    Ok(result)
                }
            }
        }

        /// Combined Buddy branch
        #[perfect_derive::perfect_derive(PartialEq, Eq)]
        pub struct $name<B, M: ManagerBase>($buddy1<$buddy2<B, M>, M>);

        impl<B: Buddy<M>, M: ManagerSerialise> Encode for $name<B, M> {
            fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
                Buddy::encode(&self.0, encoder)
            }
        }

        impl<C, B: Buddy<M>, M: ManagerDeserialise> Decode<C> for $name<B, M> {
            fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
                Ok(Self(Buddy::decode(decoder)?))
            }
        }

        impl<B, M> NewState<M> for $name<B, M>
        where
            B: Buddy<M>,
            M: ManagerBase,
        {
            fn new() -> Self
            where
                M: ManagerAlloc,
            {
                Self(NewState::new())
            }
        }

        impl<B, M> Buddy<M> for $name<B, M>
        where
            B: Buddy<M>,
            M: ManagerBase,
        {
            const PAGES: u64 = <$buddy1<$buddy2<B, M>, M> as Buddy<M>>::PAGES;

            fn allocate(&mut self, pages: u64) -> Option<u64>
            where
                M: ManagerRead + ManagerWrite,
            {
                self.0.allocate(pages)
            }

            fn allocate_fixed(&mut self, idx: u64, pages: u64, replace: bool) -> Option<()>
            where
                M: ManagerRead + ManagerWrite,
            {
                self.0.allocate_fixed(idx, pages, replace)
            }

            fn deallocate(&mut self, idx: u64, pages: u64)
            where
                M: ManagerRead + ManagerWrite,
            {
                self.0.deallocate(idx, pages)
            }

            fn longest_free_sequence(&self) -> u64
            where
                M: ManagerRead,
            {
                self.0.longest_free_sequence()
            }

            fn count_free_start(&self) -> u64
            where
                M: ManagerRead,
            {
                self.0.count_free_start()
            }

            fn count_free_end(&self) -> u64
            where
                M: ManagerRead,
            {
                self.0.count_free_end()
            }

            fn clone_state(&self) -> Self
            where
                M: ManagerClone,
            {
                $name(self.0.clone_state())
            }

            fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError>
            where
                M: ManagerSerialise,
            {
                Encode::encode(self, encoder)
            }

            fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError>
            where
                M: ManagerDeserialise,
            {
                Decode::decode(decoder)
            }
        }

        impl<B, M, F> Foldable<F> for $name<B, M>
        where
            M: ManagerBase,
            F: Fold,
            $buddy1<$buddy2<B, M>, M>: Foldable<F>,
        {
            fn fold(&self, builder: F) -> F::Folded {
                self.0.fold(builder)
            }
        }
    };
}

combined_buddy_branch!(BuddyBranch4 = BuddyBranch2 * BuddyBranch2);
combined_buddy_branch!(BuddyBranch8 = BuddyBranch4 * BuddyBranch2);
combined_buddy_branch!(BuddyBranch16 = BuddyBranch4 * BuddyBranch4);
combined_buddy_branch!(BuddyBranch32 = BuddyBranch4 * BuddyBranch8);
combined_buddy_branch!(BuddyBranch64 = BuddyBranch8 * BuddyBranch8);
combined_buddy_branch!(BuddyBranch128 = BuddyBranch16 * BuddyBranch8);
combined_buddy_branch!(BuddyBranch256 = BuddyBranch16 * BuddyBranch16);
combined_buddy_branch!(BuddyBranch1Ki = BuddyBranch32 * BuddyBranch32);
combined_buddy_branch!(BuddyBranch256Ki = BuddyBranch256 * BuddyBranch1Ki);
