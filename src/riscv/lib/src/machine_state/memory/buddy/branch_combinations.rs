// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Like [`super::branch`] but for various branch sizes
//!
//! Introducing more types instead of composing [`BuddyBranch2`]/[`BuddyBranch2Layout`] makes type
//! checking much faster.

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashError;

use super::Buddy;
use super::BuddyLayout;
use super::branch::BuddyBranch2;
use super::branch::BuddyBranch2Layout;
use crate::state::NewState;
use crate::state_backend::AllocatedOf;
use crate::state_backend::FnManager;
use crate::state_backend::Layout;
use crate::state_backend::ManagerAlloc;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerClone;
use crate::state_backend::ManagerDeserialise;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerSerialise;
use crate::state_backend::ManagerWrite;
use crate::state_backend::PartialHashError;
use crate::state_backend::ProofLayout;
use crate::state_backend::ProofTree;
use crate::state_backend::RefProveAlloc;
use crate::state_backend::RefVerifierAlloc;
use crate::state_backend::proof_backend::merkle::MerkleTree;
use crate::state_backend::proof_backend::proof::deserialiser::Deserialiser;

/// Generate a new combined Buddy branch.
macro_rules! combined_buddy_branch {
    ($name:ident = $buddy1:ident * $buddy2:ident) => {
        paste::paste! {
            /// Allocated combined Buddy branch
            #[perfect_derive::perfect_derive(PartialEq, Eq)]
            pub struct [<$name Alloc>]<B: Layout, M: ManagerBase>(AllocatedOf<[<$buddy1 Layout>]<[<$buddy2 Layout>]<B>>, M>);

            // Passthrough implementation, default derive macro can't derive this ...
            impl<B, M> Encode for [<$name Alloc>]<B, M>
            where
                B: Layout,
                M: ManagerSerialise,
                AllocatedOf<[<$buddy1 Layout>]<[<$buddy2 Layout>]<B>>, M>: Encode,
            {

                fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
                    self.0.encode(encoder)
                }
            }


            // Passthrough implementation, default derive macro can't derive this ...
            impl<B, M> Decode<()> for [<$name Alloc>]<B, M>
            where
                B: Layout,
                M: ManagerDeserialise,
                AllocatedOf<[<$buddy1 Layout>]<[<$buddy2 Layout>]<B>>, M>: Decode<()>,
            {
                fn decode<D: Decoder<Context = ()>>(decoder: &mut D) -> Result<Self, DecodeError> {
                    Ok(Self(Decode::decode(decoder)?))
                }
            }

            // NOTE: We can't use `struct_layout!` to help us define the type and impls below. The
            // macro doesn't define new type aliases for the combined layouts which results in a
            // massive type tree to traverse. This makes type checking super slow.

            /// Layout for a combined Buddy branch
            pub struct [<$name Layout>]<B>(B);

            impl<B: Layout> Layout for [<$name Layout>]<B> {
                type Allocated<M: ManagerBase> = [<$name Alloc>]<B, M>;
            }

            impl<B: ProofLayout> ProofLayout for [<$name Layout>]<B>
            where
                [<$buddy1 Layout>]<[<$buddy2 Layout>]<B>>: ProofLayout
            {
                fn to_merkle_tree<'outer, 'inner: 'outer>(
                    state: RefProveAlloc<'outer, 'inner, Self>,
                ) -> Result<MerkleTree, HashError> {
                    <[<$buddy1 Layout>]<[<$buddy2 Layout>]<B>>>::to_merkle_tree(state.0)
                }

                fn into_verifier_alloc<D: Deserialiser>(proof: D) -> $crate::state_backend::VerifierAllocResult<D, Self> {
                    use $crate::state_backend::proof_backend::proof::deserialiser::Suspended;

                    let inner = <[<$buddy1 Layout>]<[<$buddy2 Layout>]<B>>>::into_verifier_alloc(proof)?;
                    Ok(inner.map(|inner| [<$name Alloc>](inner)))
                }

                fn partial_state_hash(
                    state: RefVerifierAlloc<Self>,
                    proof: ProofTree,
                ) -> Result<Hash, PartialHashError> {
                    <[<$buddy1 Layout>]<[<$buddy2 Layout>]<B>>>::partial_state_hash(state.0, proof)
                }
            }

            impl<B: BuddyLayout> BuddyLayout for [<$name Layout>]<B>
                where [<$buddy1 Layout>]<[<$buddy2 Layout>]<B>>: 'static,
            {
                type Buddy<M: ManagerBase> = $name<B::Buddy<M>, M>;

                fn bind<M: ManagerBase>(space: Self::Allocated<M>) -> Self::Buddy<M> {
                    let inner = <[<$buddy1 Layout>]<[<$buddy2 Layout>]<B>> as BuddyLayout>::bind(space.0);
                    $name(inner)
                }

                fn struct_ref<'a, F, M: ManagerBase>(space: &'a Self::Buddy<M>) -> Self::Allocated<F::Output>
                where
                    F: FnManager<'a, M>,
                {
                    let inner = <[<$buddy1 Layout>]<[<$buddy2 Layout>]<B>> as BuddyLayout>::struct_ref::<F, M>(&space.0);
                    [<$name Alloc>](inner)
                }
            }
        }

        /// Combined Buddy branch
        #[perfect_derive::perfect_derive(PartialEq, Eq)]
        pub struct $name<B, M: ManagerBase>($buddy1<$buddy2<B, M>, M>);

        // Passthrough implementation, default derive macro can't derive this ...
        impl<B: Encode, M: ManagerSerialise> Encode for $name<B, M> {
            fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
                self.0.encode(encoder)
            }
        }

        // Passthrough implementation, default derive macro can't derive this ...
        impl<B: Decode<()>, M: ManagerDeserialise> Decode<()> for $name<B, M> {
            fn decode<D: Decoder<Context = ()>>(decoder: &mut D) -> Result<Self, DecodeError> {
                Ok(Self(Decode::decode(decoder)?))
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

            fn hash_state(&self) -> Hash
            where
                M: ManagerSerialise,
            {
                self.0.hash_state()
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
