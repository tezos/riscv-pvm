// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Key-component optimised for the AVL node representation.
//!
//! Specifically, this contains an optimised [`PartialOrd`] representation -
//! that compares by hashes as much as possible (in [`Prove`] and [`Verify`] modes).
//!
//! [`PartialOrd`]: std::cmp::PartialOrd

use std::cell::Cell;
use std::cmp::Ordering;
use std::convert::Infallible;
use std::ops::Deref;

use octez_riscv_data::codec::LeafCodec;
use octez_riscv_data::codec::LeafDecode;
use octez_riscv_data::codec::LeafEncode;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::FoldLeaf;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashFold;
use octez_riscv_data::hash::PartialHash;
use octez_riscv_data::hash::PartialHashFold;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::Partial;
use octez_riscv_data::merkle_proof::Suspended;
use octez_riscv_data::merkle_proof::SuspendedResult;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProofFold;
use octez_riscv_data::merkle_proof::proof_tree::MinimumPresence;
use octez_riscv_data::mode::Modal;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Provable;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use octez_riscv_data::mode::utils::Source;
use octez_riscv_data::mode::utils::not_found;
use perfect_derive::perfect_derive;

use crate::key::Key;

/// The AVL-node key representation of a [`Key`].
#[perfect_derive(Debug)]
pub struct NodeKey<M: Mode> {
    key: M::Select<KeyTemplate>,
}

impl<M: NodeKeyMode> NodeKey<M> {
    /// Create a new [`NodeKey`] component with the given initial key.
    pub fn new(key: Key) -> Self {
        M::new(key)
    }

    /// Compare `Self` against a plain [`Key`].
    ///
    /// *NB* we do not implement [`PartialOrd`] - as that forces us
    /// to return an `Option<Ordering>`, which would be semantically
    /// unclear as to what `None` means. Likewise, we cannot impl
    /// [`Ord`] as that is only available when comparing a type
    /// against itself.
    #[expect(
        clippy::should_implement_trait,
        reason = "`Ord`/`PartialOrd` are problematic here (see docstring)"
    )]
    pub fn cmp(&self, key: &Key) -> Ordering {
        M::cmp(self, key)
    }
}

impl<'normal> NodeKey<Prove<'normal>> {
    /// Was the value accessed (read or written) during proof generation?
    fn was_accessed(&self) -> bool {
        self.key.read.get()
    }

    /// Compare `Self` against a plain [`Key`] - without recording the access.
    ///
    /// Must only be used _during_ proof generation/folding - never
    /// during operations that must record accesses.
    ///
    /// See [`NodeKey::cmp`] for why this does not go through [`PartialOrd`].
    ///
    /// [`PartialOrd`]: std::cmp::PartialOrd
    pub fn unrecorded_cmp(&self, key: &Key) -> Ordering {
        self.key.inner.cmp(key)
    }
}

impl<M: NodeKeyMode> Clone for NodeKey<M> {
    fn clone(&self) -> Self {
        M::clone(self)
    }
}

impl<M: NodeKeyMode> PartialEq<Key> for NodeKey<M> {
    fn eq(&self, other: &Key) -> bool {
        M::eq(self, other)
    }
}

impl NodeKey<Normal> {
    /// Convert a NodeKey into proof mode.
    pub fn into_proof(self) -> NodeKey<Prove<'static>> {
        NodeKey {
            key: ProveImpl {
                inner: Source::Owned(Box::new(self.key)),
                read: Cell::new(false),
            },
        }
    }
}

impl AsRef<Key> for NodeKey<Normal> {
    fn as_ref(&self) -> &Key {
        &self.key
    }
}

impl<'normal> Provable<'normal> for NodeKey<Normal> {
    type Prover = NodeKey<Prove<'normal>>;

    fn start_proof(&'normal self) -> Self::Prover {
        NodeKey {
            key: ProveImpl {
                inner: Source::Borrowed(&self.key),
                read: Cell::new(false),
            },
        }
    }
}

impl<F: FoldLeaf> Foldable<F> for NodeKey<Normal>
where
    Key: LeafEncode<F::Codec>,
{
    fn fold(&self, builder: F) -> F::Folded {
        builder
            .fold_leaf(&self.key)
            .expect("Should be able to serialise key")
    }
}

impl<'normal, C: LeafCodec> Foldable<HashFold<C>> for NodeKey<Prove<'normal>>
where
    Key: LeafEncode<C>,
{
    fn fold(&self, builder: HashFold<C>) -> <HashFold<C> as Fold>::Folded {
        builder
            .fold_leaf(self.key.inner.deref())
            .expect("Should be able to hash NodeKey")
    }
}

impl<'normal, C: LeafCodec> Foldable<MerkleProofFold<C>> for NodeKey<Prove<'normal>>
where
    Key: LeafEncode<C>,
{
    fn fold(&self, builder: MerkleProofFold<C>) -> <MerkleProofFold<C> as Fold>::Folded {
        let data = <Key as LeafEncode<C>>::leaf_encode(self.key.inner.deref())
            .expect("Serialisation should not fail");

        // Determine whether the value has been read or written during proof generation. If so, we
        // must keep it in the Merkle tree (not blind it).
        let constraint = if self.was_accessed() {
            MinimumPresence::Present
        } else {
            MinimumPresence::MayOmit
        };

        builder.into_leaf(constraint, data)
    }
}

impl<C: LeafCodec> Foldable<PartialHashFold<C>> for NodeKey<Verify>
where
    Key: LeafEncode<C>,
{
    fn fold(&self, builder: PartialHashFold<C>) -> <PartialHashFold<C> as Fold>::Folded {
        let hash = match &self.key.inner {
            Partial::Absent => return builder.previous(),
            Partial::Blinded(hash) => *hash,
            Partial::Present(value) => {
                Hash::hash_leaf::<C, _>(value).expect("Hashing should not fail")
            }
        };

        PartialHash::Present(hash)
    }
}

impl<C: LeafCodec> FromProof<C> for NodeKey<Verify>
where
    Key: LeafDecode<C>,
{
    fn from_proof<Proof: Deserialiser<Codec = C>>(proof: Proof) -> SuspendedResult<Proof, Self> {
        let result = proof.into_leaf()?.map(|key| NodeKey {
            key: VerifyImpl { inner: key },
        });

        Ok(result)
    }
}

/// Modal template for the [`NodeKey`] component.
///
/// This type helps us pick the representation of [`NodeKey`]
/// for each mode by implementing [`Modal`].
struct KeyTemplate(Infallible);

impl Modal for KeyTemplate {
    type Normal = Key;

    type Prove<'normal> = ProveImpl<'normal>;

    type Verify = VerifyImpl;
}

/// Mode types that implement this trait support common operations on [`NodeKey`] components
///
/// The methods of the [`NodeKey`] type provide a more convenient interface to the functionality of
/// this trait.
pub trait NodeKeyMode: private::NodeKeyImpl {}

mod private {
    use std::cmp::Ordering;

    use octez_riscv_data::mode::Mode;

    use super::NodeKey;
    use crate::key::Key;

    /// Private trait to ensure callers go through the
    /// [`NodeKey`] methods, rather than the [`NodeKeyMode`]
    /// trait.
    ///
    /// [`NodeKeyMode`]: super::NodeKeyMode
    pub trait NodeKeyImpl: Mode {
        /// Create a new [`NodeKey`] from a [`Key`].
        fn new(key: Key) -> NodeKey<Self>;

        /// Implementation of [`PartialEq`] against a [`Key`].
        fn eq(this: &NodeKey<Self>, rhs: &Key) -> bool;

        /// Implementation of ordering against a [`Key`].
        ///
        /// See [`NodeKey::cmp`] for further details.
        fn cmp(this: &NodeKey<Self>, rhs: &Key) -> Ordering;

        /// Implementation of [`Clone`].
        ///
        /// This clones the entire component, not just the held [`Key`]. Consider
        /// this when cloning components in [`Prove`] mode.
        ///
        /// [`Prove`]: octez_riscv_data::mode::Prove
        fn clone(this: &NodeKey<Self>) -> NodeKey<Self>;
    }
}

impl NodeKeyMode for Normal {}

impl private::NodeKeyImpl for Normal {
    fn new(key: Key) -> NodeKey<Self> {
        NodeKey { key }
    }

    fn eq(this: &NodeKey<Self>, rhs: &Key) -> bool {
        this.key.eq(rhs)
    }

    fn cmp(this: &NodeKey<Self>, rhs: &Key) -> Ordering {
        this.key.cmp(rhs)
    }

    fn clone(this: &NodeKey<Self>) -> NodeKey<Self> {
        NodeKey {
            key: this.key.clone(),
        }
    }
}

#[derive(Clone, Debug)]
struct ProveImpl<'normal> {
    inner: Source<'normal, Key>,
    read: Cell<bool>,
}

impl<'normal> NodeKeyMode for Prove<'normal> {}

impl<'normal> private::NodeKeyImpl for Prove<'normal> {
    fn new(key: Key) -> NodeKey<Self> {
        let prove_impl = ProveImpl {
            inner: Source::owned(key),
            read: Cell::new(false),
        };

        NodeKey { key: prove_impl }
    }

    fn eq(this: &NodeKey<Self>, rhs: &Key) -> bool {
        this.key.read.set(true);
        this.key.inner.eq(rhs)
    }

    fn cmp(this: &NodeKey<Self>, rhs: &Key) -> Ordering {
        this.key.read.set(true);
        this.key.inner.cmp(rhs)
    }

    fn clone(this: &NodeKey<Self>) -> NodeKey<Self> {
        NodeKey {
            key: this.key.clone(),
        }
    }
}

#[derive(Clone, Debug)]
struct VerifyImpl {
    inner: Partial<Key>,
}

impl NodeKeyMode for Verify {}

impl private::NodeKeyImpl for Verify {
    fn new(key: Key) -> NodeKey<Self> {
        let verify_impl = VerifyImpl {
            inner: Partial::Present(key),
        };

        NodeKey { key: verify_impl }
    }

    fn eq(this: &NodeKey<Self>, rhs: &Key) -> bool {
        match &this.key.inner {
            Partial::Absent | Partial::Blinded(_) => {
                // SAFETY: `not_found` is safe to call because
                //         we're in `Verify` mode.
                unsafe { not_found() }
            }
            Partial::Present(key) => key.eq(rhs),
        }
    }

    fn cmp(this: &NodeKey<Self>, rhs: &Key) -> Ordering {
        match &this.key.inner {
            Partial::Absent | Partial::Blinded(_) => {
                // SAFETY: `not_found` is safe to call because
                //         we're in `Verify` mode.
                unsafe { not_found() }
            }
            Partial::Present(key) => key.cmp(rhs),
        }
    }

    fn clone(this: &NodeKey<Self>) -> NodeKey<Self> {
        NodeKey {
            key: this.key.clone(),
        }
    }
}

#[cfg(test)]
mod tests;
