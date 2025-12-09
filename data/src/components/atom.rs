// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! State component for a single value
//!
//! See [`Atom`] for more details.

use std::cell::Cell;
use std::convert::Infallible;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::DerefMut;

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use perfect_derive::perfect_derive;

use crate::clone::CloneState;
use crate::foldable::Foldable;
use crate::hash::Hash;
use crate::hash::HashFold;
use crate::hash::PartialHash;
use crate::hash::PartialHashFold;
use crate::merkle_proof::Deserialiser;
use crate::merkle_proof::FromProof;
use crate::merkle_proof::Partial;
use crate::merkle_proof::Suspended;
use crate::merkle_proof::SuspendedResult;
use crate::merkle_tree::MerkleTree;
use crate::merkle_tree::MerkleTreeFold;
use crate::mode::Modal;
use crate::mode::Mode;
use crate::mode::Normal;
use crate::mode::Prove;
use crate::mode::Verify;
use crate::mode::utils::Source;
use crate::mode::utils::not_found;
use crate::serialisation::serialise;

/// Single value state component
///
/// The [`Atom`] component holds a single value of type `T`.  
///
/// The held value is atomic in its presence. In [`Normal`] and [`Prove`] mode, the value is always
/// present. In [`Verify`] mode, the value is either fully present, or fully absent.
#[perfect_derive(Debug)]
#[repr(transparent)]
pub struct Atom<T: 'static, M: Mode + ?Sized> {
    atom: M::Select<AtomTemplate<T>>,
}

impl<T: 'static, M: AtomMode> Atom<T, M> {
    /// Create a new [`Atom`] component with the given initial value.
    pub fn new(value: T) -> Self {
        M::new(value)
    }

    /// Reads the current state value.
    #[inline]
    pub fn read(&self) -> T
    where
        T: Copy,
    {
        *M::deref(self)
    }

    /// Update the state value.
    #[inline]
    pub fn write(&mut self, value: T) {
        M::write(self, value)
    }
}

impl<T: 'static> Atom<T, Normal> {
    /// Offset that when applied to a pointer to an [`Atom<T, Normal>`], that yields a pointer to `T`
    pub const FIELD_OFFSET: usize = {
        // This expression will fail to type check when the internal representation of
        // `Atom` changes. E.g. when it is no longer `T`. This is important to ensure the offset
        // computed below is sound. We're effectively bringing the type assumption closer to where
        // the assumption is used.
        let _sanity = |atom: Atom<T, Normal>| -> T { atom.atom };

        std::mem::offset_of!(Self, atom)
    };

    /// Construct an [`Atom`] in [`Prove`] mode.
    ///
    /// The initial proof value is equal to that of the given [`Atom`] in [`Normal`] mode.
    pub fn start_proof(&self) -> Atom<T, Prove<'_>> {
        Atom {
            atom: ProveImpl {
                previous: Source::Borrowed(&self.atom),
                current: None,
                read: Cell::new(false),
            },
        }
    }
}

impl<T: 'static> Atom<T, Verify> {
    /// Construct an [`Atom`] in [`Verify`] mode that represents an absent value.
    pub fn absent() -> Self {
        Atom {
            atom: Partial::Absent,
        }
    }
}

impl<A, B, M, N> PartialEq<Atom<B, N>> for Atom<A, M>
where
    A: PartialEq<B> + 'static,
    B: 'static,
    M: AtomMode,
    N: AtomMode,
{
    fn eq(&self, other: &Atom<B, N>) -> bool {
        A::eq(self, other)
    }
}

impl<A: Eq + 'static, M: AtomMode> Eq for Atom<A, M> {}

impl<T: 'static, M: AtomMode> Deref for Atom<T, M> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        M::deref(self)
    }
}

impl<T: Clone + 'static, M: AtomMode> DerefMut for Atom<T, M> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        M::deref_mut(self)
    }
}

impl<T: Default + 'static, M: AtomMode> Default for Atom<T, M> {
    fn default() -> Self {
        M::new(T::default())
    }
}

impl<T: Encode + 'static> Foldable<HashFold> for Atom<T, Normal> {
    fn fold(&self, _builder: HashFold) -> Hash {
        Hash::blake3_hash(&self.atom).expect("Hashing should not fail")
    }
}

impl<'normal, T: Encode + 'static> Foldable<HashFold> for Atom<T, Prove<'normal>> {
    fn fold(&self, _builder: HashFold) -> Hash {
        let value = self
            .atom
            .current
            .as_deref()
            .unwrap_or_else(|| &self.atom.previous);
        Hash::blake3_hash(value).expect("Hashing should not fail")
    }
}

impl<T: Encode + 'static> Foldable<MerkleTreeFold> for Atom<T, Prove<'_>> {
    fn fold(&self, _builder: MerkleTreeFold) -> MerkleTree {
        let data = serialise(self.atom.previous.deref()).expect("Serialisation should not fail");

        // Determine whether the value has been read or written during proof generation. If so, we
        // must mark it as not blinded in the Merkle tree.
        let access = self.atom.read.get() || self.atom.current.is_some();

        MerkleTree::make_merkle_leaf(data, access)
    }
}

impl<T: Encode + 'static> Foldable<PartialHashFold<'_>> for Atom<T, Verify> {
    fn fold(&self, builder: PartialHashFold) -> PartialHash {
        let hash = match &self.atom {
            Partial::Absent => return builder.previous(),
            Partial::Blinded(hash) => *hash,
            Partial::Present(value) => Hash::blake3_hash(value).expect("Hashing should not fail"),
        };
        PartialHash::Present(hash)
    }
}

impl<T: Decode<()> + 'static> FromProof for Atom<T, Verify> {
    fn from_proof<Proof: Deserialiser>(proof: Proof) -> SuspendedResult<Proof, Self> {
        let result = proof.into_leaf()?.map(|value| Atom { atom: value });
        Ok(result)
    }
}

impl<T: Encode + 'static, M: EncodeAtomMode> Encode for Atom<T, M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        M::encode(self, encoder)
    }
}

impl<C, T: Decode<C> + 'static, M: AtomMode> Decode<C> for Atom<T, M> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Atom::new(T::decode(decoder)?))
    }
}

impl<T: Clone + 'static, M: CloneAtomMode> CloneState for Atom<T, M> {
    fn clone_state(&self) -> Self {
        M::clone(self)
    }
}

impl<T: Clone + 'static, M: CloneAtomMode> Clone for Atom<T, M> {
    fn clone(&self) -> Self {
        M::clone(self)
    }
}

/// Representation of the [`Atom`] component in [`Prove`] mode
#[perfect_derive(Clone, Debug)]
struct ProveImpl<'normal, T> {
    /// Previous value held by the component
    ///
    /// This is relevant because the proof generation mode will result in a Merkle proof that
    /// contains the initial values. Hence, we need to keep track of the previous value.
    previous: Source<'normal, T>,

    /// Current value held by the component
    ///
    /// This is important because in proof generation mode, the value may be updated. Subsequent
    /// reads should reflect the updated value.
    current: Option<Box<T>>,

    /// Whether the value has been read during proof generation
    ///
    /// This helps us decide whether the value should be included in the Merkle proof or not.
    read: Cell<bool>,
}

/// Modal template for the [`Atom`] component
///
/// This type helps us pick the representation of [`Atom`] for each mode by implementing [`Modal`].
struct AtomTemplate<T: ?Sized>(PhantomData<T>, Infallible);

impl<T: 'static> Modal for AtomTemplate<T> {
    type Normal = T;

    type Prove<'normal> = ProveImpl<'normal, T>;

    type Verify = Partial<T>;
}

/// Mode types that implement this trait support common operations on [`Atom`] components
///
/// The methods of the [`Atom`] type provide a more convenient interface to the functionality of
/// this trait.
pub trait AtomMode: Mode {
    /// See [`Atom::new`].
    fn new<T: 'static>(value: T) -> Atom<T, Self>;

    /// Obtain an immutable reference to the current value held.
    fn deref<T: 'static>(this: &Atom<T, Self>) -> &T;

    /// Obtain a mutable reference to the current value held.
    fn deref_mut<T: Clone + 'static>(this: &mut Atom<T, Self>) -> &mut T;

    /// See [`Atom::write`].
    fn write<T: 'static>(this: &mut Atom<T, Self>, value: T);
}

impl AtomMode for Normal {
    fn new<T: 'static>(value: T) -> Atom<T, Self> {
        Atom { atom: value }
    }

    #[inline]
    fn deref<T: 'static>(this: &Atom<T, Self>) -> &T {
        &this.atom
    }

    #[inline]
    fn deref_mut<T: 'static>(this: &mut Atom<T, Self>) -> &mut T {
        &mut this.atom
    }

    #[inline]
    fn write<T: 'static>(this: &mut Atom<T, Self>, value: T) {
        this.atom = value;
    }
}

impl<'normal> AtomMode for Prove<'normal> {
    fn new<T: 'static>(value: T) -> Atom<T, Self> {
        Atom {
            atom: ProveImpl {
                previous: Source::from(value),
                current: None,
                read: Cell::new(false),
            },
        }
    }

    fn deref<T: 'static>(this: &Atom<T, Self>) -> &T {
        // When proving, we need to ensure the value is marked as read such that it will be
        // included in the proof. This ensures the verifier, which will also call `deref`, can see
        // the value.
        this.atom.read.set(true);

        this.atom
            .current
            .as_deref()
            .unwrap_or_else(|| &this.atom.previous)
    }

    fn deref_mut<T: Clone + 'static>(this: &mut Atom<T, Self>) -> &mut T {
        // When proving, we need to ensure the value is marked as read such that it will be
        // included in the proof. This ensures the verifier, which will also call `deref`, can see
        // the value.
        this.atom.read.set(true);

        this.atom
            .current
            .get_or_insert_with(|| Box::new(this.atom.previous.deref().clone()))
    }

    fn write<T: 'static>(this: &mut Atom<T, Self>, value: T) {
        this.atom.current = Some(Box::new(value));
    }
}

impl AtomMode for Verify {
    fn new<T: 'static>(value: T) -> Atom<T, Self> {
        Atom {
            atom: Partial::Present(value),
        }
    }

    fn deref<T: 'static>(this: &Atom<T, Self>) -> &T {
        match &this.atom {
            Partial::Absent | Partial::Blinded(_) => {
                // SAFETY: `not_found` is safe to call because we're in `Verify` mode
                unsafe { not_found() }
            }
            Partial::Present(value) => value,
        }
    }

    fn deref_mut<T: Clone + 'static>(this: &mut Atom<T, Self>) -> &mut T {
        match &mut this.atom {
            Partial::Absent | Partial::Blinded(_) => {
                // SAFETY: `not_found` is safe to call because we're in `Verify` mode
                unsafe { not_found() }
            }
            Partial::Present(value) => value,
        }
    }

    fn write<T: 'static>(this: &mut Atom<T, Self>, value: T) {
        this.atom = Partial::Present(value);
    }
}

/// Mode types that implement this trait support encoding of [`Atom`] components
pub trait EncodeAtomMode: Mode {
    /// Encode the current value held by the [`Atom`] component. In other words, this encodes `T`.
    ///
    /// In [`Prove`] mode, encoding does not count as a read and will not ensure the value is
    /// present in the resulting proof.
    fn encode<T: Encode + 'static, E: Encoder>(
        this: &Atom<T, Self>,
        encoder: &mut E,
    ) -> Result<(), EncodeError>;
}

impl EncodeAtomMode for Normal {
    fn encode<T: Encode + 'static, E: Encoder>(
        this: &Atom<T, Self>,
        encoder: &mut E,
    ) -> Result<(), EncodeError> {
        this.atom.encode(encoder)
    }
}

impl EncodeAtomMode for Prove<'_> {
    fn encode<T: Encode + 'static, E: Encoder>(
        this: &Atom<T, Self>,
        encoder: &mut E,
    ) -> Result<(), EncodeError> {
        this.atom
            .current
            .as_deref()
            .unwrap_or_else(|| &this.atom.previous)
            .encode(encoder)
    }
}

/// Mode types that implement this trait support cloning of [`Atom`] components
///
/// This trait is deliberately separate from [`AtomMode`]. You likely don't want this trait to
/// appear in trait bounds that are relevant to core PVM operations.
///
/// Cloning an [`Atom`] component may have different semantics depending on the mode. For example,
/// in [`Prove`] mode, cloning the component will clone the entire internal state, not just the
/// current value. That includes information relevant to the Merkle tree proof. Hence, it is
/// different from using [`Atom::read`] + [`Atom::new`].
pub trait CloneAtomMode: AtomMode {
    /// Clones the given [`Atom`] component.
    ///
    /// This clones the entire component, not just the internal value. Consider this when cloning
    /// components in [`Prove`] mode.
    fn clone<T: Clone + 'static>(this: &Atom<T, Self>) -> Atom<T, Self>;
}

impl CloneAtomMode for Normal {
    fn clone<T: Clone + 'static>(this: &Atom<T, Self>) -> Atom<T, Self> {
        Atom {
            atom: this.atom.clone(),
        }
    }
}

impl<'normal> CloneAtomMode for Prove<'normal> {
    fn clone<T: Clone + 'static>(this: &Atom<T, Self>) -> Atom<T, Self> {
        Atom {
            atom: this.atom.clone(),
        }
    }
}

impl CloneAtomMode for Verify {
    fn clone<T: Clone + 'static>(this: &Atom<T, Self>) -> Atom<T, Self> {
        Atom {
            atom: this.atom.clone(),
        }
    }
}
