// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Key-component optimised for the AVL node representation.
//!
//! Specifically, this contains an optimised [`PartialOrd`] representation -
//! that compares by hashes as much as possible (in [`Prove`] and [`Verify`] modes).
//!
//! A key folds as a node of its length and a tree of [`PAGE_SIZE`]-byte pages,
//! rather than as one leaf holding the whole key. That shape lets a comparison
//! put only what it needs into the proof:
//!
//! - An equality check is settled against the hash of the key as a whole, which
//!   lets the key be blinded outright.
//! - Two different keys are ordered within the first page on which they differ:
//!   the pages before it are equal, which the verifier re-checks against their
//!   hashes, and the pages after it never enter into the comparison. Only that
//!   one page has to be fully included in the proof, so an ordering costs a
//!   less than including the full key in the worst case.
//! - Keys where one is a prefix of the other may agree on every page they share
//!   and are additionally ordered by their lengths.
//!
//! [`PartialOrd`]: std::cmp::PartialOrd

use std::cell::Cell;
use std::cmp::Ordering;
use std::convert::Infallible;
use std::ops::Deref;
use std::ops::Range;

use bincode::BorrowDecode;
use bincode::Decode;
use bincode::Encode;
use bincode::de::BorrowDecoder;
use bincode::de::read::Reader;
use bincode::enc::Encoder;
use bincode::enc::write::Writer;
use octez_riscv_data::codec::LeafCodec;
use octez_riscv_data::codec::LeafDecode;
use octez_riscv_data::codec::LeafEncode;
use octez_riscv_data::foldable::EncodeLeaf;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::FoldLeaf;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::NodeFold;
use octez_riscv_data::foldable::seq_tree::IndexableSeqAsTree;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashFold;
use octez_riscv_data::hash::PartialHash;
use octez_riscv_data::hash::PartialHashFold;
use octez_riscv_data::merkle_proof::Deserialiser;
use octez_riscv_data::merkle_proof::DeserialiserError;
use octez_riscv_data::merkle_proof::DeserialiserNode;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::Partial;
use octez_riscv_data::merkle_proof::Suspended;
use octez_riscv_data::merkle_proof::SuspendedResult;
use octez_riscv_data::merkle_proof::descend_tree;
use octez_riscv_data::merkle_proof::proof_tree::ForceMinimumPresence;
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

use crate::key::KEY_MAX_SIZE;
use crate::key::Key;

/// Bytes of a key held in one page of its merkle tree.
///
/// This is the page size that minimises a key's contribution to a proof. A
/// comparison reveals one page plus a hash per layer of the page tree, so
/// halving the page size trades 32 bytes of page content for a 33-byte layer.
pub const PAGE_SIZE: usize = 64;

/// Children per node of a key's page tree.
pub const NODE_ARITY: usize = 2;

/// The most pages a key can occupy.
const MAX_PAGES: usize = KEY_MAX_SIZE.div_ceil(PAGE_SIZE);

/// The number of pages a key of `len` bytes occupies.
const fn page_count(len: usize) -> usize {
    len.div_ceil(PAGE_SIZE)
}

/// The byte range of page `index` within a key of `len` bytes.
///
/// Every page but the last is full, so only the last one is shorter than
/// [`PAGE_SIZE`].
fn page_range(len: usize, index: usize) -> Range<usize> {
    let start = index * PAGE_SIZE;

    start..(start + PAGE_SIZE).min(len)
}

/// The length of a key, as it is written to the key's length leaf.
fn key_length(key: &Key) -> u8 {
    u8::try_from(key.as_ref().len()).expect("KEY_MAX_SIZE is precisely u8::MAX")
}

/// The first page on which two keys differ, if they share such a page.
///
/// Pages before it are equal, and a verifier can check that against their
/// hashes. The ordering of two keys that differ on a shared page is decided
/// within that page; keys that agree on every page they share are ordered by
/// their lengths.
fn first_differing_page(lhs: &[u8], rhs: &[u8]) -> Option<usize> {
    (0..page_count(lhs.len()).min(page_count(rhs.len())))
        .find(|&page| lhs[page_range(lhs.len(), page)] != rhs[page_range(rhs.len(), page)])
}

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
    ///
    /// [`PartialOrd`]: std::cmp::PartialOrd
    #[expect(
        clippy::should_implement_trait,
        reason = "`Ord`/`PartialOrd` are problematic here (see docstring)"
    )]
    pub fn cmp(&self, key: &Key) -> Ordering {
        M::cmp(self, key)
    }
}

impl<'normal> NodeKey<Prove<'normal>> {
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
            key: ProveImpl::new(Source::Owned(Box::new(self.key))),
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
            key: ProveImpl::new(Source::Borrowed(&self.key)),
        }
    }
}

/// Fold a key as a node of its length and its pages.
fn fold_key<F: FoldLeaf>(builder: F, key: &Key) -> F::Folded
where
    u8: LeafEncode<F::Codec>,
    for<'page> PageRef<'page>: LeafEncode<F::Codec>,
{
    let bytes = key.as_ref();
    let mut builder = builder.into_node_fold();

    builder.add(&EncodeLeaf::new(
        key_length(key),
        "Should be able to serialise a key's length",
    ));

    let page = |index: usize| {
        EncodeLeaf::new(
            PageRef(&bytes[page_range(bytes.len(), index)]),
            "Should be able to serialise a key's page",
        )
    };
    builder.add(&IndexableSeqAsTree::new(
        page_count(bytes.len()),
        NODE_ARITY,
        &page,
    ));

    builder.done()
}

/// The hash a key folds to.
fn key_hash<C: LeafCodec>(key: &Key) -> Hash
where
    u8: LeafEncode<C>,
    for<'page> PageRef<'page>: LeafEncode<C>,
{
    fold_key(HashFold::<C>::default(), key)
}

/// The hash one page of a key folds to.
fn page_hash<C: LeafCodec>(page: &[u8]) -> Hash
where
    for<'page> PageRef<'page>: LeafEncode<C>,
{
    Hash::hash_leaf::<C, _>(&PageRef(page)).expect("Hashing a key's page should not fail")
}

impl<F: FoldLeaf> Foldable<F> for NodeKey<Normal>
where
    u8: LeafEncode<F::Codec>,
    for<'page> PageRef<'page>: LeafEncode<F::Codec>,
{
    fn fold(&self, builder: F) -> F::Folded {
        fold_key(builder, &self.key)
    }
}

// NB this is the concrete [`HashFold`] rather than any [`FoldLeaf`]: coherence in this
// crate cannot rule out [`MerkleProofFold`] gaining a [`FoldLeaf`] impl upstream, which
// would collide with the proof fold below.
impl<'normal, C: LeafCodec> Foldable<HashFold<C>> for NodeKey<Prove<'normal>>
where
    u8: LeafEncode<C>,
    for<'page> PageRef<'page>: LeafEncode<C>,
{
    fn fold(&self, builder: HashFold<C>) -> <HashFold<C> as Fold>::Folded {
        fold_key(builder, self.key.inner.deref())
    }
}

impl<'normal, C: LeafCodec> Foldable<MerkleProofFold<C>> for NodeKey<Prove<'normal>>
where
    u8: LeafEncode<C>,
    for<'page> PageRef<'page>: LeafEncode<C>,
{
    fn fold(&self, builder: MerkleProofFold<C>) -> <MerkleProofFold<C> as Fold>::Folded {
        // A key compared only by its hash is blinded whole, which leaves the tree
        // below it out of the proof. The constraint has to be raised here rather
        // than by the leaves, which are all omittable in that case.
        ForceMinimumPresence {
            min_constraint: self.key.whole.get().presence(),
            inner: KeyTree(&self.key),
        }
        .fold(builder)
    }
}

/// A key's merkle tree, folded from the accesses proof generation recorded.
///
/// Kept apart from [`NodeKey`] so that [`ForceMinimumPresence`] can raise the
/// presence of the tree as a whole.
struct KeyTree<'prove, 'normal>(&'prove ProveImpl<'normal>);

impl<C: LeafCodec> Foldable<MerkleProofFold<C>> for KeyTree<'_, '_>
where
    u8: LeafEncode<C>,
    for<'page> PageRef<'page>: LeafEncode<C>,
{
    fn fold(&self, builder: MerkleProofFold<C>) -> <MerkleProofFold<C> as Fold>::Folded {
        let key = self.0.inner.deref();
        let bytes = key.as_ref();

        let mut builder = builder.into_node_fold();

        builder.add(&ProofLeaf {
            value: key_length(key),
            access: self.0.length.get(),
        });

        let page = |index: usize| ProofLeaf {
            value: PageRef(&bytes[page_range(bytes.len(), index)]),
            access: self.0.page_access(index),
        };
        builder.add(&IndexableSeqAsTree::new(
            page_count(bytes.len()),
            NODE_ARITY,
            &page,
        ));

        builder.done()
    }
}

/// A leaf of a key's merkle tree, carried by the proof to the extent the access
/// that generated it requires.
struct ProofLeaf<T> {
    /// The value the leaf holds.
    value: T,

    /// How the value was accessed during proof generation.
    access: KeyAccess,
}

impl<C: LeafCodec, T: LeafEncode<C>> Foldable<MerkleProofFold<C>> for ProofLeaf<T> {
    fn fold(&self, builder: MerkleProofFold<C>) -> <MerkleProofFold<C> as Fold>::Folded {
        let data =
            <T as LeafEncode<C>>::leaf_encode(&self.value).expect("Serialisation should not fail");

        builder.into_leaf(self.access.presence(), data)
    }
}

impl<C: LeafCodec> Foldable<PartialHashFold<C>> for NodeKey<Verify>
where
    u8: LeafEncode<C>,
    for<'page> PageRef<'page>: LeafEncode<C>,
{
    fn fold(&self, builder: PartialHashFold<C>) -> <PartialHashFold<C> as Fold>::Folded {
        let (length, pages) = match &self.key {
            VerifyImpl::Absent => return builder.previous(),
            VerifyImpl::Blinded { hash, .. } => return PartialHash::Present(*hash),
            VerifyImpl::Tree { length, pages } => (length, pages),
        };

        let mut builder = builder.into_node_fold();

        builder.add(&PartialLeaf(length.clone().map_present(|length| {
            Hash::hash_leaf::<C, _>(&length).expect("Hashing a key's length should not fail")
        })));

        let page = |index: usize| PartialLeaf(pages[index].hash::<C>());
        builder.add(&IndexableSeqAsTree::new(pages.len(), NODE_ARITY, &page));

        builder.done()
    }
}

/// A leaf of a key's merkle tree in [`Verify`] mode, folded to whichever of its
/// hash the proof carries.
struct PartialLeaf(Partial<Hash>);

impl<C: LeafCodec> Foldable<PartialHashFold<C>> for PartialLeaf {
    fn fold(&self, builder: PartialHashFold<C>) -> <PartialHashFold<C> as Fold>::Folded {
        match self.0 {
            Partial::Absent => builder.previous(),
            Partial::Blinded(hash) | Partial::Present(hash) => PartialHash::Present(hash),
        }
    }
}

impl<C: LeafCodec> FromProof<C> for NodeKey<Verify>
where
    u8: LeafDecode<C> + LeafEncode<C>,
    Page: LeafDecode<C>,
    for<'page> PageRef<'page>: LeafEncode<C>,
{
    fn from_proof<Proof: Deserialiser<Codec = C>>(proof: Proof) -> SuspendedResult<Proof, Self> {
        let node = proof.into_node()?;

        // A key compared only by its hash is blinded whole, and one that was never
        // compared at all is omitted. Either way the branches below are absent, but
        // they still have to be walked for the node to be done.
        let presence = node.presence();

        let (node, length) = node.next_branch_with(|proof| proof.into_leaf::<u8>())?;

        let (node, pages) = node.next_branch_with(|proof| {
            let Partial::Present(length) = &length else {
                // Without the length there is no telling how many pages to expect,
                // so the page tree must not be present either.
                let proof = proof.into_node()?;

                if let Partial::Present(()) = proof.presence() {
                    return Err(DeserialiserError::custom(
                        KeyProofError::LengthAbsentButPagesPresent,
                    ));
                }

                return proof.done(Vec::new());
            };

            let count = page_count(*length as usize);
            let mut pages = vec![VerifyPage::Absent; count];

            let mut for_leaf = |index: usize, proof: Proof| {
                Ok(proof.into_leaf::<Page>()?.map(|page| {
                    pages[index] = VerifyPage::new::<C>(page);
                }))
            };

            let result = descend_tree(proof, NODE_ARITY, count, &mut for_leaf)?;

            Ok(result.map(|()| pages))
        })?;

        let key = match presence {
            Partial::Absent => VerifyImpl::Absent,
            Partial::Blinded(hash) => VerifyImpl::Blinded {
                hash,
                hash_key: key_hash::<C>,
            },
            Partial::Present(()) => VerifyImpl::Tree { length, pages },
        };

        node.done(NodeKey { key })
    }
}

/// Errors indicating a bad proof for a [`NodeKey`].
#[derive(Debug, thiserror::Error)]
enum KeyProofError {
    #[error("Key length is absent but some of its pages are present")]
    LengthAbsentButPagesPresent,
}

/// One page of a key's bytes, as it is written to a proof leaf.
///
/// A page holds at most [`PAGE_SIZE`] bytes, so - as for [`Key`] itself - the
/// length prefix is a single byte rather than the `u64` bincode writes for a
/// byte sequence by default.
pub struct PageRef<'page>(&'page [u8]);

impl Encode for PageRef<'_> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), bincode::error::EncodeError> {
        let len = u8::try_from(self.0.len()).expect("A page holds at most PAGE_SIZE bytes");

        Encode::encode(&len, encoder)?;
        encoder.writer().write(self.0)?;

        Ok(())
    }
}

/// One page of a key's bytes, decoded from a proof leaf.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Page(Vec<u8>);

impl Encode for Page {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), bincode::error::EncodeError> {
        PageRef(&self.0).encode(encoder)
    }
}

impl<Context> Decode<Context> for Page {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let len = u8::decode(decoder)? as usize;

        if len > PAGE_SIZE {
            return Err(bincode::error::DecodeError::Other(
                "A key page cannot be longer than PAGE_SIZE",
            ));
        }

        let mut page = vec![0; len];
        decoder.reader().read(page.as_mut_slice())?;

        Ok(Page(page))
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for Page {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Decode::decode(decoder)
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

/// How much of a leaf of a key's merkle tree a proof must carry, given the way
/// the key was compared during proof generation.
///
/// The variants are ordered by how much they reveal, so that repeated
/// comparisons against the same key accumulate with [`Ord::max`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum KeyAccess {
    /// Never looked at, and so omittable from the proof entirely.
    None,

    /// Compared by hash, so the proof need only carry that hash.
    Hashed,

    /// Its bytes decided the comparison, so the proof must carry them.
    Read,
}

impl KeyAccess {
    /// The presence the proof must give a leaf accessed this way.
    fn presence(self) -> MinimumPresence {
        match self {
            KeyAccess::None => MinimumPresence::MayOmit,
            KeyAccess::Hashed => MinimumPresence::MayBlind,
            KeyAccess::Read => MinimumPresence::Present,
        }
    }
}

#[derive(Clone, Debug)]
struct ProveImpl<'normal> {
    inner: Source<'normal, Key>,

    /// How the key as a whole was compared.
    whole: Cell<KeyAccess>,

    /// How the key's length was read.
    length: Cell<KeyAccess>,

    /// How each of the key's pages was compared.
    pages: [Cell<KeyAccess>; MAX_PAGES],
}

impl<'normal> ProveImpl<'normal> {
    /// A key that has not been compared yet.
    fn new(inner: Source<'normal, Key>) -> Self {
        ProveImpl {
            inner,
            whole: Cell::new(KeyAccess::None),
            length: Cell::new(KeyAccess::None),
            pages: [const { Cell::new(KeyAccess::None) }; MAX_PAGES],
        }
    }

    /// Record a comparison that the verifier settles against the hash of the key
    /// as a whole.
    fn record_whole(&self) {
        self.whole.set(self.whole.get().max(KeyAccess::Hashed));
    }

    /// Record the page-by-page comparison of this key against `rhs`, which must
    /// not be equal to it.
    fn record_pages(&self, rhs: &Key) {
        // Every page comparison starts from the key's length, which fixes where
        // its pages end.
        self.length.set(KeyAccess::Read);

        let bytes = self.inner.as_ref();
        let shared = page_count(bytes.len()).min(page_count(rhs.as_ref().len()));

        // The pages up to the one that decides the ordering are equal, which the
        // verifier re-checks against their hashes. Where the keys agree on every
        // page they share, their lengths decide the ordering and every shared
        // page has to be checked.
        let divergence = first_differing_page(bytes, rhs.as_ref());

        for page in 0..divergence.unwrap_or(shared) {
            self.record_page(page, KeyAccess::Hashed);
        }

        if let Some(page) = divergence {
            self.record_page(page, KeyAccess::Read);
        }
    }

    /// Record an access to one page, keeping the most revealing one so far.
    fn record_page(&self, index: usize, access: KeyAccess) {
        let page = &self.pages[index];
        page.set(page.get().max(access));
    }

    /// How the proof must carry page `index`.
    ///
    /// A comparison settled against the hash of the key as a whole needs nothing
    /// of the pages - but only while the key is blinded whole. Once a page
    /// comparison has put the key's tree into the proof, that hash is no longer
    /// there to check against, so every page has to be checkable instead.
    fn page_access(&self, index: usize) -> KeyAccess {
        let access = self.pages[index].get();

        if self.whole.get() > KeyAccess::None && self.compared_by_page() {
            access.max(KeyAccess::Hashed)
        } else {
            access
        }
    }

    /// Was this key compared page by page at any point?
    fn compared_by_page(&self) -> bool {
        self.pages.iter().any(|page| page.get() > KeyAccess::None)
    }
}

impl<'normal> NodeKeyMode for Prove<'normal> {}

impl<'normal> private::NodeKeyImpl for Prove<'normal> {
    fn new(key: Key) -> NodeKey<Self> {
        NodeKey {
            key: ProveImpl::new(Source::owned(key)),
        }
    }

    fn eq(this: &NodeKey<Self>, rhs: &Key) -> bool {
        // The verifier settles equality by hashing `rhs` itself, so the key can
        // stay blinded whichever way the comparison goes.
        this.key.record_whole();
        this.key.inner.eq(rhs)
    }

    fn cmp(this: &NodeKey<Self>, rhs: &Key) -> Ordering {
        let ordering = this.key.inner.cmp(rhs);

        if ordering.is_eq() {
            // An exact match is an equality check, which the verifier can redo
            // against the hash of the blinded key.
            this.key.record_whole();
        } else {
            this.key.record_pages(rhs);
        }

        ordering
    }

    fn clone(this: &NodeKey<Self>) -> NodeKey<Self> {
        NodeKey {
            key: this.key.clone(),
        }
    }
}

/// The verify-mode representation of a [`NodeKey`].
#[derive(Clone, Debug)]
enum VerifyImpl {
    /// The key is missing from the proof entirely.
    Absent,

    /// The key was compared by its hash alone, and that hash is all the proof
    /// carries of it.
    Blinded {
        /// Hash the key folds to.
        hash: Hash,

        /// Hashes a [`Key`] as the proof's keys were hashed.
        ///
        /// Captured while deserialising, where the [`LeafCodec`] is known.
        hash_key: fn(&Key) -> Hash,
    },

    /// The key's merkle tree is in the proof, carrying whichever of its pages
    /// the comparison that generated the proof needed.
    Tree {
        /// The key's length in bytes.
        length: Partial<u8>,

        /// The key's pages, in order.
        pages: Vec<VerifyPage>,
    },
}

/// One page of a key in [`Verify`] mode.
#[derive(Clone, Debug)]
enum VerifyPage {
    /// The page is missing from the proof.
    Absent,

    /// The page was compared by its hash alone.
    Blinded {
        /// Hash the page folds to.
        hash: Hash,

        /// Hashes a page as the proof's pages were hashed.
        ///
        /// Captured while deserialising, where the [`LeafCodec`] is known.
        hash_page: fn(&[u8]) -> Hash,
    },

    /// The page's bytes are in the proof.
    Present(Vec<u8>),
}

impl VerifyPage {
    /// Build the verify-mode representation of a page, remembering how to hash a
    /// page under the proof's codec.
    fn new<C: LeafCodec>(page: Partial<Page>) -> Self
    where
        for<'page> PageRef<'page>: LeafEncode<C>,
    {
        match page {
            Partial::Absent => VerifyPage::Absent,
            Partial::Blinded(hash) => VerifyPage::Blinded {
                hash,
                hash_page: page_hash::<C>,
            },
            Partial::Present(page) => VerifyPage::Present(page.0),
        }
    }

    /// The hash this page folds to, as far as the proof determines it.
    fn hash<C: LeafCodec>(&self) -> Partial<Hash>
    where
        for<'page> PageRef<'page>: LeafEncode<C>,
    {
        match self {
            VerifyPage::Absent => Partial::Absent,
            VerifyPage::Blinded { hash, .. } => Partial::Blinded(*hash),
            VerifyPage::Present(page) => Partial::Present(page_hash::<C>(page)),
        }
    }

    /// Is this page equal to `rhs`?
    ///
    /// Returns `None` when the proof does not carry the page at all.
    fn eq_page(&self, rhs: &[u8]) -> Option<bool> {
        match self {
            VerifyPage::Absent => None,
            // A page's encoding is length-prefixed, so equal hashes mean equal bytes.
            VerifyPage::Blinded { hash, hash_page } => Some(*hash == hash_page(rhs)),
            VerifyPage::Present(page) => Some(page == rhs),
        }
    }
}

/// How far a proof settles the comparison of a [`NodeKey`] against a [`Key`].
enum Comparison {
    /// The comparison is settled.
    Decided(Ordering),

    /// The keys differ, but the proof does not carry the page that would order
    /// them.
    Unequal,

    /// The proof does not carry enough to compare the keys at all.
    Unknown,
}

impl VerifyImpl {
    /// Compare this key against `rhs`, as far as the proof allows.
    fn compare(&self, rhs: &Key) -> Comparison {
        let (length, pages) = match self {
            VerifyImpl::Absent => return Comparison::Unknown,
            // The key folds injectively, so equal hashes mean equal keys.
            VerifyImpl::Blinded { hash, hash_key } if *hash == hash_key(rhs) => {
                return Comparison::Decided(Ordering::Equal);
            }
            VerifyImpl::Blinded { .. } => return Comparison::Unequal,
            VerifyImpl::Tree { length, pages } => (length, pages),
        };

        let Partial::Present(length) = length else {
            return Comparison::Unknown;
        };

        let length = *length as usize;
        let rhs = rhs.as_ref();

        for index in 0..page_count(length).min(page_count(rhs.len())) {
            let rhs_page = &rhs[page_range(rhs.len(), index)];

            match &pages[index] {
                VerifyPage::Present(page) => match page.as_slice().cmp(rhs_page) {
                    // The keys are ordered within the first page on which they
                    // differ, and agree on every page before it.
                    Ordering::Equal => continue,
                    ordering => return Comparison::Decided(ordering),
                },
                page => match page.eq_page(rhs_page) {
                    Some(true) => continue,
                    // A blinded page tells the verifier that the keys differ, but
                    // not which way round they go.
                    Some(false) => return Comparison::Unequal,
                    None => return Comparison::Unknown,
                },
            }
        }

        // Keys that agree on every page they share are ordered by their lengths.
        Comparison::Decided(length.cmp(&rhs.len()))
    }
}

impl NodeKeyMode for Verify {}

impl private::NodeKeyImpl for Verify {
    fn new(key: Key) -> NodeKey<Self> {
        let bytes = key.as_ref();
        let pages = (0..page_count(bytes.len()))
            .map(|index| VerifyPage::Present(bytes[page_range(bytes.len(), index)].to_vec()))
            .collect();

        NodeKey {
            key: VerifyImpl::Tree {
                length: Partial::Present(key_length(&key)),
                pages,
            },
        }
    }

    fn eq(this: &NodeKey<Self>, rhs: &Key) -> bool {
        match this.key.compare(rhs) {
            Comparison::Decided(ordering) => ordering.is_eq(),
            // Knowing the keys differ is all equality needs.
            Comparison::Unequal => false,
            // SAFETY: `not_found` is safe to call because
            //         we're in `Verify` mode.
            Comparison::Unknown => unsafe { not_found() },
        }
    }

    fn cmp(this: &NodeKey<Self>, rhs: &Key) -> Ordering {
        match this.key.compare(rhs) {
            Comparison::Decided(ordering) => ordering,
            // SAFETY: `not_found` is safe to call because
            //         we're in `Verify` mode.
            Comparison::Unequal | Comparison::Unknown => unsafe { not_found() },
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
