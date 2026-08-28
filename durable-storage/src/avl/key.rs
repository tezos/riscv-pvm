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
//!   the subtrees before it are equal, which the verifier re-checks against the
//!   subtree hash, and the subtree after it is never compared. Only that
//!   one page has to be fully included in the proof, so an ordering costs a that
//!   page, plus the hash of a precending/subsequent subtree/page.
//! - Keys where one is a prefix of the other may agree on every page they share
//!   and are additionally ordered by their lengths.
//!
//! [`PartialOrd`]: std::cmp::PartialOrd

use std::cell::Cell;
use std::cmp::Ordering;
use std::convert::Infallible;
use std::ops::Deref;
use std::ops::Range;

use bincode::Decode;
use bincode::Encode;
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
use octez_riscv_data::foldable::seq_tree::tree_depth;
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
/// halving the page size trades 32 bytes of page content for a 34-byte layer.
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
fn first_differing_page(lhs: &[u8], rhs: &[u8]) -> Option<usize> {
    (0..page_count(lhs.len()).min(page_count(rhs.len())))
        .find(|&page| lhs[page_range(lhs.len(), page)] != rhs[page_range(rhs.len(), page)])
}

/// Where a chunk of pages sits within a key's page tree.
///
/// A chunk is one subtree of the layout [`IndexableSeqAsTree`] gives the pages,
/// which is what lets a verifier recompute the hash a blinded subtree carries
/// from the pages of the key it is comparing against: a chunk the two keys agree
/// on folds to the same hash whatever else the keys hold.
#[derive(Clone, Copy, Debug)]
struct Chunk {
    /// Pages of the whole tree this chunk belongs to, not of the chunk itself.
    total_pages: usize,

    /// Index of the chunk's first page.
    start: usize,

    /// Depth of the chunk within the tree. A chunk at depth zero is a single
    /// page, which the layout does not wrap in a node.
    depth: u32,
}

impl Chunk {
    /// The whole page tree of a key of `total_pages` pages.
    fn root(total_pages: usize) -> Self {
        Chunk {
            total_pages,
            start: 0,
            depth: tree_depth(total_pages, NODE_ARITY),
        }
    }

    /// One past the last page this chunk covers.
    fn end(self) -> usize {
        let span = NODE_ARITY.saturating_pow(self.depth);

        self.start.saturating_add(span).min(self.total_pages)
    }

    /// The chunks of this chunk's children, in order.
    ///
    /// Mirrors how [`IndexableSeqAsTree`] lays out a node's children: each spans
    /// a [`NODE_ARITY`]th of this one - which at the layer above the leaves is a
    /// single page each - and children starting past the end of the sequence are
    /// dropped rather than left empty. Only called on a chunk that is a node, so
    /// never at depth zero.
    fn children(self) -> impl Iterator<Item = Chunk> {
        let depth = self.depth.saturating_sub(1);
        let span = NODE_ARITY.saturating_pow(depth);

        (0..NODE_ARITY)
            .map(move |child| self.start + child * span)
            .take_while(move |&start| start < self.total_pages)
            .map(move |start| Chunk {
                start,
                depth,
                ..self
            })
    }
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

    // TODO TZX-192: the empty key has no pages, so its page tree is a node with no children,
    // which no comparison marks and a proof therefore blinds to a full-length hash. Carrying
    // such a node in full instead - it encodes shorter than a digest, as blinding already does
    // for leaves shorter than one - is what stops an ordering against the empty key costing
    // more than one against a key of a few bytes, whose single page is already cheaper revealed
    // than hashed.
    let page = page_leaf(bytes);
    builder.add(&IndexableSeqAsTree::new(
        page_count(bytes.len()),
        NODE_ARITY,
        &page,
    ));

    builder.done()
}

/// The pages of `bytes`, as leaves to fold a page tree from.
fn page_leaf<'key>(bytes: &'key [u8]) -> impl Fn(usize) -> EncodeLeaf<PageRef<'key>> {
    move |index| {
        EncodeLeaf::new(
            PageRef(&bytes[page_range(bytes.len(), index)]),
            "Should be able to serialise a key's page",
        )
    }
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

/// The hash the pages of `key` fold to over the given chunk of a page tree.
///
/// The chunk's shape comes from the key the proof was generated for, and its
/// contents from `key` - so this is the hash the blinded subtree would carry if
/// the two keys agreed over that chunk.
fn chunk_hash<C: LeafCodec>(key: &Key, chunk: Chunk) -> Hash
where
    for<'page> PageRef<'page>: LeafEncode<C>,
{
    let page = page_leaf(key.as_ref());

    if chunk.depth == 0 {
        return Hash::from_foldable_with::<C>(&page(chunk.start));
    }

    Hash::from_foldable_with::<C>(&IndexableSeqAsTree::chunk(
        chunk.total_pages,
        NODE_ARITY,
        &page,
        chunk.start,
        chunk.depth,
    ))
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
            access: self.0.pages[index].get(),
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

        builder.add(&length.clone().map_present(|length| {
            PartialHash::Present(
                Hash::hash_leaf::<C, _>(&length).expect("Hashing a key's length should not fail"),
            )
        }));
        builder.add(pages);

        builder.done()
    }
}

/// As much of a key's page tree as a proof carries.
///
/// A comparison leaves most of the tree blinded, and a blinded subtree is kept
/// here as the hash it folds to rather than dropped: that hash is what lets the
/// pages under it be checked in one go, against the same chunk of the key being
/// compared. Flattening the tree to a list of pages would lose it.
#[derive(Clone, Debug)]
enum PageTree {
    /// Nothing of this subtree is in the proof.
    Absent,

    /// Only the hash this subtree folds to.
    Blinded {
        /// Hash the subtree folds to.
        hash: Hash,

        /// Hashes the same chunk of another key's pages, so that the two can be
        /// compared.
        ///
        /// Captured while deserialising, where the [`LeafCodec`] is known.
        hash_chunk: fn(&Key, Chunk) -> Hash,
    },

    /// One page's bytes.
    Page(Vec<u8>),

    /// A node of the tree, with its children in order.
    Node(Vec<PageTree>),
}

impl PageTree {
    /// The tree of a key whose every page is known.
    fn present(bytes: &[u8], chunk: Chunk) -> Self {
        if chunk.depth == 0 {
            return PageTree::Page(bytes[page_range(bytes.len(), chunk.start)].to_vec());
        }

        PageTree::Node(
            chunk
                .children()
                .map(|child| PageTree::present(bytes, child))
                .collect(),
        )
    }

    /// Build the tree of a page leaf from its deserialised proof leaf.
    fn leaf<C: LeafCodec>(page: Partial<Page>) -> Self
    where
        for<'page> PageRef<'page>: LeafEncode<C>,
    {
        match page {
            Partial::Absent => PageTree::Absent,
            Partial::Blinded(hash) => PageTree::Blinded {
                hash,
                hash_chunk: chunk_hash::<C>,
            },
            Partial::Present(page) => PageTree::Page(page.0),
        }
    }
}

impl<C: LeafCodec> Foldable<PartialHashFold<C>> for PageTree
where
    for<'page> PageRef<'page>: LeafEncode<C>,
{
    fn fold(&self, builder: PartialHashFold<C>) -> <PartialHashFold<C> as Fold>::Folded {
        match self {
            PageTree::Absent => builder.previous(),
            PageTree::Blinded { hash, .. } => PartialHash::Present(*hash),
            PageTree::Page(page) => PartialHash::Present(page_hash::<C>(page)),
            PageTree::Node(children) => {
                let mut builder = builder.into_node_fold();

                for child in children {
                    builder.add(child);
                }

                builder.done()
            }
        }
    }
}

/// Rebuild as much of a key's page tree as the proof carries, keeping the hash of
/// every subtree it blinded.
///
/// The recursion follows the layout [`IndexableSeqAsTree`] gives the pages, and
/// stops at anything that is not present - a blinded subtree has nothing below it
/// to read. Its depth is bounded by the key's `u8` length, so a proof cannot
/// choose how far it descends.
fn page_tree_from_proof<C: LeafCodec, Proof: Deserialiser<Codec = C>>(
    proof: Proof,
    chunk: Chunk,
) -> SuspendedResult<Proof, PageTree>
where
    Page: LeafDecode<C>,
    for<'page> PageRef<'page>: LeafEncode<C>,
{
    if chunk.depth == 0 {
        return Ok(proof.into_leaf::<Page>()?.map(PageTree::leaf::<C>));
    }

    let mut node = proof.into_node()?;

    match node.presence() {
        Partial::Absent => return node.done(PageTree::Absent),
        Partial::Blinded(hash) => {
            return node.done(PageTree::Blinded {
                hash,
                hash_chunk: chunk_hash::<C>,
            });
        }
        Partial::Present(()) => {}
    }

    let mut children = Vec::new();

    for child in chunk.children() {
        let (next, tree) =
            node.next_branch_with(|proof| page_tree_from_proof::<C, _>(proof, child))?;

        node = next;
        children.push(tree);
    }

    node.done(PageTree::Node(children))
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
                // Without the length there is no telling what shape the page tree
                // has, so it must not be present either.
                let proof = proof.into_node()?;

                if let Partial::Present(()) = proof.presence() {
                    return Err(DeserialiserError::custom(
                        KeyProofError::LengthAbsentButPagesPresent,
                    ));
                }

                return proof.done(PageTree::Absent);
            };

            page_tree_from_proof::<C, _>(proof, Chunk::root(page_count(*length as usize)))
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
    /// The key itself.
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
    fn record_equality(&self) {
        self.whole.set(self.whole.get().max(KeyAccess::Hashed));
    }

    /// Record an ordering of this key against `rhs`, which must not be equal to it.
    fn record_ordering(&self, rhs: &Key) {
        // Every page comparison starts from the key's length, which fixes where
        // its pages end.
        self.length.set(KeyAccess::Read);

        let bytes = self.inner.as_ref();
        let pages = page_count(bytes.len());
        let shared = pages.min(page_count(rhs.as_ref().len()));

        // Only the boundary page is marked. The path down to it stays in the
        // proof, which leaves the subtrees spanning the pages before it blinded as
        // units - one hash to check per layer rather than one per page.
        match first_differing_page(bytes, rhs.as_ref()) {
            Some(page) => self.record_page(page, KeyAccess::Read),

            // The keys agree on every shared page, so their lengths order them -
            // but the shared pages still have to be checkable, and hashing the
            // first page past them holds the path open.
            None if shared < pages => self.record_page(shared, KeyAccess::Hashed),

            // The tree spans exactly the shared pages, so it blinds as one.
            None => {}
        }
    }

    /// Record an access to one page, keeping the most revealing one so far.
    fn record_page(&self, index: usize, access: KeyAccess) {
        let page = &self.pages[index];
        page.set(page.get().max(access));
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
        this.key.record_equality();
        this.key.inner.eq(rhs)
    }

    fn cmp(this: &NodeKey<Self>, rhs: &Key) -> Ordering {
        let ordering = this.key.inner.cmp(rhs);

        if ordering.is_eq() {
            // An exact match is an equality check, which the verifier can redo
            // against the hash of the blinded key.
            this.key.record_equality();
        } else {
            this.key.record_ordering(rhs);
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

        /// As much of the key's page tree as the proof carries.
        pages: PageTree,
    },
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

/// Compare one chunk of a key's pages against `rhs`, whose first `shared` pages
/// are the ones the comparison reaches.
///
/// `None` means every page of the chunk that the comparison reaches matches.
fn compare_chunk(tree: &PageTree, chunk: Chunk, shared: usize, rhs: &Key) -> Option<Comparison> {
    // Pages past the ones the two keys share take no part in the comparison: the
    // keys are ordered before them, or by their lengths.
    if chunk.start >= shared {
        return None;
    }

    match tree {
        PageTree::Absent => Some(Comparison::Unknown),

        PageTree::Blinded { hash, hash_chunk } => {
            // A chunk reaching past the shared pages holds pages of this key that
            // `rhs` has no counterpart for, so its hash cannot be reproduced.
            if chunk.end() > shared {
                return Some(Comparison::Unknown);
            }

            // Pages fold injectively, so equal chunk hashes mean the two keys hold
            // the same bytes right across the chunk.
            (*hash != hash_chunk(rhs, chunk)).then_some(Comparison::Unequal)
        }

        PageTree::Page(page) => {
            let bytes = rhs.as_ref();

            // The keys are ordered within the first page on which they differ.
            match page
                .as_slice()
                .cmp(&bytes[page_range(bytes.len(), chunk.start)])
            {
                Ordering::Equal => None,
                ordering => Some(Comparison::Decided(ordering)),
            }
        }

        PageTree::Node(children) => children
            .iter()
            .zip(chunk.children())
            .find_map(|(tree, chunk)| compare_chunk(tree, chunk, shared, rhs)),
    }
}

impl VerifyImpl {
    /// Compare this key against `rhs`, as far as the proof allows.
    fn compare(&self, rhs: &Key) -> Comparison {
        let (length, tree) = match self {
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
        let rhs_length = rhs.as_ref().len();
        let pages = page_count(length);
        let shared = pages.min(page_count(rhs_length));

        match compare_chunk(tree, Chunk::root(pages), shared, rhs) {
            // Keys that agree on every page they share are ordered by their lengths.
            None => Comparison::Decided(length.cmp(&rhs_length)),

            // Keys of different lengths are different whatever their pages say, so
            // a proof too thin to order them still settles that much. This is what
            // an equality check falls back on when some other comparison has put
            // the key's tree in the proof: the chunk spanning the pages past the
            // shared ones is blinded, and it cannot be checked against a key that
            // has no such pages.
            Some(Comparison::Unknown) if length != rhs_length => Comparison::Unequal,

            Some(comparison) => comparison,
        }
    }
}

impl NodeKeyMode for Verify {}

impl private::NodeKeyImpl for Verify {
    fn new(key: Key) -> NodeKey<Self> {
        let bytes = key.as_ref();

        NodeKey {
            key: VerifyImpl::Tree {
                length: Partial::Present(key_length(&key)),
                pages: PageTree::present(bytes, Chunk::root(page_count(bytes.len()))),
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
