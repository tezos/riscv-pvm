// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Merkle proofs

pub mod proof;
pub mod proof_binary;
pub mod proof_tree;
pub mod tag;

use std::error;

use bincode::error::DecodeError;

use crate::codec::Bincode;
use crate::codec::LeafCodec;
use crate::codec::LeafDecode;
use crate::codec::LeafDecodeError;
use crate::foldable::Foldable;
use crate::foldable::seq_tree::tree_depth;
use crate::hash::Hash;
use crate::hash::PartialHash;
use crate::hash::PartialHashFold;
use crate::merkle_proof::proof_tree::MerkleProof;

/// Possible outcomes when parsing a node or a leaf from a Merkle proof
/// where the leaf is assumed to have type `T`.
#[derive(Clone, Debug)]
pub enum Partial<T> {
    /// The leaf or node is absent from the proof.
    Absent,

    /// A blinded subtree and its [`struct@Hash`] is provided.
    Blinded(Hash),

    /// Data successfully parsed and its type is `T`.
    Present(T),
}

impl<T> Partial<T> {
    /// Map the present result of a [`Partial<T>`] into [`Partial<R>`].
    pub fn map_present<R>(self, f: impl FnOnce(T) -> R) -> Partial<R> {
        match self {
            Partial::Absent => Partial::Absent,
            Partial::Blinded(hash) => Partial::Blinded(hash),
            Partial::Present(data) => Partial::Present(f(data)),
        }
    }

    /// Same as [`Partial::map_present`] but can fail.
    pub fn map_present_fallible<R, E>(
        self,
        f: impl FnOnce(T) -> Result<R, E>,
    ) -> Result<Partial<R>, E> {
        match self {
            Partial::Absent => Ok(Partial::Absent),
            Partial::Blinded(hash) => Ok(Partial::Blinded(hash)),
            Partial::Present(data) => Ok(Partial::Present(f(data)?)),
        }
    }

    /// Convert a [`Partial<T>`] into an [`Option<T>`], discarding blinded and absent cases.
    pub fn to_present(self) -> Option<T> {
        match self {
            Partial::Present(data) => Some(data),
            Partial::Absent | Partial::Blinded(_) => None,
        }
    }
}

impl<C: LeafCodec, T: Foldable<PartialHashFold<C>>> Foldable<PartialHashFold<C>> for Partial<T> {
    fn fold(&self, builder: PartialHashFold<C>) -> PartialHash {
        match self {
            Partial::Absent => builder.previous(),
            Partial::Blinded(hash) => builder.present(*hash),
            Partial::Present(data) => data.fold(builder),
        }
    }
}

/// Error type that can occur during proof deserialisation
pub trait DeserialiserError: error::Error {
    /// Create a custom deserialiser error from any error type.
    fn custom<E: error::Error + 'static>(error: E) -> Self;
}

/// Errors occurring when parsing a Merkle proof
#[derive(Debug, thiserror::Error)]
pub enum ProofError {
    #[error("Error during deserialisation: {0}")]
    Deserialise(#[from] DecodeError),

    #[error("Error during leaf deserialisation: {0}")]
    LeafDecode(#[from] LeafDecodeError),

    #[error("Deserialising as a stream and not all bytes were consumed")]
    RemainingBytes,

    #[error("Proof tree is absent")]
    AbsentProof,

    #[error("Encountered a node with a bad number of branches: expected {expected}, got {got}")]
    BadNumberOfBranches { expected: usize, got: usize },

    #[error("Expected a leaf of size {expected}, got {got}")]
    UnexpectedLeafSize { expected: usize, got: usize },

    #[error("Encountered a leaf where a node was expected")]
    UnexpectedLeaf,

    #[error("Encountered a node where a leaf was expected")]
    UnexpectedNode,

    #[error("Custom error: {0}")]
    Custom(Box<dyn std::error::Error>),
}

impl DeserialiserError for ProofError {
    fn custom<E: error::Error + 'static>(error: E) -> Self {
        // SAFETY: `ProofError` does not contain lifetimes, so unty-ing is safe.
        match unsafe { unty::unty(error) } {
            Ok(this) => this,
            Err(error) => Self::Custom(Box::new(error)),
        }
    }
}

/// [`Deserialiser::Suspended<T>`] wrapped in a [`Result`].
///
/// Clippy yells at us for making the type signatures in [`Deserialiser`] too complex, so we
/// provide some aliases to simplify them on the surface level.
pub type SuspendedResult<D, T> =
    Result<<D as Deserialiser>::Suspended<T>, <D as Deserialiser>::Error>;

/// The main trait used for deserialising a proof.
///
/// Having an object of this trait is equivalent to having a proof and being able to deserialise it.
///
/// A proof can be interpreted in 3 cases:
/// 1. [`Deserialiser::into_leaf_raw`] The proof is a leaf and raw bytes are obtained.
/// 2. [`Deserialiser::into_leaf<T>`] The proof is a leaf and the type `T` is parsed.
/// 3. [`Deserialiser::into_node`] The proof is a node in the tree.
pub trait Deserialiser {
    /// Error type
    type Error: DeserialiserError;

    /// The codec used to decode leaf values from the proof.
    type Codec: LeafCodec;

    /// After deserialising a proof, a [`Suspended<R>`] computation is obtained.
    type Suspended<R>: Suspended<Output = R, Parent = Self>;

    /// In case the proof is a node, [`Deserialiser::DeserialiserNode`] is the deserialiser for the branch case.
    type DeserialiserNode: DeserialiserNode<Parent = Self>;

    /// It is expected for the proof to be a leaf. Obtain the raw bytes from that leaf.
    fn into_leaf_raw<const LEN: usize>(self) -> SuspendedResult<Self, Partial<Box<[u8; LEN]>>>;

    /// It is expected for the proof to be a leaf. Parse the raw bytes of that leaf into a type `T`.
    fn into_leaf<T: LeafDecode<Self::Codec>>(self) -> SuspendedResult<Self, Partial<T>>;

    /// It is expected for the proof to be a node. Obtain the deserialiser for the branch case.
    fn into_node(self) -> Result<Self::DeserialiserNode, Self::Error>;

    /// Capture an owned snapshot of the proof at the current position, if available.
    ///
    /// Deserialisers backed by an in-memory proof tree (e.g. [`crate::merkle_proof::proof_tree`])
    /// can clone their internal `MerkleProof`. Stream deserialisers cannot reconstruct the proof
    /// without re-parsing, so they return `None` (the default).
    ///
    /// Used by verify-mode types (notably the AVL `VerifyNodeId` / `VerifyTreeId`) that need to
    /// retain their original sub-proof so they can fold against it later, even after the working
    /// tree has been structurally rearranged.
    ///
    // TODO RV-994: Investigate avoiding the extra allocation required for this.
    // TODO TZX-161: StreamDeserialiser should allow capturing owned proof
    fn capture_owned_proof(&self) -> Option<MerkleProof> {
        None
    }
}

/// The trait used for deserialising a proof's node.
/// Having an object of this trait is equivalent to knowing the current proof is a node.
/// Deserialisers for each of its branches are expected to be provided to continue the deserialisation.
pub trait DeserialiserNode: Sized {
    type Parent: Deserialiser;

    /// Get the presence information for the node that is being parsed.
    fn presence(&self) -> Partial<()>;

    /// The next branch of the current node is deserialised using the provided deserialiser
    /// `branch_deserialiser`.
    fn next_branch_with<T>(
        self,
        branch_deserialiser: impl FnOnce(Self::Parent) -> SuspendedResult<Self::Parent, T>,
    ) -> Result<(Self, T), <Self::Parent as Deserialiser>::Error>;

    /// The next branch of the current node is deserialised using the [`FromProof`] implementation
    /// of type `T`.
    fn next_branch<T: FromProof<<Self::Parent as Deserialiser>::Codec>>(
        self,
    ) -> Result<(Self, T), <Self::Parent as Deserialiser>::Error> {
        self.next_branch_with(|deser| T::from_proof(deser))
    }

    /// Signal the end of deserialisation of the node's branches.
    /// Call this method after all calls to [`DeserialiserNode::next_branch`] have been made.
    fn done<T>(self, value: T) -> SuspendedResult<Self::Parent, T>;
}

/// The trait represents a computation function obtained after deserialising a proof.
pub trait Suspended {
    /// End result of the computation.
    type Output;

    type Parent: Deserialiser;

    /// Helper to map the current result into a new type.
    fn map<T>(
        self,
        f: impl FnOnce(Self::Output) -> T,
    ) -> <Self::Parent as Deserialiser>::Suspended<T>;
}

/// Trait for types that can be constructed from a Merkle proof whose leaves use codec `C`.
///
/// Parameterised by the leaf [`LeafCodec`]; defaults to [`Bincode`] so existing `FromProof`
/// references keep the historical (bincode) proof format.
pub trait FromProof<C: LeafCodec = Bincode>: Sized {
    /// Parse the given proof to construct an instance of `Self`.
    fn from_proof<Proof: Deserialiser<Codec = C>>(proof: Proof) -> SuspendedResult<Proof, Self>;
}

impl<C: LeafCodec, A: FromProof<C>, B: FromProof<C>> FromProof<C> for (A, B) {
    fn from_proof<Proof: Deserialiser<Codec = C>>(proof: Proof) -> SuspendedResult<Proof, Self> {
        let proof = proof.into_node()?;

        let (proof, a) = proof.next_branch()?;
        let (proof, b) = proof.next_branch()?;

        proof.done((a, b))
    }
}

impl<C: LeafCodec, Item: FromProof<C>, const LEN: usize> FromProof<C> for [Item; LEN] {
    fn from_proof<Proof: Deserialiser<Codec = C>>(proof: Proof) -> SuspendedResult<Proof, Self> {
        let proof = proof.into_node()?;

        let mut items: [Option<Item>; LEN] = std::array::from_fn(|_| None);
        let proof = (0..LEN).try_fold(proof, |proof, index| {
            let (proof, item) = proof.next_branch()?;
            items[index] = Some(item);
            Ok(proof)
        })?;

        let items = items.map(|opt| opt.expect("All items should have been filled"));
        proof.done(items)
    }
}

// Internal helper that mirrors `IndexableSeqAsTree::fold`: it keeps a fixed total length and
// decrements depth on each recursive step.
fn descend_tree_helper<Proof, LeafHandler>(
    proof: Proof,
    arity: usize,
    total_leaves: usize,
    current_depth: u32,
    current_start: usize,
    for_leaf: &mut LeafHandler,
) -> SuspendedResult<Proof, ()>
where
    LeafHandler: FnMut(usize, Proof) -> SuspendedResult<Proof, ()>,
    Proof: Deserialiser,
{
    let mut ctx = proof.into_node()?;

    // An absent or blinded node carries no information about its descendants, so descending into
    // one costs time proportional to `total_leaves` - a figure the proof itself supplies - rather
    // than to the size of the proof. Stop here instead.
    //
    // This is a correctness requirement rather than an optimisation. Nothing bounds what a proof
    // may claim here, and hashing is no help: a proof carrying an astronomically large length can
    // still hash correctly, so there is no later check that would catch it. The bound has to be
    // structural.
    //
    // Pruning does not change the proof that gets reconstructed. Both deserialisers hand out
    // absent branches beneath an absent or blinded node without consuming any input, and
    // `OwnedProofTree::node_from_children` discards the children of such a node outright, so the
    // recovered tree - and the state hash taken from it - is identical either way.
    if !matches!(ctx.presence(), Partial::Present(())) {
        return ctx.done(());
    }

    // Time to add leaves.
    if current_depth <= 1 {
        for idx in current_start..current_start + arity {
            if idx >= total_leaves {
                break;
            }

            let (new_ctx, ()) = ctx.next_branch_with(|ctx| for_leaf(idx, ctx))?;
            ctx = new_ctx;
        }

        return ctx.done(());
    }

    let next_chunk_len = arity.pow(current_depth - 1);

    for child_no in 0..arity {
        let next_start = current_start + child_no * next_chunk_len;

        if next_start >= total_leaves {
            break;
        }

        let (new_ctx, ()) = ctx.next_branch_with(|ctx| {
            descend_tree_helper(
                ctx,
                arity,
                total_leaves,
                current_depth - 1,
                next_start,
                for_leaf,
            )
        })?;
        ctx = new_ctx;
    }

    ctx.done(())
}

/// Descend a Merkle proof tree, calling `for_leaf` on each leaf encountered. The tree is described
/// to have a number of leaves equal to the given `total_leaves`. The node arity is given by
/// `arity`. Leaves are encountered in depth-first order left-to-right.
pub fn descend_tree<Proof, LeafHandler>(
    proof: Proof,
    arity: usize,
    total_leaves: usize,
    for_leaf: &mut LeafHandler,
) -> SuspendedResult<Proof, ()>
where
    LeafHandler: FnMut(usize, Proof) -> SuspendedResult<Proof, ()>,
    Proof: Deserialiser,
{
    // For compatibility with the previous Merklisation scheme, a single-leaf root is represented
    // as a leaf rather than a node.
    if total_leaves == 1 {
        return for_leaf(0, proof);
    }

    let depth = tree_depth(total_leaves, arity);

    descend_tree_helper(proof, arity, total_leaves, depth, 0, for_leaf)
}

/// Parse proof as a sequence with length represented as a Merkle tree.
///
/// # Merkle Tree Shape
///
/// The Merkle tree is split into length and contents at the root.
///
/// ```text
///    root
///    /  \
/// len    contents
/// ```
///
/// The length node is a leaf containing the length of the sequence. It must be present when any
/// node or leaf in the contents subtree is present.
///
/// The contents subtree contains the leaves which represent elements from the sequence.
///
/// With `arity = 2`:
///
/// ```text
///       N0
///      /  \
///    N1    N2
///   / \    / \
/// L0  L1  L2  L3
/// ```
///
/// L0, L1, etc are the leaves containing the sequenced items.
pub fn sequence_as_tree_from_proof<Length, State, Proof>(
    proof: Proof,
    arity: usize,
    with_length: impl FnOnce(Partial<Length>) -> (State, Partial<usize>),
    mut with_item: impl FnMut(&mut State, usize, Proof) -> SuspendedResult<Proof, ()>,
) -> SuspendedResult<Proof, State>
where
    Length: LeafDecode<Proof::Codec>,
    Proof: Deserialiser,
{
    let proof = proof.into_node()?;

    let (proof, length) = proof.next_branch_with(|proof| proof.into_leaf())?;
    let (mut state, length) = with_length(length);

    let (proof, state) = proof.next_branch_with(|proof| {
        // When the length node is present, we can properly parse all leaves.
        // But when the length node is not present, we cannot parse any leaves. This needs to be
        // validated. In other words, the node for the sequence items must be blinded or absent.
        let Partial::Present(len) = length else {
            // XXX: We can't pick whether this is a node or leaf given we don't know the
            // length. However, absent or blinded leaves are encoded the same way as nodes.
            // In the case where the node is present (which is an error in here), we would
            // trigger an unexpected leaf error instead of the more appropriate error below.
            let proof = proof.into_node()?;

            // When the node for the items is present, that's a problem. There may be items and
            // we don't know how to extract them because we don't know how many there are.
            if let Partial::Present(_) = proof.presence() {
                return Err(DeserialiserError::custom(
                    SeqTreeProofError::LengthAbsentButItemsPresent,
                ));
            }

            return proof.done(state);
        };

        let mut for_leaf = |idx, proof: Proof| with_item(&mut state, idx, proof);

        let result = descend_tree(proof, arity, len, &mut for_leaf)?;
        Ok(result.map(|()| state))
    })?;

    proof.done(state)
}

/// Errors indicating a bad proof for a sequence-as-tree
#[derive(Debug, thiserror::Error)]
enum SeqTreeProofError {
    #[error("Length node is absent but some item nodes are present")]
    LengthAbsentButItemsPresent,
}

#[cfg(test)]
mod tests;
