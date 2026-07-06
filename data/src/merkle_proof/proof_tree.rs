// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT

use std::borrow::Borrow;
use std::marker::PhantomData;

use bincode::enc::write::Writer;

use super::Deserialiser;
use super::DeserialiserNode;
use super::FromProof;
use super::Partial;
use super::ProofError;
use super::Suspended;
use super::tag::LeafTag;
use super::tag::Tag;
use crate::codec::Bincode;
use crate::codec::LeafCodec;
use crate::codec::LeafDecode;
use crate::foldable::Fold;
use crate::foldable::Foldable;
use crate::foldable::NodeFold;
use crate::hash::Hash;
use crate::tree::Tree;

/// Merkle proof tree structure.
///
/// Leaves can be read and/or written to.
/// If a read was done, the content will be stored in the proof.
/// If a write was done, the written content is not necessary since the semantics of running the step will
/// deduce the written contents.
///
/// A proof has the shape of a subtree of the full PVM state.
/// The full state layout is fixed by the state's [`Foldable`] implementation,
/// so node arity and leaf sizes are known statically.
/// Therefore, neither the proof nor its encoding needs to store that metadata.
pub type MerkleProof = Tree<MerkleProofLeaf>;

impl MerkleProof {
    /// Create a new Merkle proof as a read leaf.
    pub fn leaf_read(data: Vec<u8>) -> Self {
        MerkleProof::Leaf(MerkleProofLeaf::Read(data))
    }

    /// Create a new Merkle proof as a blind leaf.
    pub fn leaf_blind(hash: Hash) -> Self {
        MerkleProof::Leaf(MerkleProofLeaf::Blind(hash))
    }

    /// Compute the root hash of the Merkle proof.
    pub fn root_hash(&self) -> Hash {
        // Child nodes are stored in normal order in `nodes`.
        let mut nodes: Vec<(&MerkleProof, usize)> = vec![(self, 0)];
        // Child nodes are stored in reverse order in `hash_states`.
        let mut hash_states: Vec<HashState> = vec![];

        while let Some((node, parent_index)) = nodes.pop() {
            match node {
                Tree::Leaf(MerkleProofLeaf::Blind(hash)) => {
                    hash_states.push(HashState::new_leaf(parent_index, *hash));
                }
                Tree::Leaf(MerkleProofLeaf::Read(data)) => {
                    hash_states.push(HashState::new_leaf(
                        parent_index,
                        Hash::hash_bytes(data.as_slice()),
                    ));
                }
                Tree::Node(node) => {
                    hash_states.push(HashState::new_node(parent_index));
                    let new_parent_index = hash_states.len() - 1;
                    for child in node.children.iter() {
                        nodes.push((child, new_parent_index));
                    }
                }
            }
        }

        while hash_states.len() > 1 {
            let hash_state = hash_states
                .pop()
                .expect("hash_states can't be empty at this point");
            // Note that child hashes are added in normal order to `hash_states`.
            hash_states[hash_state.parent_index()].push(hash_state.hash());
        }

        // Hash states is not empty at this point.
        hash_states[0].hash()
    }

    /// Is this proof a blind leaf?
    pub(crate) fn is_blind(&self) -> bool {
        matches!(self, MerkleProof::Leaf(MerkleProofLeaf::Blind(_)))
    }

    /// Fold the given data structure into a compressed Merkle proof tree (bincode leaf codec).
    pub fn from_foldable(foldable: &impl Foldable<MerkleProofFold>) -> Self {
        foldable.fold(MerkleProofFold::new()).tree
    }

    /// Fold the given data structure into a compressed Merkle proof tree, using the given leaf
    /// [`LeafCodec`].
    pub fn from_foldable_with<C: LeafCodec>(foldable: &impl Foldable<MerkleProofFold<C>>) -> Self {
        foldable.fold(MerkleProofFold::<C>::new()).tree
    }

    /// Blind the tree as much as feasible. In general, this will fully blind the tree.
    ///
    /// If the tree is a 'short leaf' (ie, the size of the data is less than [Hash::DIGEST_SIZE])
    /// then the tree will not be blinded, to save proof space. A parent node that is blinded
    /// subsequently will still omit the leaf.
    fn blind(&self) -> Self {
        match self {
            Self::Leaf(MerkleProofLeaf::Read(data)) if data.len() < Hash::DIGEST_SIZE => {
                Self::Leaf(MerkleProofLeaf::Read(data.clone()))
            }
            tree => {
                let hash = tree.root_hash();
                Self::leaf_blind(hash)
            }
        }
    }
}

impl From<&MerkleProof> for Tag {
    fn from(value: &MerkleProof) -> Self {
        match value {
            MerkleProof::Node(_) => Tag::Node,
            MerkleProof::Leaf(MerkleProofLeaf::Blind(_)) => Tag::Leaf(LeafTag::Blind),
            MerkleProof::Leaf(MerkleProofLeaf::Read(_)) => Tag::Leaf(LeafTag::Read),
        }
    }
}

impl bincode::Encode for MerkleProof {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        let mut nodes = vec![self];

        while let Some(node) = nodes.pop() {
            match node {
                Self::Node(node) => {
                    Tag::Node.encode(encoder)?;

                    // We add the children in reverse order so that when we pop them from the
                    // `nodes` stack, they are in the original order.
                    nodes.extend(node.children.iter().rev());
                }

                Self::Leaf(MerkleProofLeaf::Read(data)) => {
                    Tag::Leaf(LeafTag::Read).encode(encoder)?;

                    // We want to write the raw data, and avoid the bincode length prefix. The decoder
                    // will know how many bytes to read.
                    encoder.writer().write(data.as_slice())?;
                }

                Self::Leaf(MerkleProofLeaf::Blind(hash)) => {
                    Tag::Leaf(LeafTag::Blind).encode(encoder)?;
                    hash.encode(encoder)?;
                }
            }
        }

        Ok(())
    }
}

/// Type used to describe the leaves of a [`MerkleProof`].
/// For more details see the documentation of [`MerkleProof`].
#[derive(Clone, Debug, PartialEq)]
pub enum MerkleProofLeaf {
    /// A leaf that is not read. It may be written.
    /// Contains the hash of the contents from initial state.
    ///
    /// Note: a blinded leaf can correspond to a blinded subtree in a
    /// [`crate::merkle_proof::proof_tree::MerkleProof`] due to compression.
    Blind(Hash),
    /// A leaf that is read. It may also be written.
    /// Contains the read data from the initial state.
    /// The initial hash can be deduced based on the read data.
    Read(Vec<u8>),
}

/// Whether a part of the tree must be present, may be blinded or may be omitted
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MinimumPresence {
    /// The tree may be omitted.
    MayOmit,

    /// The tree may be blinded, but cannot be omitted.
    MayBlind,

    /// The tree must be present and cannot be blinded or omitted.
    Present,
}

impl MinimumPresence {
    /// Infer the minimum presence requirement for a node based on the minimum presence requirements
    /// of its children.
    pub fn for_node<MP, CS>(children: CS) -> Self
    where
        MP: Borrow<Self>,
        CS: IntoIterator<Item = MP>,
    {
        let may_omit_all = children
            .into_iter()
            .all(|child| *child.borrow() <= MinimumPresence::MayOmit);

        // When there is at least one child that is either `Present` or `MayBlind`, then the node
        // must be `Present`. It is important to remember, that this node must not be compressed
        // into a blinded tree when the child trees requested not to be omitted. The compression is
        // prevented by marking this node as `Present`.

        if may_omit_all {
            MinimumPresence::MayOmit
        } else {
            MinimumPresence::Present
        }
    }
}

/// Merkle tree compression information
///
/// This structure should not be constructible by itself, it must always be paired with a
/// [`MerkleProofFold`] via the [`Foldable`] implementation. Consider using these functions
/// instead:
///
/// - [`MerkleProofFold::into_leaf`] to create a leaf proof using the builder
/// - [`MerkleProofFold::new_leaf`] to create a leaf that needs to be folded into a node
#[derive(Clone)]
pub struct CompressibleMerkleProof {
    /// Constraints on the presence of this subtree within the full Merkle proof tree
    constraint: MinimumPresence,

    /// Merkle proof tree that may be compressed
    tree: MerkleProof,
}

impl CompressibleMerkleProof {
    /// Create a new compressible Merkle tree proof leaf.
    ///
    /// This method must be private! See note on [`MerkleProofFold::new_leaf`] for details.
    fn new_leaf(constraint: MinimumPresence, data: Vec<u8>) -> Self {
        let mut tree = MerkleProof::leaf_read(data);

        // If the leaf does not need to be present, we can compress it into a blinded tree already.
        if constraint < MinimumPresence::Present {
            tree = tree.blind()
        }

        CompressibleMerkleProof { constraint, tree }
    }
}

impl<C: LeafCodec> Foldable<MerkleProofFold<C>> for CompressibleMerkleProof {
    fn fold(&self, _builder: MerkleProofFold<C>) -> <MerkleProofFold<C> as Fold>::Folded {
        if self.constraint == MinimumPresence::Present || self.tree.is_blind() {
            return self.clone();
        }

        CompressibleMerkleProof {
            constraint: self.constraint,
            // Here we know the presence constraint is not `Present`, so the tree can be compressed
            // into a blinded tree if it is not already.
            tree: self.tree.blind(),
        }
    }
}

/// Elevate the minimum presence requirement of a Merkle proof tree `Foldable`
///
/// When `T` gets folded using a `MerkleProofFold`, the result is a proof tree with a presence
/// constraint attached to it. This struct allows you to impose a minimum on the presence
/// constraint.
pub struct ForceMinimumPresence<T> {
    /// Minimum presence requirement for the proof tree resulting from folding `T`
    pub min_constraint: MinimumPresence,

    /// Subject that gets folded into a proof tree with the given minimum presence requirement
    pub inner: T,
}

impl<C: LeafCodec, T: Foldable<MerkleProofFold<C>>> Foldable<MerkleProofFold<C>>
    for ForceMinimumPresence<T>
{
    fn fold(&self, builder: MerkleProofFold<C>) -> <MerkleProofFold<C> as Fold>::Folded {
        let mut proof = self.inner.fold(builder);
        proof.constraint = self.min_constraint.max(proof.constraint);
        proof
    }
}

/// [`Fold`] for creating a [`MerkleProof`] tree from a foldable structure
///
/// Parameterised by the leaf [`LeafCodec`]; defaults to [`Bincode`].
pub struct MerkleProofFold<C = Bincode> {
    _codec: PhantomData<C>,
}

impl<C: LeafCodec> MerkleProofFold<C> {
    /// Create a new fold builder for Merkle proofs.
    ///
    /// NOTE: This should be private! We don't want users to create this directly.
    fn new() -> Self {
        MerkleProofFold {
            _codec: PhantomData,
        }
    }

    /// Fold into a Merkle tree proof leaf.
    pub fn into_leaf(self, constraint: MinimumPresence, data: Vec<u8>) -> CompressibleMerkleProof {
        CompressibleMerkleProof::new_leaf(constraint, data)
    }

    /// Create a new compressible Merkle tree proof from a leaf.
    ///
    /// # Return type
    ///
    /// This is intentionally not a constructor for [`CompressibleMerkleProof`].
    ///
    /// We want every leaf to pass through [`CompressibleMerkleProof::fold`], where compression can
    /// convert it into a blind leaf.
    ///
    /// To enforce that, this function returns an opaque [`Foldable`] value that must be consumed by
    /// a node folder such as [`MerkleProofNodeFold`]. Returning [`CompressibleMerkleProof`]
    /// directly would allow implementations of [`Foldable::fold`] to bypass that step and skip
    /// compression.
    pub fn new_leaf(constraint: MinimumPresence, data: Vec<u8>) -> impl Foldable<Self> {
        CompressibleMerkleProof::new_leaf(constraint, data)
    }

    /// Create a blinded proof leaf from a known hash.
    ///
    /// Use this when a subtree was not accessed during proof generation and should be represented
    /// only by its hash in the proof.
    /// TODO: RV-968 - remove workaround for creating CompressibleMerkleProof, as users can also be
    /// in "may-blind" state.
    pub fn into_blind(self, hash: Hash) -> CompressibleMerkleProof {
        CompressibleMerkleProof {
            constraint: MinimumPresence::MayOmit,
            tree: MerkleProof::leaf_blind(hash),
        }
    }
}

impl<C: LeafCodec> Fold for MerkleProofFold<C> {
    type Folded = CompressibleMerkleProof;

    type NodeFold = MerkleProofNodeFold<C>;

    type Codec = C;

    fn into_node_fold(self) -> Self::NodeFold {
        MerkleProofNodeFold {
            children: Vec::new(),
            _codec: PhantomData,
        }
    }
}

/// [`NodeFold`] for creating a [`MerkleProof`] node from a foldable structure
pub struct MerkleProofNodeFold<C = Bincode> {
    /// Children of the node that is being folded
    children: Vec<CompressibleMerkleProof>,

    _codec: PhantomData<C>,
}

impl<C: LeafCodec> NodeFold for MerkleProofNodeFold<C> {
    type Parent = MerkleProofFold<C>;

    fn add<F: Foldable<Self::Parent>>(&mut self, child: &F) {
        let child_info = child.fold(MerkleProofFold::new());
        self.children.push(child_info);
    }

    fn done(self) -> <Self::Parent as Fold>::Folded {
        let presence_constraint =
            MinimumPresence::for_node(self.children.iter().map(|child| &child.constraint));

        let mut tree = MerkleProof::node_without_data(
            self.children.into_iter().map(|child| child.tree).collect(),
        );

        if let MinimumPresence::MayOmit | MinimumPresence::MayBlind = presence_constraint {
            let hash = tree.root_hash();
            tree = MerkleProof::leaf_blind(hash);
        }

        CompressibleMerkleProof {
            constraint: presence_constraint,
            tree,
        }
    }
}

/// [`enum@HashState`] is associated with the state of hashing a [`MerkleProof`].
/// We record whether the node is a leaf or an internal node, the index of its parent(
/// see [`MerkleProof::root_hash`] for more details) and the hashes of its children
/// if it's a node and its own hash if its a leaf.
enum HashState {
    Node {
        parent_index: usize,
        hashes: Vec<Hash>,
    },
    Leaf {
        parent_index: usize,
        hash: Hash,
    },
}

impl HashState {
    fn new_leaf(parent_index: usize, hash: Hash) -> Self {
        HashState::Leaf { parent_index, hash }
    }

    fn new_node(parent_index: usize) -> Self {
        HashState::Node {
            parent_index,
            hashes: vec![],
        }
    }

    /// Push a hash to node's hash list.
    ///
    /// # Panics
    ///
    /// Panics if the hash state is a Leaf.
    fn push(&mut self, hash: Hash) {
        match self {
            HashState::Node { hashes, .. } => hashes.push(hash),
            _ => unreachable!("A leaf node must not have children"),
        }
    }

    fn hash(&self) -> Hash {
        match self {
            HashState::Node { hashes, .. } => Hash::combine_hashes(hashes),
            HashState::Leaf { hash, .. } => *hash,
        }
    }

    fn parent_index(&self) -> usize {
        match self {
            HashState::Node { parent_index, .. } | HashState::Leaf { parent_index, .. } => {
                *parent_index
            }
        }
    }
}

/// Part of a tree that may be absent
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProofPart<T> {
    /// This part of the tree is absent.
    Absent,

    /// There is a proof for this part of the tree.
    Present(T),
}

impl<T> ProofPart<T> {
    /// Obtain the inner value, if present. Otherwise, return `None`.
    pub fn into_present(self) -> Option<T> {
        match self {
            Self::Present(inner) => Some(inner),
            Self::Absent => None,
        }
    }
}

/// Part of a Merkle proof tree, viewed as a deserialiser parameterised by the leaf [`LeafCodec`].
///
/// Defaults to [`Bincode`]. Only leaf decoding depends on the codec; the tree structure is shared.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProofTree<'a, C = Bincode> {
    part: ProofPart<&'a MerkleProof>,
    _codec: PhantomData<C>,
}

impl<'a, C> ProofTree<'a, C> {
    /// A present proof tree backed by the given [`MerkleProof`].
    pub fn present(tree: &'a MerkleProof) -> Self {
        ProofTree {
            part: ProofPart::Present(tree),
            _codec: PhantomData,
        }
    }

    /// An absent proof tree.
    pub fn absent() -> Self {
        ProofTree {
            part: ProofPart::Absent,
            _codec: PhantomData,
        }
    }

    /// Deserialise the proof tree as a leaf.
    fn as_leaf(&self) -> Result<Partial<&'a [u8]>, ProofError> {
        let ProofPart::Present(tree) = self.part else {
            return Ok(Partial::Absent);
        };

        let leaf = match tree {
            Tree::Leaf(MerkleProofLeaf::Blind(hash)) => Partial::Blinded(*hash),
            Tree::Leaf(MerkleProofLeaf::Read(items)) => Partial::Present(items.as_slice()),
            Tree::Node(_) => return Err(ProofError::UnexpectedNode),
        };

        Ok(leaf)
    }

    /// Deserialise the proof tree as a node.
    fn as_node(&self) -> Result<Partial<Vec<Self>>, ProofError> {
        let ProofPart::Present(tree) = self.part else {
            return Ok(Partial::Absent);
        };

        let node = match tree {
            Tree::Leaf(leaf) => match leaf {
                MerkleProofLeaf::Blind(hash) => Partial::Blinded(*hash),
                MerkleProofLeaf::Read(_) => return Err(ProofError::UnexpectedLeaf),
            },
            Tree::Node(node) => Partial::Present(
                node.children
                    .iter()
                    .map(|c| ProofTree::present(c))
                    .collect(),
            ),
        };

        Ok(node)
    }
}

impl<'t, C: LeafCodec> Deserialiser for ProofTree<'t, C> {
    type Error = ProofError;

    type Codec = C;

    type Suspended<R> = ProofTreeResult<'t, C, R>;

    type DeserialiserNode = Partial<std::vec::IntoIter<Self>>;

    fn into_leaf_raw<const LEN: usize>(
        self,
    ) -> Result<Self::Suspended<Partial<Box<[u8; LEN]>>>, Self::Error> {
        self.as_leaf()?
            .map_present_fallible(|data| {
                let data_len = data.len();
                let bytes: Box<[u8; LEN]> =
                    data.to_vec()
                        .try_into()
                        .map_err(|_| ProofError::UnexpectedLeafSize {
                            expected: LEN,
                            got: data_len,
                        })?;
                Ok(bytes)
            })
            .map(ProofTreeResult::new)
    }

    fn into_leaf<T: LeafDecode<C>>(self) -> Result<Self::Suspended<Partial<T>>, Self::Error> {
        let result = self
            .as_leaf()?
            .map_present_fallible(<T as LeafDecode<C>>::leaf_decode)?;
        Ok(ProofTreeResult::new(result))
    }

    fn into_node(self) -> Result<Self::DeserialiserNode, Self::Error> {
        Ok(self.as_node()?.map_present(Vec::into_iter))
    }

    fn capture_owned_proof(&self) -> Option<MerkleProof> {
        match self.part {
            ProofPart::Absent => None,
            ProofPart::Present(proof) => Some(proof.clone()),
        }
    }
}

/// Similar to [`ProofTree`], but owns the underlying [`MerkleProof`]
pub type OwnedProofTree = ProofPart<MerkleProof>;

impl OwnedProofTree {
    /// Obtain an [`OwnedProofTree`] from a [`Partial<T>`] considering it a leaf.
    pub fn leaf_from_partial<T>(partial: Partial<T>, f: impl FnOnce(T) -> Vec<u8>) -> Self {
        match partial {
            Partial::Absent => OwnedProofTree::Absent,
            Partial::Blinded(hash) => OwnedProofTree::Present(MerkleProof::leaf_blind(hash)),
            Partial::Present(data) => OwnedProofTree::Present(MerkleProof::leaf_read(f(data))),
        }
    }

    /// Construct a node from its child proofs. The `parent` parameter allows us to reconstruct the
    /// blinded state of the parent.
    pub fn node_from_children<I>(parent: Partial<()>, children: I) -> Self
    where
        I: IntoIterator<Item = Self>,
        I::IntoIter: ExactSizeIterator,
    {
        match parent {
            Partial::Absent => return OwnedProofTree::Absent,
            Partial::Blinded(hash) => {
                return OwnedProofTree::Present(MerkleProof::leaf_blind(hash));
            }
            Partial::Present(_) => {}
        }

        let children = children.into_iter();
        let mut partial_children = Vec::with_capacity(children.len());

        for item in children {
            match item {
                OwnedProofTree::Absent => return OwnedProofTree::Absent,
                OwnedProofTree::Present(tree) => partial_children.push(tree),
            }
        }

        OwnedProofTree::Present(MerkleProof::node_without_data(partial_children))
    }
}

impl<'t, C: LeafCodec, BS: Iterator<Item = ProofTree<'t, C>>> DeserialiserNode for Partial<BS> {
    type Parent = ProofTree<'t, C>;

    fn presence(&self) -> Partial<()> {
        match self {
            Partial::Absent => Partial::Absent,
            Partial::Blinded(hash) => Partial::Blinded(*hash),
            Partial::Present(_) => Partial::Present(()),
        }
    }

    fn next_branch_with<T>(
        mut self,
        branch_deserialiser: impl FnOnce(
            Self::Parent,
        ) -> Result<
            <Self::Parent as Deserialiser>::Suspended<T>,
            ProofError,
        >,
    ) -> Result<(Self, T), ProofError> {
        let next_branch = match self {
            // If the node is absent or blinded, the branch to be deserialised as a tree is absent.
            Partial::Absent | Partial::Blinded(_) => ProofTree::absent(),
            Partial::Present(ref mut branches) => {
                branches.next().ok_or(ProofError::BadNumberOfBranches {
                    expected: 1,
                    got: 0,
                })?
            }
        };

        let result = branch_deserialiser(next_branch)?.result;
        Ok((self, result))
    }

    fn done<T>(self, value: T) -> Result<<Self::Parent as Deserialiser>::Suspended<T>, ProofError> {
        if let Partial::Present(branches) = self {
            let remaining_items = branches.count();
            if remaining_items > 0 {
                return Err(ProofError::BadNumberOfBranches {
                    expected: 0,
                    got: remaining_items,
                });
            }
        }

        Ok(ProofTreeResult {
            result: value,
            _pd: PhantomData,
        })
    }
}

/// Result of parsing a [`ProofTree`]
pub struct ProofTreeResult<'t, C, R> {
    result: R,
    _pd: PhantomData<fn(ProofTree<'t, C>)>,
}

impl<C, R> ProofTreeResult<'_, C, R> {
    /// Construct a new result.
    fn new(result: R) -> Self {
        Self {
            result,
            _pd: PhantomData,
        }
    }

    /// Unwrap the result type.
    pub fn into_result(self) -> R {
        self.result
    }
}

impl<'t, C: LeafCodec, R> Suspended for ProofTreeResult<'t, C, R> {
    type Output = R;

    type Parent = ProofTree<'t, C>;

    fn map<T>(
        self,
        f: impl FnOnce(Self::Output) -> T,
    ) -> <Self::Parent as Deserialiser>::Suspended<T> {
        ProofTreeResult {
            result: f(self.result),
            _pd: PhantomData,
        }
    }
}

/// Given a [`ProofTree`] deserialise it as `T`.
pub fn deserialise<C: LeafCodec, T: FromProof<C>>(
    proof: ProofTree<C>,
) -> Result<(T, OwnedProofTree), ProofError> {
    let owned_proof = match proof.part {
        ProofPart::Absent => OwnedProofTree::Absent,
        ProofPart::Present(proof) => OwnedProofTree::Present(proof.clone()),
    };

    let parser = T::from_proof(proof)?;
    let result = parser.into_result();

    Ok((result, owned_proof))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialisation::bincode_default_config;

    #[test]
    fn merkle_proofs_can_be_encoded() {
        let merkle_proofs = [
            MerkleProof::leaf_read([1, 2, 3].to_vec()),
            MerkleProof::leaf_blind(Hash::hash_bytes(&[1, 3, 4])),
            Tree::node_without_data(
                [
                    MerkleProof::leaf_read([1, 2, 3].to_vec()),
                    MerkleProof::leaf_blind(Hash::hash_bytes(&[1, 3, 4])),
                ]
                .to_vec(),
            ),
        ];
        for merkle_proof in merkle_proofs.iter() {
            bincode::encode_to_vec(merkle_proof, bincode_default_config())
                .expect("Failed to encode the merkle proof");
        }
    }

    #[test]
    fn we_can_take_the_merkle_proof_of_the_root_hash() {
        let node = Tree::node_without_data(
            [
                MerkleProof::leaf_read([1, 2, 3].to_vec()),
                MerkleProof::leaf_blind(Hash::hash_bytes(&[1, 3, 4])),
            ]
            .to_vec(),
        );
        let _ = node.root_hash();
    }

    #[test]
    fn child_node_hashes_are_pushed_back_in_normal_order() {
        let merkle_proof = Tree::node_without_data(
            [
                Tree::node_without_data(
                    [
                        MerkleProof::leaf_read([1, 2, 3].to_vec()),
                        MerkleProof::leaf_blind(Hash::hash_bytes(&[4, 5, 6])),
                    ]
                    .to_vec(),
                ),
                MerkleProof::leaf_blind(Hash::hash_bytes(&[7, 8, 9])),
            ]
            .to_vec(),
        );

        let calculated_root_hash = merkle_proof.root_hash();

        let mut first_child_node = HashState::new_node(0);
        first_child_node.push(Hash::hash_bytes(&[1, 2, 3]));
        first_child_node.push(Hash::hash_bytes(&[4, 5, 6]));

        let mut root_node = HashState::new_node(0);
        root_node.push(first_child_node.hash());
        root_node.push(Hash::hash_bytes(&[7, 8, 9]));

        assert_eq!(root_node.hash(), calculated_root_hash);
    }

    struct Leaf {
        constraint: MinimumPresence,
        data: &'static [u8],
    }

    impl Foldable<MerkleProofFold> for Leaf {
        fn fold(&self, builder: MerkleProofFold) -> <MerkleProofFold as Fold>::Folded {
            builder.into_leaf(self.constraint, self.data.to_vec())
        }
    }

    #[test]
    fn proof_tree_leaf_present() {
        static DATA: &[u8] = b"hello";

        assert_eq!(
            MerkleProof::from_foldable(&Leaf {
                constraint: MinimumPresence::Present,
                data: DATA,
            }),
            MerkleProof::leaf_read(DATA.to_vec())
        );
    }

    #[test]
    fn proof_tree_leaf_blindable() {
        static DATA: &[u8] = &[17; Hash::DIGEST_SIZE];
        let hash = Hash::hash_bytes(DATA);

        assert_eq!(
            MerkleProof::from_foldable(&Leaf {
                constraint: MinimumPresence::MayBlind,
                data: DATA
            }),
            MerkleProof::leaf_blind(hash)
        );
    }

    #[test]
    fn proof_tree_leaf_omittable() {
        static DATA: &[u8] = &[19; Hash::DIGEST_SIZE];
        let hash = Hash::hash_bytes(DATA);

        assert_eq!(
            MerkleProof::from_foldable(&Leaf {
                constraint: MinimumPresence::MayOmit,
                data: DATA
            }),
            MerkleProof::leaf_blind(hash)
        );
    }

    #[test]
    fn proof_tree_leaf_omittable_short() {
        static DATA: &[u8] = b"hello";
        assert_eq!(
            MerkleProof::from_foldable(&Leaf {
                constraint: MinimumPresence::MayOmit,
                data: DATA
            }),
            MerkleProof::leaf_read(DATA.to_vec())
        );
    }

    struct Node<T> {
        children: Vec<T>,
    }

    impl<T: Foldable<MerkleProofFold>> Foldable<MerkleProofFold> for Node<T> {
        fn fold(&self, builder: MerkleProofFold) -> <MerkleProofFold as Fold>::Folded {
            let mut node = builder.into_node_fold();

            for child in self.children.iter() {
                node.add(child);
            }

            node.done()
        }
    }

    #[test]
    fn proof_tree_node_present_children() {
        let node = Node {
            children: vec![
                Leaf {
                    constraint: MinimumPresence::Present,
                    data: b"hello",
                },
                Leaf {
                    constraint: MinimumPresence::Present,
                    data: b"world",
                },
            ],
        };

        let expected = Tree::node_without_data(vec![
            MerkleProof::leaf_read(b"hello".to_vec()),
            MerkleProof::leaf_read(b"world".to_vec()),
        ]);

        assert_eq!(MerkleProof::from_foldable(&node), expected);
    }

    #[test]
    fn proof_tree_node_present_child_omittable_child() {
        static DATA_OMIT: &[u8] = &[129; Hash::DIGEST_SIZE];

        let node = Node {
            children: vec![
                Leaf {
                    constraint: MinimumPresence::Present,
                    data: b"hello",
                },
                Leaf {
                    constraint: MinimumPresence::MayOmit,
                    data: DATA_OMIT,
                },
            ],
        };

        let expected = Tree::node_without_data(vec![
            MerkleProof::leaf_read(b"hello".to_vec()),
            MerkleProof::leaf_blind(Hash::hash_bytes(DATA_OMIT)),
        ]);

        assert_eq!(MerkleProof::from_foldable(&node), expected);
    }

    #[test]
    fn proof_tree_node_present_child_omittable_child_short() {
        let node = Node {
            children: vec![
                Leaf {
                    constraint: MinimumPresence::Present,
                    data: b"hello",
                },
                Leaf {
                    constraint: MinimumPresence::MayOmit,
                    data: b"world",
                },
            ],
        };

        let expected = Tree::node_without_data(vec![
            MerkleProof::leaf_read(b"hello".to_vec()),
            MerkleProof::leaf_read(b"world".to_vec()),
        ]);

        assert_eq!(MerkleProof::from_foldable(&node), expected);
    }

    #[test]
    fn proof_tree_node_present_child_blindable_child() {
        let node = Node {
            children: vec![
                Leaf {
                    constraint: MinimumPresence::Present,
                    data: b"hello",
                },
                Leaf {
                    constraint: MinimumPresence::MayBlind,
                    data: &[15; Hash::DIGEST_SIZE],
                },
            ],
        };

        let expected = Tree::node_without_data(vec![
            MerkleProof::leaf_read(b"hello".to_vec()),
            MerkleProof::leaf_blind(Hash::hash_bytes(&[15; Hash::DIGEST_SIZE])),
        ]);

        assert_eq!(MerkleProof::from_foldable(&node), expected);
    }

    #[test]
    fn proof_tree_node_present_child_blindable_child_short() {
        let node = Node {
            children: vec![
                Leaf {
                    constraint: MinimumPresence::Present,
                    data: b"hello",
                },
                Leaf {
                    constraint: MinimumPresence::MayBlind,
                    data: b"world",
                },
            ],
        };

        let expected = Tree::node_without_data(vec![
            MerkleProof::leaf_read(b"hello".to_vec()),
            MerkleProof::leaf_read(b"world".to_vec()),
        ]);

        assert_eq!(MerkleProof::from_foldable(&node), expected);
    }

    #[test]
    fn proof_tree_node_blindable_children() {
        let node = Node {
            children: vec![
                Leaf {
                    constraint: MinimumPresence::MayBlind,
                    data: &[81; Hash::DIGEST_SIZE + 5],
                },
                Leaf {
                    constraint: MinimumPresence::MayBlind,
                    data: b"world",
                },
            ],
        };

        let expected = Tree::node_without_data(vec![
            MerkleProof::leaf_blind(Hash::hash_bytes(&[81; Hash::DIGEST_SIZE + 5])),
            MerkleProof::leaf_read(b"world".to_vec()),
        ]);

        assert_eq!(MerkleProof::from_foldable(&node), expected);
    }

    #[test]
    fn proof_tree_node_blindable_child_omittable_child() {
        let node = Node {
            children: vec![
                Leaf {
                    constraint: MinimumPresence::MayBlind,
                    data: &[29; Hash::DIGEST_SIZE],
                },
                Leaf {
                    constraint: MinimumPresence::MayOmit,
                    data: &[36; Hash::DIGEST_SIZE + 1],
                },
            ],
        };

        let expected = Tree::node_without_data(vec![
            MerkleProof::leaf_blind(Hash::hash_bytes(&[29; Hash::DIGEST_SIZE])),
            MerkleProof::leaf_blind(Hash::hash_bytes(&[36; Hash::DIGEST_SIZE + 1])),
        ]);

        assert_eq!(MerkleProof::from_foldable(&node), expected);
    }

    #[test]
    fn proof_tree_node_blindable_child_omittable_child_short() {
        let node = Node {
            children: vec![
                Leaf {
                    constraint: MinimumPresence::MayBlind,
                    data: b"hello",
                },
                Leaf {
                    constraint: MinimumPresence::MayOmit,
                    data: b"world",
                },
            ],
        };

        let expected = Tree::node_without_data(vec![
            MerkleProof::leaf_read(b"hello".to_vec()),
            MerkleProof::leaf_read(b"world".to_vec()),
        ]);

        assert_eq!(MerkleProof::from_foldable(&node), expected);
    }

    #[test]
    fn proof_tree_node_omittable_children() {
        let node = Node {
            children: vec![
                Leaf {
                    constraint: MinimumPresence::MayOmit,
                    data: b"hello",
                },
                Leaf {
                    constraint: MinimumPresence::MayOmit,
                    data: b"world",
                },
            ],
        };

        let expected = MerkleProof::leaf_blind(Hash::combine_hashes([
            Hash::hash_bytes(b"hello"),
            Hash::hash_bytes(b"world"),
        ]));

        assert_eq!(MerkleProof::from_foldable(&node), expected);
    }
}
