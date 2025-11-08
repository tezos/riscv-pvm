// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::collections::VecDeque;

use bincode::Decode;
use bincode::Encode;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use perfect_derive::perfect_derive;

use super::AllocatedOf;
use super::Array;
use super::Atom;
use super::DynArray;
use super::Layout;
use super::Many;
use super::Ref;
use super::RefProofGenOwnedAlloc;
use super::RefVerifierAlloc;
use super::hash::Hash;
use super::hash::HashError;
use super::owned_backend::Owned;
use super::proof_backend::merkle::MERKLE_ARITY;
use super::proof_backend::merkle::MERKLE_LEAF_SIZE;
use super::proof_backend::merkle::MerkleTree;
use super::proof_backend::merkle::MerkleWriter;
use super::proof_backend::merkle::build_custom_merkle_tree;
use super::proof_backend::merkle::chunks_to_writer;
use super::proof_backend::proof::MerkleProof;
use super::proof_backend::proof::MerkleProofLeaf;
use super::proof_backend::proof::deserialiser::Deserialiser;
use super::proof_backend::proof::deserialiser::DeserialiserNode;
use super::proof_backend::proof::deserialiser::Result;
use super::proof_backend::proof::deserialiser::Suspended;
use super::proof_backend::tree::Tree;
use super::verify_backend;
use super::verify_backend::PartialState;
use super::verify_backend::Verifier;
use crate::array_utils::boxed_array;
use crate::state_backend::proof_backend::proof::InvalidTagError;
use crate::state_backend::proof_backend::proof::NotEnoughBytesError;
use crate::state_backend::proof_backend::proof::deserialiser::Partial;
use crate::state_backend::verify_backend::PageId;
use crate::storage::binary;

/// Errors occurring when parsing the tag structure of a Merkle proof.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum TagError {
    #[error("Invalid tag encountered: {0}")]
    InvalidTag(#[from] InvalidTagError),

    #[error("Not enough bytes available")]
    NotEnoughBytes(#[from] NotEnoughBytesError),
}

/// Errors occurring when parsing a Merkle proof
#[derive(Debug, thiserror::Error)]
pub enum ProofError {
    #[error("Error during deserialisation: {0}")]
    Deserialise(#[from] DecodeError),

    #[error("Not enough bytes")]
    NotEnoughBytes(#[from] NotEnoughBytesError),

    #[error("Deserialising as a stream and not all bytes were consumed")]
    RemainingBytes,

    #[error("Error during tag deserialisation: {0}")]
    TagDeserialise(#[from] TagError),

    #[error("Proof tree is absent")]
    AbsentProof,

    #[error("A part of the proof required to parse further is absent")]
    DependentNodeIsAbsent,

    #[error("Encountered a node with a bad number of branches: expected {expected}, got {got}")]
    BadNumberOfBranches { expected: usize, got: usize },

    #[error("Expected a leaf of size {expected}, got {got}")]
    UnexpectedLeafSize { expected: usize, got: usize },

    #[error("Encountered a leaf where a node was expected")]
    UnexpectedLeaf,

    #[error("Encountered a node where a leaf was expected")]
    UnexpectedNode,
}

/// Common result type for parsing a Merkle proof.
pub(crate) type VerifierAllocResult<D, L> =
    Result<<D as Deserialiser>::Suspended<VerifierAlloc<L>>>;

/// Regions for the verifier backend for a specific layout.
pub type VerifierAlloc<L> = <L as Layout>::Allocated<verify_backend::Verifier>;

/// Errors that may occur when hashing a [`verify_backend::Verifier`] state
#[derive(Debug, thiserror::Error)]
pub enum PartialHashError {
    /// The hash could not be computed because encoding a value to bytes failed. The byte
    /// representation is used as input to the hash function.
    #[error("Error while encoding a to-be-hashed value: {0}")]
    Encode(#[from] EncodeError),

    #[error("Error from proof: {0}")]
    FromProof(#[from] ProofError),

    /// Indicates that a hash could not be computed due to absent data,
    /// but from which it is possible to recover if the level at which
    /// it was raised is part of a blinded subtree and its hash is present
    /// in the proof.
    #[error("Potentially recoverable error")]
    PotentiallyRecoverable,

    /// Indicates that a hash could not be computed because the data being
    /// hashed is only partially available.
    #[error("Fatal error")]
    Fatal,
}

/// Part of a tree that may be absent
#[derive(Debug, PartialEq)]
#[perfect_derive(Clone, Copy)]
pub enum ProofPart<'a, T: ?Sized> {
    /// This part of the tree is absent.
    Absent,

    /// There is a proof for this part of the tree.
    Present(&'a T),
}

/// Part of a Merkle proof tree
pub type ProofTree<'a> = ProofPart<'a, MerkleProof>;

impl<'a> ProofTree<'a> {
    /// Interpret this part of the Merkle proof as a node with `LEN` branches.
    pub fn into_branches<const LEN: usize>(self) -> Result<Box<[Self; LEN]>> {
        let ProofTree::Present(proof) = self else {
            // The requested branches are not represented in the Merkle proof at all, not even
            // through a blinded node.
            return Ok(boxed_array![ProofTree::Absent; LEN]);
        };

        match proof {
            Tree::Node(branches) => {
                let branches: &[MerkleProof; LEN] =
                    branches.as_slice().try_into().map_err(|_| {
                        ProofError::BadNumberOfBranches {
                            got: branches.len(),
                            expected: LEN,
                        }
                    })?;
                Ok(branches
                    .iter()
                    .map(ProofTree::Present)
                    .collect::<Vec<_>>()
                    .try_into()
                    .map_err(|_| {
                        unreachable!(
                            "Converting a vector to an array of the same size always succeeds"
                        )
                    })
                    .unwrap())
            }

            Tree::Leaf(leaf) => match leaf {
                MerkleProofLeaf::Blind(_hash) => Ok(boxed_array![ProofTree::Absent; LEN]),
                _ => Err(ProofError::UnexpectedLeaf)?,
            },
        }
    }

    /// Interpret this part of the Merkle proof as a leaf.
    pub fn into_leaf(self) -> Result<ProofPart<'a, [u8]>> {
        if let ProofTree::Present(proof) = self {
            match proof {
                Tree::Node(_) => Err(ProofError::UnexpectedNode),
                Tree::Leaf(leaf) => match leaf {
                    MerkleProofLeaf::Blind(_) => Ok(ProofPart::Absent),
                    MerkleProofLeaf::Read(data) => Ok(ProofPart::Present(data.as_slice())),
                },
            }
        } else {
            Ok(ProofPart::Absent)
        }
    }

    /// For the purpose of computing the final hash of a `Verifier` state,
    /// interpret this part of a Merkle proof as a leaf and return its hash if
    /// it is a blinded leaf or hash the data if it is present.
    pub(crate) fn partial_hash_leaf(self) -> Result<Hash, PartialHashError> {
        let ProofTree::Present(proof) = self else {
            return Err(PartialHashError::PotentiallyRecoverable);
        };

        let Tree::Leaf(leaf) = proof else {
            return Err(ProofError::UnexpectedNode.into());
        };

        let hash = match leaf {
            MerkleProofLeaf::Blind(hash) => *hash,
            MerkleProofLeaf::Read(data) => Hash::blake3_hash_bytes(data),
        };

        Ok(hash)
    }

    /// For the purpose of computing the final hash of a `Verifier` state,
    /// if present, try to interpret this part of a Merkle proof as:
    /// - a node with `LEN` branches, in which case return the proof branches
    ///   and no proof hash
    /// - a blinded leaf which corresponds to a node with `LEN` children,
    ///   in which case return absent branches and the proof hash
    ///
    /// If the proof tree is absent, return absent branches and no proof hash.
    pub fn into_branches_with_hash<const LEN: usize>(
        self,
    ) -> Result<(Box<[ProofTree<'a>; LEN]>, Option<Hash>), PartialHashError> {
        let ProofTree::Present(proof) = self else {
            return Ok((boxed_array![ProofTree::Absent; LEN], None));
        };

        match proof {
            Tree::Node(branches) if branches.len() != LEN => Err(PartialHashError::FromProof(
                ProofError::BadNumberOfBranches {
                    got: branches.len(),
                    expected: LEN,
                },
            )),
            Tree::Node(branches) => Ok((
                branches
                    .iter()
                    .map(ProofTree::Present)
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
                    .try_into()
                    .map_err(|_| PartialHashError::Fatal)?,
                None,
            )),
            Tree::Leaf(leaf) => match leaf {
                MerkleProofLeaf::Blind(hash) => {
                    Ok((boxed_array![ProofTree::Absent; LEN], Some(*hash)))
                }
                _ => Err(ProofError::UnexpectedLeaf)?,
            },
        }
    }
}

/// Similar to [`ProofPart`], but owns the underlying [`MerkleProof`].
#[derive(Clone)]
pub enum OwnedProofPart {
    /// This part of the tree is absent.
    Absent,
    /// There is a proof for this part of the tree.
    Present(MerkleProof),
}

impl OwnedProofPart {
    /// Obtain an [`OwnedProofPart`] from a [`Partial<T>`] considering it a leaf.
    pub fn leaf_from_partial<T>(partial: Partial<T>, f: impl FnOnce(T) -> Vec<u8>) -> Self {
        match partial {
            Partial::Absent => OwnedProofPart::Absent,
            Partial::Blinded(hash) => OwnedProofPart::Present(MerkleProof::leaf_blind(hash)),
            Partial::Present(data) => OwnedProofPart::Present(MerkleProof::leaf_read(f(data))),
        }
    }

    /// Obtain an [`OwnedProofPart`] from a [`Partial<Vec<MerkleProof>>`] considering it a node.
    pub fn node_from_partial(partial: Partial<Vec<MerkleProof>>) -> Self {
        match partial {
            Partial::Absent => OwnedProofPart::Absent,
            Partial::Blinded(hash) => OwnedProofPart::Present(MerkleProof::leaf_blind(hash)),
            Partial::Present(children) => OwnedProofPart::Present(MerkleProof::Node(children)),
        }
    }

    /// Construct a node from its child proofs. The `parent` parameter allows us to restruct the
    /// blinded state of the parent.
    pub fn node_from_children(
        parent: Partial<()>,
        children: impl IntoIterator<Item = Self>,
    ) -> Self {
        match parent {
            Partial::Absent => return OwnedProofPart::Absent,
            Partial::Blinded(hash) => {
                return OwnedProofPart::Present(MerkleProof::leaf_blind(hash));
            }
            Partial::Present(_) => {}
        }

        let mut partial_children = Vec::with_capacity(MERKLE_ARITY);

        for item in children {
            match item {
                OwnedProofPart::Absent => return OwnedProofPart::Absent,
                OwnedProofPart::Present(tree) => partial_children.push(tree),
            }
        }

        OwnedProofPart::Present(MerkleProof::Node(partial_children))
    }
}

/// [`Layouts`] which may be used in a Merkle proof
///
/// [`Layouts`]: crate::state_backend::Layout
pub trait ProofLayout: Layout {
    /// Obtain the complete Merkle tree which captures an execution trace
    /// using the proof-generating backend.
    fn to_merkle_tree(state: RefProofGenOwnedAlloc<Self>) -> Result<MerkleTree, HashError>;

    /// Parse a Merkle proof into the allocated form of this layout.
    fn into_verifier_alloc<D: Deserialiser>(proof: D) -> VerifierAllocResult<D, Self>;

    /// Compute the state hash of a partial `Verifier` state using its
    /// corresponding proof tree where data is missing.
    fn partial_state_hash(
        state: RefVerifierAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError>;
}

impl<T: ProofLayout> ProofLayout for Box<T> {
    fn to_merkle_tree(state: RefProofGenOwnedAlloc<Self>) -> Result<MerkleTree, HashError> {
        T::to_merkle_tree(*state)
    }

    fn into_verifier_alloc<D: Deserialiser>(proof: D) -> VerifierAllocResult<D, Self> {
        Ok(T::into_verifier_alloc(proof)?.map(Box::new))
    }

    fn partial_state_hash(
        state: RefVerifierAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        T::partial_state_hash(*state, proof)
    }
}

impl<T> ProofLayout for Atom<T>
where
    T: Encode + Decode<()> + 'static,
{
    fn to_merkle_tree(state: RefProofGenOwnedAlloc<Self>) -> Result<MerkleTree, HashError> {
        // The Merkle leaf must hold the serialisation of the initial state.
        // Directly serialising the `ProofGen` state would produce the serialisation
        // of the final state. Therefore, we rebind and serialise the wrapped `Owned` state.
        let region = state.into_region();
        let access_info = region.get_access_info();
        let cell = super::Cell::<T, Ref<'_, Owned>>::bind(region.inner_region_ref());
        let serialised = binary::serialise(&cell)?;
        Ok(MerkleTree::make_merkle_leaf(serialised, access_info))
    }

    fn into_verifier_alloc<D: Deserialiser>(proof: D) -> VerifierAllocResult<D, Self> {
        let f = Array::<T, 1>::into_verifier_alloc(proof)?;
        Ok(f.map(super::Cell::from))
    }

    fn partial_state_hash(
        state: RefVerifierAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        let region = state.into_region();
        match region.get_partial_region() {
            PartialState::Complete(region) => Ok(Hash::blake3_hash(region)?),
            PartialState::Absent => proof.partial_hash_leaf(),
            PartialState::Incomplete => Err(PartialHashError::Fatal),
        }
    }
}

impl<T, const LEN: usize> ProofLayout for Array<T, LEN>
where
    T: Encode + Decode<()> + 'static,
{
    fn to_merkle_tree(state: RefProofGenOwnedAlloc<Self>) -> Result<MerkleTree, HashError> {
        // RV-282: Break down into multiple leaves if the size of the `Cells`
        // is too large for a proof.
        //
        // The Merkle leaf must hold the serialisation of the initial state.
        // Directly serialising the `ProofGen` state would produce the serialisation
        // of the final state. Therefore, we rebind and serialise the wrapped `Owned` state.
        let region = state.into_region();
        let access_info = region.get_access_info();
        let cells = super::Cells::<T, LEN, Ref<'_, Owned>>::bind(region.inner_region_ref());
        let serialised = binary::serialise(&cells)?;
        Ok(MerkleTree::make_merkle_leaf(serialised, access_info))
    }

    fn into_verifier_alloc<D: Deserialiser>(proof: D) -> VerifierAllocResult<D, Self> {
        use super::proof_backend::proof::deserialiser::Partial;

        Ok(proof
            .into_leaf::<super::Cells<T, LEN, Owned>>()?
            .map(|region| {
                let region = match region {
                    Partial::Absent | Partial::Blinded(_) => verify_backend::Region::Absent,
                    Partial::Present(cells) => {
                        let arr: Box<[Option<T>; LEN]> = Box::new(cells.into_region().map(Some));
                        verify_backend::Region::Partial(arr)
                    }
                };
                super::Cells::bind(region)
            }))
    }

    fn partial_state_hash(
        state: RefVerifierAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        let region = state.into_region();
        match region.get_partial_region() {
            PartialState::Complete(region) => Ok(Hash::blake3_hash(region)?),
            PartialState::Absent => proof.partial_hash_leaf(),
            PartialState::Incomplete => Err(PartialHashError::Fatal),
        }
    }
}

impl ProofLayout for DynArray {
    fn to_merkle_tree(state: RefProofGenOwnedAlloc<Self>) -> Result<MerkleTree, HashError> {
        let len = state.len();

        let region = state.region_ref();
        let mut writer = MerkleWriter::new(
            MERKLE_LEAF_SIZE,
            MERKLE_ARITY,
            region.get_read(),
            region.get_write(),
            len.div_ceil(MERKLE_ARITY),
        );
        let read = |address| -> [u8; MERKLE_LEAF_SIZE.get()] {
            // SAFETY: The chunk writer will only request data within the given bounds. The bounds
            // are set to the length of the dynamic array.
            unsafe { region.inner_dyn_region_read(address) }
        };
        chunks_to_writer::<_, _>(&mut writer, len, read)?;

        let pages_node = writer.finalise()?;

        let length_node = MerkleTree::make_merkle_leaf(
            binary::serialise(len as u64)?,
            region.need_length_in_proof(),
        );

        let root_node = MerkleTree::make_merkle_node(vec![length_node, pages_node]);

        Ok(root_node)
    }

    fn into_verifier_alloc<D: Deserialiser>(proof: D) -> VerifierAllocResult<D, Self> {
        type PageData = (
            PageId<{ MERKLE_LEAF_SIZE.get() }>,
            Box<[u8; MERKLE_LEAF_SIZE.get()]>,
        );

        // Obtain a deserialiser for a given start and length.
        // We parse only for PageData, since this is the raw data in the Merkle proof.
        fn parse_pages_fn_getter<D: Deserialiser>(
            start: usize,
            left_length: usize,
            proof: D,
        ) -> Result<D::Suspended<Vec<PageData>>> {
            let page = verify_backend::PageId::from_address(start);

            if left_length <= MERKLE_LEAF_SIZE.get() {
                let ctx = proof.into_leaf_raw()?;
                let ctx = ctx.map(move |data| match data {
                    Partial::Absent => vec![],
                    Partial::Blinded(_hash) => vec![],
                    Partial::Present(data) => vec![(page, data.clone())],
                });
                Ok(ctx)
            } else {
                let ctx = proof.into_node()?;

                let mut pages_acc = Vec::new();

                let mut work_brackets = work_merkle_params::<MERKLE_ARITY>(start, left_length);
                let ctx = work_brackets.try_fold(
                    ctx,
                    |ctx, (start, length)| -> Result<_, ProofError> {
                        let (ctx, pages) =
                            ctx.next_branch(|proof| parse_pages_fn_getter(start, length, proof))?;

                        pages_acc.extend(pages);

                        Ok(ctx)
                    },
                )?;

                ctx.done(pages_acc)
            }
        }

        let proof = proof.into_node()?;
        let (proof, length) = proof.next_branch(|proof| proof.into_leaf::<u64>())?;

        let (proof, pages) = proof.next_branch(|proof| {
            let length = length.to_present().map(|len| len as usize);

            let pages_handler = match length {
                // When the length node is present, we can properly parse all pages.
                Some(len) => parse_pages_fn_getter::<D>(0, len, proof)?,

                // When the length node is not present, we cannot parse any pages. This needs to be
                // validated. In other words, the node for the pages must be blinded or absent.
                None => {
                    // XXX: We can't pick whether this is a node or leaf given we don't know the
                    // length. However, absent or blinded leaves are encoded the same way as nodes.
                    // In the case where the node is present (which is an error in here), we would
                    // trigger an unexpected leaf error instead of the more appropriate error below.
                    let proof = proof.into_node()?;

                    match proof.presence() {
                        Partial::Absent | Partial::Blinded(_) => {
                            // Fine, hence we extract no pages.
                        }

                        Partial::Present(_) => {
                            // Not fine, there may be pages and we don't know how to extract them.
                            return Err(ProofError::DependentNodeIsAbsent);
                        }
                    }

                    proof.done(Vec::new())?
                }
            };

            // After the recursive parsing, convert all pages into cells.
            Ok(pages_handler.map(|pages| {
                let region = verify_backend::DynRegion::from_pages(length, pages);
                super::DynCells::bind(region)
            }))
        })?;

        proof.done(pages)
    }

    fn partial_state_hash(
        state: RefVerifierAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        let region = state.region_ref();

        let proof = match proof {
            ProofPart::Absent => {
                if !region.is_completely_absent() {
                    // The verifier contains data, but the proof is not in the right shape for us
                    // to insert back the data and obtain a root hash. This indicates an invalid
                    // proof.
                    return Err(PartialHashError::Fatal);
                }

                // The dynamic region is untouched, so re-hashing needs to resume higher up in the
                // proof tree.
                return Err(PartialHashError::PotentiallyRecoverable);
            }

            ProofPart::Present(proof) => proof,
        };

        let children = match proof {
            Tree::Leaf(MerkleProofLeaf::Blind(hash)) => {
                if !region.is_completely_absent() {
                    // The verifier contains data, but the proof is not in the right shape for us
                    // to insert back the data and obtain a root hash. This indicates an invalid
                    // proof.
                    //
                    // From a partial Merkle tree perspective, technically you can blind a node in
                    // the proof and then write all the data necessary to expand the node fully
                    // after the verification is done. In the context of a dynamic region that
                    // would entail setting the length and writing all pages. However, dynamic
                    // regions cannot be re-created hence you cannot "write" the length. Any
                    // modification requires at least the length node to be present. Hence,
                    // blinding the pages node can be paired with modifications, but you cannot
                    // blind the dynamic region's root node if there are modifications.
                    return Err(PartialHashError::Fatal);
                }

                // We already know that the dynamic region has been untouched at this point.
                return Ok(*hash);
            }

            Tree::Leaf(MerkleProofLeaf::Read(_)) => {
                return Err(PartialHashError::FromProof(ProofError::UnexpectedLeaf));
            }

            Tree::Node(children) => children.as_slice(),
        };

        let [length_tree, pages_tree] = children else {
            return Err(PartialHashError::FromProof(
                ProofError::BadNumberOfBranches {
                    expected: 2,
                    got: children.len(),
                },
            ));
        };

        let length = if let Tree::Leaf(MerkleProofLeaf::Read(data)) = length_tree {
            // The length data must be present if the node is present. Practically, if there is any
            // pages data to be dealt with we require the length. Or if there is no pages data,
            // then the only reason the parent node would be present is if the length was to be
            // read during verification.
            binary::deserialise::<u64>(data).map_err(ProofError::Deserialise)? as usize
        } else {
            return Err(PartialHashError::Fatal);
        };

        enum Event<'a> {
            Span(usize, usize, ProofTree<'a>),
            Node(Option<Hash>),
        }

        let mut queue = VecDeque::new();
        queue.push_back(Event::Span(0usize, length, ProofTree::Present(pages_tree)));

        let mut hashes: Vec<Result<Hash, PartialHashError>> = Vec::new();

        while let Some(event) = queue.pop_front() {
            match event {
                Event::Span(start, length, tree) => {
                    if length <= MERKLE_LEAF_SIZE.get() {
                        // TODO RV-463: Leaves smaller than `MERKLE_LEAF_SIZE` should also be accepted.
                        // The span's size if that of a leaf, obtain its hash if possible and push
                        // the result to the `hashes` stack.
                        match state
                            .region_ref()
                            .get_partial_page(PageId::from_address(start))
                        {
                            PartialState::Absent => hashes.push(tree.partial_hash_leaf()),
                            PartialState::Complete(data) => {
                                hashes.push(Ok(Hash::blake3_hash_bytes(data)))
                            }
                            PartialState::Incomplete => {
                                return Err(PartialHashError::Fatal);
                            }
                        };
                    } else {
                        // TODO RV-463: Nodes with fewer than `MERKLE_ARITY` children should also be accepted.
                        // The span's size is that of a node, produce `Event::Span` work items for each of its
                        // children and add them to the work queue, followed by an `Event::Node`.
                        let (branches, proof_hash) =
                            tree.into_branches_with_hash::<{ MERKLE_ARITY }>()?;

                        push_work_items_for_branches(
                            start,
                            length,
                            branches.as_ref(),
                            |branch_start, branch_length, branch| {
                                queue.push_back(Event::Span(branch_start, branch_length, branch));
                            },
                        );

                        queue.push_back(Event::Node(proof_hash));
                    }
                }
                Event::Node(proof_hash) => {
                    if hashes.is_empty() {
                        // The hashes which need to be combined have not yet been computed because
                        // their processing resulted in more `Event::Span` items. Push to the back
                        // of the work queue.
                        queue.push_back(Event::Node(proof_hash));
                        continue;
                    }
                    if hashes.len() < MERKLE_ARITY {
                        return Err(PartialHashError::Fatal);
                    };

                    // Take `MERKLE_ARITY` children hashes, compute their parent's hash, and
                    // push it to the `hashes` stack.
                    let node_hashes: Vec<_> = hashes.drain(hashes.len() - MERKLE_ARITY..).collect();
                    hashes.push(combine_partial_hashes(node_hashes, proof_hash))
                }
            }
        }

        // There must only be a single hash, the root hash. If there are more or less, that is an
        // error.
        let pages_hash = hashes
            .pop()
            .filter(|_| hashes.is_empty())
            .map_or(Err(PartialHashError::Fatal), |hash| hash)?;

        let length_hash = Hash::blake3_hash(length as u64)?;
        let root_hash = Hash::combine([length_hash, pages_hash]);

        Ok(root_hash)
    }
}

// This doctest is ignored because the macro is not part of the public API.
/// Given a [`DeserialiserNode`] and a list of types implementing [`ProofLayout`],
/// obtain a deserialiser which parses all the branches and places them in a tuple.
///
/// Usage:
/// ```ignore
/// use octez_riscv::state_backend::proof_backend::proof::deserialiser::Deserialiser;
/// use octez_riscv::state_backend::proof_backend::proof::deserialiser::DeserialiserNode;
/// use octez_riscv::state_backend::ProofLayout;
/// use octez_riscv::state_backend::VerifierAlloc;
/// use octez_riscv::state_backend::FromProofError;
///
/// fn compute_branch_case<A: ProofLayout, B: ProofLayout, D: Deserialiser>(
///     de: D,
/// ) -> VerifierAllocResult<D, (A, B)>
/// {
///     tuple_branches_proof_layout!(de, A, B)
/// }
/// ```
macro_rules! tuple_branches_proof_layout {
    ($proof:expr, $($branches:ident),+) => {{
        let ctx = $proof.into_node()?;

        paste::paste! {
            $(
                let (ctx, [<$branches:lower>]) = ctx.next_branch(|child_proof| [<$branches>]::into_verifier_alloc(child_proof))?;
            )+

            let value = (
                $(
                    [<$branches:lower>]
                ),+
            );
        }

        ctx.done(value)
    }};
}

pub(crate) use tuple_branches_proof_layout;

impl<A, B> ProofLayout for (A, B)
where
    A: ProofLayout,
    B: ProofLayout,
{
    fn to_merkle_tree(state: RefProofGenOwnedAlloc<Self>) -> Result<MerkleTree, HashError> {
        let children = vec![A::to_merkle_tree(state.0)?, B::to_merkle_tree(state.1)?];
        Ok(MerkleTree::make_merkle_node(children))
    }

    fn into_verifier_alloc<De: Deserialiser>(proof: De) -> VerifierAllocResult<De, Self> {
        tuple_branches_proof_layout!(proof, A, B)
    }

    fn partial_state_hash(
        state: RefVerifierAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        let (branches, proof_hash) = proof.into_branches_with_hash::<2>()?;

        let hashes = [
            A::partial_state_hash(state.0, branches[0]),
            B::partial_state_hash(state.1, branches[1]),
        ];

        combine_partial_hashes(hashes, proof_hash)
    }
}

impl<A, B, C> ProofLayout for (A, B, C)
where
    A: ProofLayout,
    B: ProofLayout,
    C: ProofLayout,
{
    fn to_merkle_tree(state: RefProofGenOwnedAlloc<Self>) -> Result<MerkleTree, HashError> {
        let children = vec![
            A::to_merkle_tree(state.0)?,
            B::to_merkle_tree(state.1)?,
            C::to_merkle_tree(state.2)?,
        ];
        Ok(MerkleTree::make_merkle_node(children))
    }

    fn into_verifier_alloc<De: Deserialiser>(proof: De) -> VerifierAllocResult<De, Self> {
        tuple_branches_proof_layout!(proof, A, B, C)
    }

    fn partial_state_hash(
        state: RefVerifierAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        let (branches, proof_hash) = proof.into_branches_with_hash::<3>()?;

        let hashes = [
            A::partial_state_hash(state.0, branches[0]),
            B::partial_state_hash(state.1, branches[1]),
            C::partial_state_hash(state.2, branches[2]),
        ];

        combine_partial_hashes(hashes, proof_hash)
    }
}

impl<A, B, C, D> ProofLayout for (A, B, C, D)
where
    A: ProofLayout,
    B: ProofLayout,
    C: ProofLayout,
    D: ProofLayout,
{
    fn to_merkle_tree(state: RefProofGenOwnedAlloc<Self>) -> Result<MerkleTree, HashError> {
        let children = vec![
            A::to_merkle_tree(state.0)?,
            B::to_merkle_tree(state.1)?,
            C::to_merkle_tree(state.2)?,
            D::to_merkle_tree(state.3)?,
        ];
        Ok(MerkleTree::make_merkle_node(children))
    }

    fn into_verifier_alloc<De: Deserialiser>(proof: De) -> VerifierAllocResult<De, Self> {
        tuple_branches_proof_layout!(proof, A, B, C, D)
    }

    fn partial_state_hash(
        state: RefVerifierAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        let (branches, proof_hash) = proof.into_branches_with_hash::<4>()?;

        let hashes = [
            A::partial_state_hash(state.0, branches[0]),
            B::partial_state_hash(state.1, branches[1]),
            C::partial_state_hash(state.2, branches[2]),
            D::partial_state_hash(state.3, branches[3]),
        ];

        combine_partial_hashes(hashes, proof_hash)
    }
}

impl<A, B, C, D, E> ProofLayout for (A, B, C, D, E)
where
    A: ProofLayout,
    B: ProofLayout,
    C: ProofLayout,
    D: ProofLayout,
    E: ProofLayout,
{
    fn to_merkle_tree(state: RefProofGenOwnedAlloc<Self>) -> Result<MerkleTree, HashError> {
        let children = vec![
            A::to_merkle_tree(state.0)?,
            B::to_merkle_tree(state.1)?,
            C::to_merkle_tree(state.2)?,
            D::to_merkle_tree(state.3)?,
            E::to_merkle_tree(state.4)?,
        ];
        Ok(MerkleTree::make_merkle_node(children))
    }

    fn into_verifier_alloc<De: Deserialiser>(proof: De) -> VerifierAllocResult<De, Self> {
        tuple_branches_proof_layout!(proof, A, B, C, D, E)
    }
    fn partial_state_hash(
        state: RefVerifierAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        let (branches, proof_hash) = proof.into_branches_with_hash::<5>()?;

        let hashes = [
            A::partial_state_hash(state.0, branches[0]),
            B::partial_state_hash(state.1, branches[1]),
            C::partial_state_hash(state.2, branches[2]),
            D::partial_state_hash(state.3, branches[3]),
            E::partial_state_hash(state.4, branches[4]),
        ];

        combine_partial_hashes(hashes, proof_hash)
    }
}

impl<A, B, C, D, E, F> ProofLayout for (A, B, C, D, E, F)
where
    A: ProofLayout,
    B: ProofLayout,
    C: ProofLayout,
    D: ProofLayout,
    E: ProofLayout,
    F: ProofLayout,
{
    fn to_merkle_tree(state: RefProofGenOwnedAlloc<Self>) -> Result<MerkleTree, HashError> {
        let children = vec![
            A::to_merkle_tree(state.0)?,
            B::to_merkle_tree(state.1)?,
            C::to_merkle_tree(state.2)?,
            D::to_merkle_tree(state.3)?,
            E::to_merkle_tree(state.4)?,
            F::to_merkle_tree(state.5)?,
        ];
        Ok(MerkleTree::make_merkle_node(children))
    }

    fn into_verifier_alloc<De: Deserialiser>(proof: De) -> VerifierAllocResult<De, Self> {
        tuple_branches_proof_layout!(proof, A, B, C, D, E, F)
    }

    fn partial_state_hash(
        state: RefVerifierAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        let (branches, proof_hash) = proof.into_branches_with_hash::<6>()?;

        let hashes = [
            A::partial_state_hash(state.0, branches[0]),
            B::partial_state_hash(state.1, branches[1]),
            C::partial_state_hash(state.2, branches[2]),
            D::partial_state_hash(state.3, branches[3]),
            E::partial_state_hash(state.4, branches[4]),
            F::partial_state_hash(state.5, branches[5]),
        ];

        combine_partial_hashes(hashes, proof_hash)
    }
}

impl<T, const LEN: usize> ProofLayout for [T; LEN]
where
    T: ProofLayout,
{
    fn to_merkle_tree(state: RefProofGenOwnedAlloc<Self>) -> Result<MerkleTree, HashError> {
        let children = state
            .into_iter()
            .map(T::to_merkle_tree)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(MerkleTree::make_merkle_node(children))
    }

    fn into_verifier_alloc<D: Deserialiser>(proof: D) -> VerifierAllocResult<D, Self> {
        let ctx = proof.into_node()?;

        let mut children_acc = Vec::with_capacity(LEN);

        let ctx = (0..LEN).try_fold(ctx, |ctx, _| -> Result<_, ProofError> {
            let (ctx, child) =
                ctx.next_branch(|child_proof| T::into_verifier_alloc(child_proof))?;

            children_acc.push(child);

            Ok(ctx)
        })?;

        let Ok(children) = children_acc.try_into() else {
            // We can't use expected because the error can't be displayed
            unreachable!("Conversion to array of fixed length doesn't fail")
        };

        ctx.done(children)
    }

    fn partial_state_hash(
        state: RefVerifierAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        let (branches, proof_hash) = proof.into_branches_with_hash::<LEN>()?;
        let hashes = state
            .into_iter()
            .zip(branches.iter())
            .map(|(state, proof)| T::partial_state_hash(state, *proof))
            .collect::<Vec<Result<Hash, PartialHashError>>>();
        combine_partial_hashes(hashes, proof_hash)
    }
}

impl<T, const LEN: usize> ProofLayout for Many<T, LEN>
where
    T: ProofLayout,
{
    fn to_merkle_tree(state: RefProofGenOwnedAlloc<Self>) -> Result<MerkleTree, HashError> {
        let leaves = state
            .into_iter()
            .map(T::to_merkle_tree)
            .collect::<Result<Vec<_>, _>>()?;

        build_custom_merkle_tree(MERKLE_ARITY, leaves)
    }

    fn into_verifier_alloc<D: Deserialiser>(proof: D) -> VerifierAllocResult<D, Self> {
        // Avoids clippy warnings about the type being too complex.
        type NestedSuspendedResult<T> = Vec<AllocatedOf<T, Verifier>>;

        // Obtain a deserialiser for a given length, i.e. Many<T, length>
        // We know that AllocatedOf<Many<T, LEN>, Verifier> = Vec<AllocatedOf<T, Verifier>>.

        // Ideally, this function should return Many<T, LEN> (wrapped in suspended + result)
        // but the function is recursive & dynamic in LEN
        fn parametrised_deserialiser<T: ProofLayout, D: Deserialiser>(
            length: usize,
            proof: D,
        ) -> Result<D::Suspended<NestedSuspendedResult<T>>> {
            if length == 1 {
                Ok(T::into_verifier_alloc(proof)?.map(|data| vec![data]))
            } else {
                let ctx = proof.into_node()?;

                let mut children_acc = Vec::with_capacity(MERKLE_ARITY);

                let mut child_length_iter = work_merkle_params::<MERKLE_ARITY>(0, length);
                let ctx = child_length_iter.try_fold(
                    ctx,
                    |ctx, (_, child_length)| -> Result<_, ProofError> {
                        let (ctx, children) = ctx.next_branch(|child_proof| {
                            parametrised_deserialiser::<T, D>(child_length, child_proof)
                        })?;

                        children_acc.extend(children);

                        Ok(ctx)
                    },
                )?;

                ctx.done(children_acc)
            }
        }

        // The below function's result type Result<D::Suspended<Vec<AllocatedOf<T, Verifier>>>> is precisely
        // this functions's Result<D::Suspended<VerifierAlloc<Self>>> type, so we can directly return it.
        parametrised_deserialiser::<T, D>(LEN, proof)
    }

    fn partial_state_hash(
        state: RefVerifierAlloc<Self>,
        proof: ProofTree,
    ) -> Result<Hash, PartialHashError> {
        enum Event<'a> {
            Span(usize, usize, ProofTree<'a>),
            Node(Option<Hash>),
        }

        // `T::partial_state_hash` needs to take ownership of the elements of `state`.
        // Given that `T` is not `Copy`, in order to take ownership of arbitrary elements
        // of `state` we'd first need to duplicate it and wrap each element in a type
        // which supports taking ownership.
        // However, in practice, we compute the hash of each element sequentially, meaning
        // that we can simply iterate over the state directly when calling `T::partial_state_hash`.
        let mut state = state.into_iter();
        let mut next_vec_index = 0;

        let mut queue = VecDeque::new();
        queue.push_back(Event::Span(0usize, LEN, proof));

        let mut hashes: Vec<Result<Hash, PartialHashError>> = Vec::new();

        while let Some(event) = queue.pop_front() {
            match event {
                Event::Span(start, length, tree) => {
                    if length == 1 {
                        // Check that iterating over the state is equivalent to calling `state[start]`
                        debug_assert_eq!(start, next_vec_index);
                        next_vec_index += 1;
                        hashes.push(T::partial_state_hash(
                            state.next().ok_or(PartialHashError::Fatal)?,
                            tree,
                        ))
                    } else {
                        // TODO RV-463: Nodes with fewer than `MERKLE_ARITY` children should also be accepted.
                        // The span's size is that of a node, produce `Event::Span` work items for each of its
                        // children and add them to the work queue, followed by an `Event::Node`.
                        let (branches, proof_hash) =
                            tree.into_branches_with_hash::<{ MERKLE_ARITY }>()?;

                        push_work_items_for_branches(
                            start,
                            length,
                            branches.as_ref(),
                            |branch_start, branch_length, branch| {
                                queue.push_back(Event::Span(branch_start, branch_length, branch));
                            },
                        );

                        queue.push_back(Event::Node(proof_hash));
                    }
                }
                Event::Node(proof_hash) => {
                    if hashes.is_empty() {
                        // The hashes which need to be combined have not yet been computed because
                        // their processing resulted in more `Event::Span` items. Push to the back
                        // of the work queue.
                        queue.push_back(Event::Node(proof_hash));
                        continue;
                    }
                    if hashes.len() < MERKLE_ARITY {
                        return Err(PartialHashError::Fatal);
                    };

                    // Take `MERKLE_ARITY` children hashes, compute their parent's hash, and
                    // push it to the `hashes` stack.
                    let node_hashes: Vec<_> = hashes.drain(hashes.len() - MERKLE_ARITY..).collect();
                    hashes.push(combine_partial_hashes(node_hashes, proof_hash))
                }
            }
        }

        // Check that we iterated over all the elements of the state
        debug_assert_eq!(next_vec_index, LEN);

        // There must only be a single hash, the root hash. If there are more or less, that is an
        // error.
        hashes
            .pop()
            .filter(|_| hashes.is_empty())
            .map_or(Err(PartialHashError::Fatal), |hash| hash)
    }
}

/// Attempt to compute the partial hash of a node from its children's partial
/// hashes if they are present. If none of the children hashes can be computed
/// due to absent data, this node is either a blinded leaf in the proof, in which
/// case its hash can be recovered from the proof, or it is part of a blinded
/// subtree whose hash cannot be computed as this point.
pub fn combine_partial_hashes(
    hash_results: impl AsRef<[Result<Hash, PartialHashError>]>,
    proof_hash: Option<Hash>,
) -> Result<Hash, PartialHashError> {
    let hash_results = hash_results.as_ref();
    if hash_results.is_empty() {
        return Ok(Hash::combine::<Hash, _>([]));
    }

    // If the first result is a hash, all results need to be a hash in order to
    // compute the combined hash. If the first result is a potentially
    // recoverable error, all results need to to be potentially recoverable
    // errors in order to fall back on the proof hash. Anything else is a fatal error.
    let expect_ok = match hash_results[0] {
        Ok(_) => true,
        Err(PartialHashError::PotentiallyRecoverable) => false,
        _ => return Err(PartialHashError::Fatal),
    };

    let mut hashes = Vec::with_capacity(hash_results.len());
    let hash_results_len = hash_results.len();
    for r in hash_results {
        match r {
            Ok(hash) if expect_ok => hashes.push(*hash),
            Err(PartialHashError::PotentiallyRecoverable) if !expect_ok => (),
            _ => return Err(PartialHashError::Fatal),
        }
    }

    if expect_ok {
        debug_assert_eq!(hashes.len(), hash_results_len);
        return Ok(Hash::combine(hashes));
    };

    proof_hash.ok_or(PartialHashError::PotentiallyRecoverable)
}

fn work_merkle_params<const CHILDREN: usize>(
    mut branch_start: usize,
    mut length_left: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let branch_max_length = length_left.div_ceil(MERKLE_ARITY);

    (0..CHILDREN).map(move |_| {
        let this_branch_length = branch_max_length.min(length_left);

        let item = (branch_start, this_branch_length);

        branch_start = branch_start.saturating_add(this_branch_length);
        length_left = length_left.saturating_sub(this_branch_length);

        item
    })
}

fn push_work_items_for_branches<'a, const CHILDREN: usize>(
    branch_start: usize,
    length_left: usize,
    branches: &'_ [ProofTree<'a>; CHILDREN],
    mut push: impl FnMut(usize, usize, ProofTree<'a>),
) {
    let children = work_merkle_params::<CHILDREN>(branch_start, length_left);
    for (branch, (child_start, child_length)) in branches.iter().zip(children) {
        if child_length > 0 {
            push(child_start, child_length, *branch);
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prop_assert;
    use proptest::prop_assert_eq;
    use proptest::proptest;
    use tests::verify_backend::handle_stepper_panics;

    use super::*;
    use crate::state_backend::Cells;
    use crate::state_backend::CommitmentLayout;
    use crate::state_backend::DynCells;
    use crate::state_backend::FnManagerIdent;
    use crate::state_backend::ManagerWrite;
    use crate::state_backend::proof_backend::ProofGen;
    use crate::state_backend::proof_backend::ProofRegion;
    use crate::state_backend::proof_backend::ProofWrapper;
    use crate::state_backend::proof_backend::proof::deserialise_owned;

    const CELLS_SIZE: usize = 32;

    // When producing a proof from a `ProofGen` state, values written during
    // the execution of the tick being proven should not be blinded, whereas
    // values which were not accessed should be blinded. When a proof contains
    // blinded values, it should be possible to compute the final hash of the
    // `Verifier` state constructed from this proof.
    #[test]
    fn test_proof_blinding() {
        type TestLayout = (Array<u64, CELLS_SIZE>, Array<u64, CELLS_SIZE>);

        proptest!(|(value_before: u64, value_after: u64, i in 0..CELLS_SIZE)| {
            // Bind `ProofGen` cells and write at one address
            let cells1 = [value_before; CELLS_SIZE];
            let mut proof_region1: ProofRegion<u64, CELLS_SIZE, Ref<'_, Owned>> =
                ProofRegion::bind(&cells1);
            ProofGen::<Ref<'_, Owned>>::region_write(&mut proof_region1, i, value_after);
            let proof_cells1: Cells<u64, CELLS_SIZE, Ref<'_, ProofGen<Ref<'_, Owned>>>> =
                Cells::bind(&proof_region1);

            // Bind `ProofGen` cells and do not access them
            let cells2 = [value_before; CELLS_SIZE];
            let proof_region2: ProofRegion<u64, CELLS_SIZE, Ref<'_, Owned>> =
                ProofRegion::bind(&cells2);
            let proof_cells2: Cells<u64, CELLS_SIZE, Ref<'_, ProofGen<Ref<'_, Owned>>>> =
                Cells::bind(&proof_region2);

            let proof_state = (proof_cells1, proof_cells2);

            let merkle_proof = <TestLayout as ProofLayout>::to_merkle_tree(proof_state)
                .unwrap()
                .to_merkle_proof();

            let verifier_state = deserialise_owned::deserialise::<TestLayout>(
                ProofTree::Present(&merkle_proof)
            ).unwrap();

            // The first component of the state was present in the proof, can be
            // fully read, and contains the initial state.
            prop_assert_eq!(verifier_state.0.0.read_all(), vec![value_before; CELLS_SIZE]);

            // The second component of the state is fully blinded: no values can
            // be read from the array.
            for i in 0..CELLS_SIZE {
                prop_assert!(handle_stepper_panics(|| verifier_state.0.1.read(i)).is_err());
            };

            let ref_verifier_state = (
                verifier_state.0.0.struct_ref::<FnManagerIdent>(),
                verifier_state.0.1.struct_ref::<FnManagerIdent>(),
            );
            prop_assert!(
                <TestLayout as ProofLayout>::partial_state_hash(ref_verifier_state, ProofTree::Present(&merkle_proof)).is_ok()
            );
        })
    }

    /// Test the proof generation and verification for a computation against a dynamic region.
    ///
    /// # Safety
    ///
    /// The `test_proof` and `test_verify` function must be the same function instantiated to
    /// different managers.
    ///
    /// Due to Rust's limitation on higher-ranked polymorphism, we can't accept
    /// a single function and instantiate it within the function body with the respective managers
    /// `ProofGen<_>` and `Verifier`. One could work around this restriction by using a trait to
    /// simulate the rank-2-ness, but that means you can't provide closures as the implementation
    /// any more. If any of the given `test_proof` or `test_verify` capture an environment, this
    /// would no longer work.
    unsafe fn test_dyn_array_with_funs(
        len: usize,
        test_proof: impl FnOnce(&mut DynCells<ProofGen<Ref<'_, Owned>>>),
        test_verify: impl FnOnce(&mut DynCells<Verifier>),
    ) {
        let owned_cell = DynCells::new(len);

        // We require the initial hash to ensure that the generated proof, but also the
        // instantiated state from the proof match the "before" state.
        let init_hash = {
            let state_ref = owned_cell.struct_ref::<FnManagerIdent>();
            DynArray::state_hash(state_ref).unwrap()
        };

        // The `ProofWrapper` transformer ensures the resulting dynamic region (via `DynCells`) is
        // setup for proof generation. You can think of this as starting the recording for a proof.
        let mut proof_cell = owned_cell.struct_ref::<ProofWrapper>();

        test_proof(&mut proof_cell);

        // The post-hash is required to ensure that the verifier's final state matches the prover's
        // final state.
        let post_hash = DynArray::state_hash(proof_cell.struct_ref::<FnManagerIdent>()).unwrap();

        let tree = DynArray::to_merkle_tree(proof_cell.struct_ref::<FnManagerIdent>()).unwrap();
        let proof_tree = tree.to_merkle_proof();
        assert_eq!(proof_tree.root_hash(), init_hash);

        // Instantiating the verifier state allows us to replay the computation and verify it does
        // the right things.
        let (mut verify_cell, out_proof) =
            deserialise_owned::deserialise::<DynArray>(ProofTree::Present(&proof_tree)).unwrap();

        let OwnedProofPart::Present(out_proof) = out_proof else {
            panic!("Expected present proof");
        };
        assert_eq!(proof_tree, out_proof);

        let out_proof_tree = ProofTree::Present(&out_proof);

        // The initial verifier state must match that of the initial state against which we
        // produced the proof.
        let verifier_init_hash = {
            let state_ref = verify_cell.struct_ref::<FnManagerIdent>();
            DynArray::partial_state_hash(state_ref, out_proof_tree).unwrap()
        };
        assert_eq!(verifier_init_hash, init_hash);

        test_verify(&mut verify_cell);

        // Once we're doing replaying the computation on the verifier side, the final state must
        // match that of the prover's. If not, that means we produced a proof that results in a
        // transition that we did not intend to prove.
        let verifier_post_hash = {
            let state_ref = verify_cell.struct_ref::<FnManagerIdent>();
            DynArray::partial_state_hash(state_ref, out_proof_tree).unwrap()
        };
        assert_eq!(verifier_post_hash, post_hash);
    }

    /// Generate a test for dynamic regions using a given size and closure which operates on the
    /// [`DynCells`]. This effectively demonstrates that the actions performed by the given closure
    /// can be proven and verified correctly.
    macro_rules! test_dyn_array_with {
        ($len:literal, | $param:ident | { $($body:tt)* }) => {
            {
                let test_proof = |$param: &mut DynCells<ProofGen<Ref<'_, Owned>>>| {
                    $($body)*
                };

                let test_verify = |$param: &mut DynCells<Verifier>| {
                    $($body)*
                };

                // SAFETY: This function is intended to be used only in this macro.
                unsafe {
                    test_dyn_array_with_funs($len, test_proof, test_verify);
                }
            }
        };
    }

    #[test]
    fn test_dyn_array_proofs_nothing() {
        test_dyn_array_with!(65536, |_cell| {});
    }

    #[test]
    fn test_dyn_array_proofs_read() {
        proptest!(|(addr in 0..65528usize)| {
            test_dyn_array_with!(65536, |cell| {
                unsafe {
                    cell.read::<u64>(addr);
                }
            });
        });
    }

    #[test]
    fn test_dyn_array_proofs_write() {
        proptest!(|(addr in 0..65528usize, val: u64)| {
            test_dyn_array_with!(65536, |cell| {
                unsafe {
                    cell.write::<u64>(addr, val);
                }
            });
        });
    }

    #[test]
    fn test_dyn_array_proofs_len() {
        test_dyn_array_with!(65536, |cell| {
            cell.len();
        });
    }

    #[test]
    fn test_dyn_array_proofs_read_and_len() {
        proptest!(|(addr in 0..65528usize)| {
            test_dyn_array_with!(65536, |cell| {
                unsafe {
                    cell.read::<u64>(addr);
                }

                cell.len();
            });
        });
    }

    #[test]
    fn test_dyn_array_proofs_write_and_len() {
        proptest!(|(addr in 0..65528usize, val: u64)| {
            test_dyn_array_with!(65536, |cell| {
                unsafe {
                    cell.write::<u64>(addr, val);
                }

                cell.len();
            });
        });
    }

    #[test]
    fn test_dyn_array_proofs_read_and_write() {
        proptest!(|(addr in 0..65528usize, val: u64)| {
            test_dyn_array_with!(65536, |cell| {
                unsafe {
                    let x = cell.read::<u64>(addr);
                    cell.write(addr, x.wrapping_add(val));
                }
            });
        });
    }

    #[test]
    fn test_dyn_array_proofs_read_and_write_and_len() {
        proptest!(|(addr in 0..65528usize, val: u64)| {
            test_dyn_array_with!(65536, |cell| {
                unsafe {
                    let x = cell.read::<u64>(addr);
                    cell.write(addr, x.wrapping_add(val));
                }

                cell.len();
            });
        });
    }
}
