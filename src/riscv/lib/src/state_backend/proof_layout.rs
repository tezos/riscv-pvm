// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::error;

use bincode::error::DecodeError;
use bincode::error::EncodeError;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::merkle_proof::DeserialiserError;
use octez_riscv_data::merkle_proof::Partial;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProofLeaf;
use octez_riscv_data::merkle_proof::tag::InvalidTagError;
use octez_riscv_data::mode::Verify;
use octez_riscv_data::tree::Tree;
use perfect_derive::perfect_derive;

use super::Layout;
use super::proof_backend::merkle::MERKLE_ARITY;
use super::proof_backend::proof::deserialiser::Result;
use crate::array_utils::boxed_array;
use crate::state_backend::proof_backend::proof::NotEnoughBytesError;

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

/// Regions for the verifier backend for a specific layout.
pub type VerifyAlloc<L> = <L as Layout>::Allocated<Verify>;

/// Errors that may occur when hashing a state in [`Verify`] mode
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

    /// For the purpose of computing the final hash of a state in `Verify` mode,
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
#[derive(Debug, Clone)]
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

    /// Obtain the [`ProofTree`] reference corresponding to this owned proof part.
    pub fn as_ref(&self) -> Option<&MerkleProof> {
        match self {
            Self::Present(proof) => Some(proof),
            Self::Absent => None,
        }
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

#[cfg(test)]
mod tests {
    use octez_riscv_data::hash::PartialHash;
    use octez_riscv_data::merkle_tree::MerkleTree;
    use octez_riscv_data::mode::Prove;
    use proptest::prop_assert;
    use proptest::prop_assert_eq;
    use proptest::proptest;

    use super::*;
    use crate::state_backend::AllocatedOf;
    use crate::state_backend::Array;
    use crate::state_backend::Cells;
    use crate::state_backend::DynArray;
    use crate::state_backend::DynCells;
    use crate::state_backend::ManagerWrite;
    use crate::state_backend::proof_backend::ProofRegion;
    use crate::state_backend::proof_backend::ProofWrapper;
    use crate::state_backend::proof_backend::merkle::merkle_tree_to_merkle_proof;
    use crate::state_backend::proof_backend::proof::deserialise_owned;
    use crate::state_backend::verify_backend::handle_stepper_panics;

    const CELLS_SIZE: usize = 32;

    // When producing a proof from a state in `Prove` mode, values written during
    // the execution of the tick being proven should not be blinded, whereas
    // values which were not accessed should be blinded. When a proof contains
    // blinded values, it should be possible to compute the final hash of the
    // state in `Verify` mode constructed from this proof.
    #[test]
    fn test_proof_blinding() {
        type TestLayout = (Array<u64, CELLS_SIZE>, Array<u64, CELLS_SIZE>);

        proptest!(|(value_before: u64, value_after: u64, i in 0..CELLS_SIZE)| {
            // Bind `Prove` cells and write at one address
            let cells1 = [value_before; CELLS_SIZE];
            let mut proof_region1: ProofRegion<u64, CELLS_SIZE> = ProofRegion::bind(&cells1);
            Prove::region_write(&mut proof_region1, i, value_after);
            let proof_cells1: Cells<u64, CELLS_SIZE, Prove> = Cells::bind(proof_region1);

            // Bind `Prove` cells and do not access them
            let cells2 = [value_before; CELLS_SIZE];
            let proof_region2: ProofRegion<u64, CELLS_SIZE> = ProofRegion::bind(&cells2);
            let proof_cells2: Cells<u64, CELLS_SIZE, Prove> = Cells::bind(proof_region2);

            let proof_state = (proof_cells1, proof_cells2);

            let merkle_proof = merkle_tree_to_merkle_proof(MerkleTree::from_foldable(&proof_state));

            let verifier_state =
                deserialise_owned::deserialise::<AllocatedOf<TestLayout, Verify>, _>(
                    ProofTree::Present(&merkle_proof),
                    (),
                ).unwrap();

            // The first component of the state was present in the proof, can be
            // fully read, and contains the initial state.
            prop_assert_eq!(verifier_state.0.0.read_all(), vec![value_before; CELLS_SIZE]);

            // The second component of the state is fully blinded: no values can
            // be read from the array.
            for i in 0..CELLS_SIZE {
                prop_assert!(handle_stepper_panics(|| verifier_state.0.1.read(i)).is_err());
            };

            let partial_hash = PartialHash::from_foldable(
                Some(&merkle_proof),
                &verifier_state.0,
            );
            prop_assert!(partial_hash.to_hash().is_some());
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
    /// a single function and instantiate it within the function body with the respective modes
    /// `Prove<_>` and `Verify`. One could work around this restriction by using a trait to
    /// simulate the rank-2-ness, but that means you can't provide closures as the implementation
    /// any more. If any of the given `test_proof` or `test_verify` capture an environment, this
    /// would no longer work.
    unsafe fn test_dyn_array_with_funs(
        len: usize,
        test_proof: impl FnOnce(&mut DynCells<Prove>),
        test_verify: impl FnOnce(&mut DynCells<Verify>),
    ) {
        let owned_cell = DynCells::new(len);

        // We require the initial hash to ensure that the generated proof, but also the
        // instantiated state from the proof match the "before" state.
        let init_hash = Hash::from_foldable(&owned_cell);

        // The `ProofWrapper` transformer ensures the resulting dynamic region (via `DynCells`) is
        // setup for proof generation. You can think of this as starting the recording for a proof.
        let mut proof_cell = owned_cell.struct_ref::<ProofWrapper>();

        test_proof(&mut proof_cell);

        // The post-hash is required to ensure that the verifier's final state matches the prover's
        // final state.
        let post_hash = Hash::from_foldable(&proof_cell);

        let tree = MerkleTree::from_foldable(&proof_cell);
        let proof_tree = merkle_tree_to_merkle_proof(tree);
        assert_eq!(proof_tree.root_hash(), init_hash);

        // Instantiating the verifier state allows us to replay the computation and verify it does
        // the right things.
        let (mut verify_cell, out_proof) = deserialise_owned::deserialise::<
            AllocatedOf<DynArray, Verify>,
            _,
        >(ProofTree::Present(&proof_tree), ())
        .unwrap();

        let OwnedProofPart::Present(out_tree) = &out_proof else {
            panic!("Expected present proof");
        };
        assert_eq!(&proof_tree, out_tree);

        // The initial verifier state must match that of the initial state against which we
        // produced the proof.
        let verifier_init_hash = PartialHash::from_foldable(out_proof.as_ref(), &verify_cell)
            .to_hash()
            .unwrap();
        assert_eq!(verifier_init_hash, init_hash);

        test_verify(&mut verify_cell);

        // Once we're doing replaying the computation on the verifier side, the final state must
        // match that of the prover's. If not, that means we produced a proof that results in a
        // transition that we did not intend to prove.
        let verifier_post_hash = PartialHash::from_foldable(out_proof.as_ref(), &verify_cell)
            .to_hash()
            .unwrap();
        assert_eq!(verifier_post_hash, post_hash);
    }

    /// Generate a test for dynamic regions using a given size and closure which operates on the
    /// [`DynCells`]. This effectively demonstrates that the actions performed by the given closure
    /// can be proven and verified correctly.
    macro_rules! test_dyn_array_with {
        ($len:literal, | $param:ident | { $($body:tt)* }) => {
            {
                let test_proof = |$param: &mut DynCells<Prove>| {
                    $($body)*
                };

                let test_verify = |$param: &mut DynCells<Verify>| {
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
