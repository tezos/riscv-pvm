// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use std::error;

use bincode::error::DecodeError;
use octez_riscv_data::merkle_proof::DeserialiserError;
use octez_riscv_data::merkle_proof::Partial;
use octez_riscv_data::merkle_proof::proof_tree::MerkleProof;
use octez_riscv_data::merkle_proof::tag::InvalidTagError;
use perfect_derive::perfect_derive;

use super::proof_backend::merkle::MERKLE_ARITY;
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

#[cfg(test)]
mod tests {
    use octez_riscv_data::components::atom::Atom;
    use octez_riscv_data::hash::Hash;
    use octez_riscv_data::hash::PartialHash;
    use octez_riscv_data::merkle_tree::MerkleTree;
    use octez_riscv_data::mode::Prove;
    use octez_riscv_data::mode::Verify;
    use octez_riscv_data::mode::utils::catch_not_found;
    use proptest::prop_assert;
    use proptest::prop_assert_eq;
    use proptest::proptest;

    use super::*;
    use crate::state_backend::DynCells;
    use crate::state_backend::proof_backend::merkle::merkle_tree_to_merkle_proof;
    use crate::state_backend::proof_backend::proof::deserialise_owned;

    const ATOMS_SIZE: usize = 32;

    // When producing a proof from a state in `Prove` mode, values written during
    // the execution of the tick being proven should not be blinded, whereas
    // values which were not accessed should be blinded. When a proof contains
    // blinded values, it should be possible to compute the final hash of the
    // state in `Verify` mode constructed from this proof.
    #[test]
    fn test_proof_blinding() {
        type TestState<M> = (Atom<[u64; ATOMS_SIZE], M>, Atom<[u64; ATOMS_SIZE], M>);

        proptest!(|(value_before: u64, value_after: u64, i in 0..ATOMS_SIZE)| {
            let data_before = [value_before; ATOMS_SIZE];

            // Bind `Prove` atoms and write to one index
            let mut proof_atoms1: Atom<[u64; ATOMS_SIZE], Prove> = Atom::new(data_before);
            proof_atoms1[i] = value_after;


            // Bind `Prove` atoms and do not access them
            let proof_atoms2: Atom<[u64; ATOMS_SIZE], Prove> = Atom::new(data_before);

            let proof_state = (proof_atoms1, proof_atoms2);

            let merkle_proof = merkle_tree_to_merkle_proof(MerkleTree::from_foldable(&proof_state));

            let verifier_state =
                deserialise_owned::deserialise::<TestState<Verify>>(
                    ProofTree::Present(&merkle_proof),
                ).unwrap();

            // The first component of the state was present in the proof, can be
            // fully read, and contains the initial state.
            prop_assert_eq!(verifier_state.0.0.read(), [value_before; ATOMS_SIZE]);

            // The second component of the state is fully blinded: no values can
            // be read from the array.
            for i in 0..ATOMS_SIZE {
                prop_assert!(catch_not_found(|| verifier_state.0.1[i]).is_err());
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
        let mut proof_cell = owned_cell.start_proof();

        test_proof(&mut proof_cell);

        // The post-hash is required to ensure that the verifier's final state matches the prover's
        // final state.
        let post_hash = Hash::from_foldable(&proof_cell);

        let tree = MerkleTree::from_foldable(&proof_cell);
        let proof_tree = merkle_tree_to_merkle_proof(tree);
        assert_eq!(proof_tree.root_hash(), init_hash);

        // Instantiating the verifier state allows us to replay the computation and verify it does
        // the right things.
        let (mut verify_cell, out_proof) =
            deserialise_owned::deserialise::<DynCells<Verify>>(ProofTree::Present(&proof_tree))
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
