// SPDX-FileCopyrightText: 2023,2025 TriliTech <contact@trili.tech>
// SPDX-FileCopyrightText: 2024-2025 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

use bincode::Decode;
use bincode::Encode;
use bincode::de::Decoder;
use bincode::de::read::Reader;
use bincode::enc::Encoder;
use bincode::error::DecodeError;
use bincode::error::EncodeError;
use octez_riscv_data::clone::CloneState;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::NodeFold;
use octez_riscv_data::foldable::seq_tree::IndexableSeqAsTree;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashFold;
use octez_riscv_data::hash::PartialHash;
use octez_riscv_data::hash::PartialHashFold;
use octez_riscv_data::merkle_proof;
use octez_riscv_data::merkle_proof::DeserialiserNode;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::merkle_proof::Partial;
use octez_riscv_data::merkle_proof::Suspended;
use octez_riscv_data::merkle_tree::MerkleTree;
use octez_riscv_data::merkle_tree::MerkleTreeFold;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Prove;
use octez_riscv_data::mode::Verify;
use octez_riscv_data::serialisation::serialise;

use super::ManagerAlloc;
use super::ManagerBase;
use super::ManagerClone;
use super::ManagerRead;
use super::ManagerSerialise;
use super::ManagerWrite;
use crate::state_backend::Elem;
use crate::state_backend::ProofError;
use crate::state_backend::proof_backend;
use crate::state_backend::proof_backend::merkle::MERKLE_ARITY;
use crate::state_backend::proof_backend::merkle::MERKLE_LEAF_SIZE;
use crate::state_backend::verify_backend;

/// Multiple elements of an unspecified type
pub struct DynCells<M: ManagerBase> {
    region: M::DynRegion,
}

impl<M: ManagerBase> DynCells<M> {
    /// Allocate a new dynamic region with the given length in bytes.
    pub fn new(len: usize) -> Self
    where
        M: ManagerAlloc,
    {
        let region = M::allocate_dyn_region(len);
        Self { region }
    }

    /// Bind this state to the given dynamic region.
    pub fn bind(region: M::DynRegion) -> Self {
        Self { region }
    }

    /// Obtain a reference to the underlying dynamic region.
    pub fn region_ref(&self) -> &M::DynRegion {
        &self.region
    }

    /// Obtain the underlying dynamic region.
    pub fn into_region(self) -> M::DynRegion {
        self.region
    }

    /// Is the dynamic region empty?
    pub fn is_empty(&self) -> bool
    where
        M: ManagerRead,
    {
        self.len() == 0
    }

    /// Retrieve the number of bytes in the dynamic region.
    #[inline]
    pub fn len(&self) -> usize
    where
        M: ManagerRead,
    {
        M::dyn_region_len(&self.region)
    }

    /// Read an element in the region. `address` is in bytes.
    ///
    /// # Safety
    ///
    /// See [`ManagerRead::dyn_region_read`] for safety requirements.
    #[inline]
    pub unsafe fn read<E: Elem>(&self, address: usize) -> E
    where
        M: ManagerRead,
    {
        unsafe { M::dyn_region_read(&self.region, address) }
    }

    /// Read elements from the region. `address` is in bytes.
    #[inline]
    pub fn read_all<E: Elem>(&self, address: usize, values: &mut [E])
    where
        M: ManagerRead,
    {
        M::dyn_region_read_all(&self.region, address, values)
    }

    /// Update an element in the region. `address` is in bytes.
    ///
    /// # Safety
    ///
    /// See [`ManagerWrite::dyn_region_write`] for safety requirements.
    #[inline]
    pub unsafe fn write<E: Elem>(&mut self, address: usize, value: E)
    where
        M: ManagerWrite,
    {
        unsafe { M::dyn_region_write(&mut self.region, address, value) }
    }

    /// Update multiple elements in the region. `address` is in bytes.
    #[inline]
    pub fn write_all<E: Elem + Copy>(&mut self, address: usize, values: &[E])
    where
        M: ManagerWrite + ManagerRead,
    {
        M::dyn_region_write_all(&mut self.region, address, values)
    }
}

impl DynCells<Normal> {
    /// Return a proof-generating version of these DynCells.
    pub fn start_proof(&self) -> DynCells<Prove<'_>> {
        DynCells {
            region: proof_backend::ProofDynRegion::bind(&self.region),
        }
    }
}

impl<M: ManagerSerialise> Encode for DynCells<M> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        M::serialise_dyn_region(&self.region, encoder)
    }
}

impl<C> Decode<C> for DynCells<Normal> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let len = u64::decode(decoder)? as usize;

        let mut region = Normal::allocate_dyn_region(len);
        decoder.reader().read(&mut region)?;

        Ok(DynCells { region })
    }
}

impl<M: ManagerRead, N: ManagerRead> PartialEq<DynCells<N>> for DynCells<M> {
    fn eq(&self, other: &DynCells<N>) -> bool {
        let len = self.len();

        if len != other.len() {
            return false;
        }

        for i in 0..len {
            // SAFETY: We know that `i < len` from the loop condition. Therefore, the reads are
            // always within the maximum bounds.
            unsafe {
                if self.read::<u8>(i) != other.read::<u8>(i) {
                    return false;
                }
            }
        }

        true
    }
}

impl<M: ManagerRead> Eq for DynCells<M> {}

impl<M: ManagerClone> Clone for DynCells<M> {
    fn clone(&self) -> Self {
        Self {
            region: M::clone_dyn_region(&self.region),
        }
    }
}

impl<M: ManagerClone> CloneState for DynCells<M> {
    fn clone_state(&self) -> Self {
        self.clone()
    }
}

impl<M: ManagerSerialise> Foldable<HashFold> for DynCells<M> {
    fn fold(&self, builder: HashFold) -> Hash {
        let length = self.len();
        let length_node = Hash::blake3_hash(length as u64).expect("Hashing length should not fail");

        let generator = |idx: usize| {
            let address = MERKLE_LEAF_SIZE
                .get()
                .checked_mul(idx)
                .expect("This should not overflow as we split the length into chunks of MERKLE_LEAF_SIZE bytes before");

            // SAFETY: The chunk writer will only request data within the bounds that we specified.
            // Given we provided the correct length, this is safe.
            let data = unsafe { self.read::<[u8; MERKLE_LEAF_SIZE.get()]>(address) };
            Hash::blake3_hash_bytes(&data)
        };

        let pages = length.div_ceil(MERKLE_LEAF_SIZE.get());

        let mut builder = builder.into_node_fold();
        builder.add(&length_node);
        builder.add(&IndexableSeqAsTree::new(pages, MERKLE_ARITY, &generator));
        builder.done()
    }
}

impl FromProof for DynCells<Verify> {
    fn from_proof<D: merkle_proof::Deserialiser>(
        proof: D,
    ) -> merkle_proof::SuspendedResult<D, Self> {
        let proof = proof.into_node()?;

        let (proof, length) = proof.next_branch_with(|proof| proof.into_leaf::<u64>())?;
        let length = length.to_present().map(|len| len as usize);

        let (proof, pages) = proof.next_branch_with(|proof| {
            // When the length node is present, we can properly parse all pages.
            // But when the length node is not present, we cannot parse any pages. This needs to be
            // validated. In other words, the node for the pages must be blinded or absent.
            let Some(len) = length else {
                // XXX: We can't pick whether this is a node or leaf given we don't know the
                // length. However, absent or blinded leaves are encoded the same way as nodes.
                // In the case where the node is present (which is an error in here), we would
                // trigger an unexpected leaf error instead of the more appropriate error below.
                let proof = proof.into_node()?;

                // When the node for the pages is present, that's a problem. There may be pages and
                // we don't know how to extract them because we don't know how many there are.
                if let Partial::Present(_) = proof.presence() {
                    return Err(merkle_proof::DeserialiserError::custom(
                        ProofError::AbsentProof,
                    ));
                }

                return proof.done(Vec::new());
            };

            let mut pages = Vec::new();
            let mut for_leaf = |idx, proof: D| {
                // The index is the page number, but the page ID is the starting address.
                let Some(address) = MERKLE_LEAF_SIZE.get().checked_mul(idx) else {
                    return Err(merkle_proof::DeserialiserError::custom(
                        verify_backend::PageIdOverflow,
                    ));
                };
                let page_id = verify_backend::PageId::from_address(address);

                let result = proof.into_leaf_raw()?;
                let result = result.map(|data| {
                    if let Partial::Present(data) = data {
                        pages.push((page_id, data));
                    }
                });

                Ok(result)
            };

            let num_leaves = len.div_ceil(MERKLE_LEAF_SIZE.get());
            let result =
                merkle_proof::descend_tree(proof, MERKLE_ARITY, 0, num_leaves, &mut for_leaf)?;

            Ok(result.map(|()| pages))
        })?;

        // After the parsing, convert all pages into cells.
        let region = verify_backend::DynRegion::from_pages(length, pages);
        let pages = super::DynCells::bind(region);

        proof.done(pages)
    }
}

impl Foldable<MerkleTreeFold> for DynCells<Prove<'_>> {
    fn fold(&self, builder: MerkleTreeFold) -> MerkleTree {
        let length = self.region.unrecorded_len();
        let length_data = serialise(length as u64).expect("Serialising length should not fail");
        let length_needed = self.region.need_length_in_proof();
        let length_node = MerkleTree::make_merkle_leaf(length_data, length_needed);

        let region = self.region_ref();
        let reads = region.get_read();
        let writes = region.get_write();

        let page_tree_generator = |idx| {
            let address = MERKLE_LEAF_SIZE
                .get()
                .checked_mul(idx)
                .expect("This should not overflow as we split the length into chunks of MERKLE_LEAF_SIZE bytes before");
            let range = address..(address + MERKLE_LEAF_SIZE.get());
            let accessed = reads.includes_range(range.clone()) || writes.includes_range(range);
            let data: [u8; MERKLE_LEAF_SIZE.get()] =
                unsafe { region.inner_dyn_region_read(address) };
            MerkleTree::make_merkle_leaf(data.to_vec(), accessed)
        };

        let pages = length.div_ceil(MERKLE_LEAF_SIZE.get());

        let mut builder = builder.into_node_fold();
        builder.add(&length_node);
        builder.add(&IndexableSeqAsTree::new(
            pages,
            MERKLE_ARITY,
            &page_tree_generator,
        ));
        builder.done()
    }
}

impl Foldable<PartialHashFold<'_>> for DynCells<Verify> {
    fn fold(&self, builder: PartialHashFold) -> PartialHash {
        if self.region.is_completely_absent() {
            return PartialHash::Previous;
        }

        // The length must be present if the region is not completely absent. Otherwise we can't
        // properly construct the partial Merkle tree and therefore obtain the final hash.
        let Some(len) = self.region.len_opt() else {
            return PartialHash::InvalidProof;
        };
        let length_hash = Hash::blake3_hash(len as u64).expect("Hashing length should not fail");

        let page_hash_generator = |idx| {
            let address = MERKLE_LEAF_SIZE
                .get()
                .checked_mul(idx)
                .expect("This should not overflow as we split the length into chunks of MERKLE_LEAF_SIZE bytes before");
            let page_id = verify_backend::PageId::from_address(address);

            let page = self.region.get_partial_page(page_id);
            match page {
                verify_backend::PartialState::Incomplete => PartialHash::InvalidProof,
                verify_backend::PartialState::Absent => PartialHash::Previous,
                verify_backend::PartialState::Complete(data) => {
                    let hash = Hash::blake3_hash(data).expect("Hashing page should not fail");
                    PartialHash::Present(hash)
                }
            }
        };

        let mut builder = builder.into_node_fold();
        builder.add(&PartialHash::Present(length_hash));
        builder.add(&IndexableSeqAsTree::new(
            len.div_ceil(MERKLE_LEAF_SIZE.get()),
            MERKLE_ARITY,
            &page_hash_generator,
        ));
        builder.done()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::num::NonZeroUsize;

    use bincode::Encode;
    use bincode::enc::Encoder;
    use bincode::error::EncodeError;
    use octez_riscv_data::components::atom::Atom;
    use octez_riscv_data::foldable::Fold;
    use octez_riscv_data::foldable::Foldable;
    use octez_riscv_data::foldable::NodeFold;
    use octez_riscv_data::hash::Hash;
    use octez_riscv_data::hash::PartialHash;
    use octez_riscv_data::merkle_tree::MerkleTree;
    use octez_riscv_data::mode::Normal;
    use octez_riscv_data::mode::utils::catch_not_found;

    use crate::backend_test;
    use crate::default::ConstDefault;
    use crate::state_backend::DynCells;
    use crate::state_backend::Elem;
    use crate::state_backend::ManagerBase;
    use crate::state_backend::ProofPart;
    use crate::state_backend::proof_backend::merkle::merkle_tree_to_merkle_proof;
    use crate::state_backend::proof_backend::proof::deserialise_owned;

    /// Dummy type that helps us implement custom normalisation via [`Elem`]
    #[repr(C, packed)]
    #[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Ord, Eq, Default)]
    struct Flipper {
        a: u8,
        b: u8,
    }

    impl ConstDefault for Flipper {
        const DEFAULT: Self = Self { a: 0, b: 0 };
    }

    impl Encode for Flipper {
        fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
            self.b.encode(encoder)?;
            self.a.encode(encoder)?;
            Ok(())
        }
    }

    impl Elem for Flipper {
        const STORED_SIZE: NonZeroUsize = NonZeroUsize::new(2).unwrap();

        unsafe fn read_unaligned(source: *const u8) -> Self {
            unsafe {
                Self {
                    a: source.add(1).read(),
                    b: source.read(),
                }
            }
        }

        unsafe fn write_unaligned(self, dest: *mut u8) {
            unsafe {
                dest.add(1).write(self.a);
                dest.write(self.b);
            }
        }
    }

    backend_test!(
        #[should_panic]
        test_dynregion_oob_2,
        F,
        {
            const LEN: usize = 4096;

            let mut state = DynCells::<F>::new(LEN);

            // This should panic because we are trying to write an element at the address which
            // corresponds to the end of the buffer.
            unsafe {
                state.write(LEN * Flipper::STORED_SIZE.get(), Flipper { a: 1, b: 2 });
            }
        }
    );

    backend_test!(test_dynregion_stored_format, F, {
        // Writing to one item of the region must convert to stored format.
        let mut region = DynCells::<F>::new(4096);

        unsafe {
            region.write(0, Flipper { a: 13, b: 37 });
            assert_eq!(region.read::<Flipper>(0), Flipper { a: 13, b: 37 });
        }

        let buffer = unsafe { region.read::<[u8; 2]>(0) };
        assert_eq!(buffer, [37, 13]);

        // Writing to the entire region must convert properly to stored format.
        region.write_all::<Flipper>(0, &[
            Flipper { a: 11, b: 22 },
            Flipper { a: 13, b: 24 },
            Flipper { a: 15, b: 26 },
            Flipper { a: 17, b: 28 },
        ]);

        let mut buff = [Flipper::default(); 4];
        region.read_all::<Flipper>(0, &mut buff);
        assert_eq!(buff, [
            Flipper { a: 11, b: 22 },
            Flipper { a: 13, b: 24 },
            Flipper { a: 15, b: 26 },
            Flipper { a: 17, b: 28 },
        ]);

        let buffer = unsafe { region.read::<[u8; 8]>(0) };
        assert_eq!(buffer, [22, 11, 24, 13, 26, 15, 28, 17]);
    });

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct MyFoo(u64);

    impl ConstDefault for MyFoo {
        const DEFAULT: Self = MyFoo(42);
    }

    impl Default for MyFoo {
        fn default() -> Self {
            Self::DEFAULT
        }
    }

    // Test that the Atom initialises correctly.
    backend_test!(test_atom_init, F, {
        assert_eq!(Atom::<MyFoo, F>::default().read(), MyFoo::DEFAULT);
    });

    #[test]
    fn test_struct_example() {
        struct Foo<M: ManagerBase> {
            bar: Atom<u64, M>,
            qux: Atom<[u8; 64], M>,
        }

        impl<F: Fold, M: ManagerBase> Foldable<F> for Foo<M>
        where
            Atom<u64, M>: Foldable<F>,
            Atom<[u8; 64], M>: Foldable<F>,
        {
            fn fold(&self, builder: F) -> <F as Fold>::Folded {
                let mut builder = builder.into_node_fold();
                builder.add(&self.bar);
                builder.add(&self.qux);
                builder.done()
            }
        }

        fn inner(bar: u64, qux: [u8; 64]) {
            let mut foo = Foo::<Normal> {
                bar: Atom::default(),
                qux: Atom::new([0u8; 64]),
            };

            foo.bar.write(bar);
            foo.qux.write(qux);

            // Obtain the state hash
            let hash = Hash::from_foldable(&foo);

            // Obtain the Merkle tree via the `Prove` mode
            let mut proof_foo = Foo {
                bar: foo.bar.start_proof(),
                qux: foo.qux.start_proof(),
            };

            let tree = MerkleTree::from_foldable(&proof_foo);
            let tree_root_hash = tree.root_hash();
            assert_eq!(hash, tree_root_hash);

            // Modify the values so they appear in the proof
            proof_foo.bar.write(bar.wrapping_add(1));
            proof_foo.qux.write(qux.map(|x| x.wrapping_add(1)));

            // Obtain the Merkle tree, again, to make sure the root hash has not changed
            let tree = MerkleTree::from_foldable(&proof_foo);
            let tree_root_hash = tree.root_hash();
            assert_eq!(hash, tree_root_hash);

            // Produce a proof
            let proof = merkle_tree_to_merkle_proof(tree);
            let proof_hash = proof.root_hash();
            assert_eq!(hash, proof_hash);

            // Apply the same modification on the `Normal` state in order to obtain
            // the final state hash
            foo.bar.write(bar.wrapping_add(1));
            foo.qux.write(qux.map(|x| x.wrapping_add(1)));
            let final_hash = Hash::from_foldable(&foo);

            // Verify the proof and check the final hash
            catch_not_found(|| {
                let mut verify_foo = {
                    let (bar, qux) = deserialise_owned::deserialise(ProofPart::Present(&proof))
                        .unwrap()
                        .0;
                    Foo { bar, qux }
                };

                assert_eq!(bar, verify_foo.bar.read());
                assert_eq!(qux, verify_foo.qux.read());

                // Apply the same modification to the state in `Verify` mode and check
                // that the final hash is correct
                verify_foo.bar.write(bar.wrapping_add(1));
                verify_foo.qux.write(qux.map(|x| x.wrapping_add(1)));

                let verify_hash = PartialHash::from_foldable(Some(&proof), &verify_foo)
                    .to_hash()
                    .unwrap();
                assert_eq!(verify_hash, final_hash)
            })
            .unwrap();
        }

        proptest::proptest!(|(bar: u64, qux: [u8; 64])| {
            inner(bar, qux);
        });
    }
}
