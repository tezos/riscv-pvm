use std::collections::HashSet;

use crate::state_backend::Layout;
use crate::state_backend::proof_backend::tree::Tree;

#[derive(Default)]
pub struct MerkleTracker {
    reads: HashSet<Vec<u8>>,
    writes: HashSet<Vec<u8>>,
}

pub enum AccessType {
    Read,
    Write,
    ReadWrite,
    NeedHash,
}

pub type AccessTree = Tree<AccessType>;

impl MerkleTracker {
    pub fn log_read(&mut self, path: &Vec<u8>) {
        self.reads.insert(path.clone());
    }

    pub fn log_write(&mut self, path: &Vec<u8>) {
        self.writes.insert(path.clone());
    }

    pub fn reset(&mut self) {
        self.reads.clear();
        self.writes.clear();
    }

    pub fn compute_proof_map<L: Layout>(&self) -> AccessTree {
        // Compute the tree based on the layout and access patterns

        todo!()
    }
}
