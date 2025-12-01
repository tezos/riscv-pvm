use crate::merkle_layer::hash;
use crate::merkle_layer::node::MavlNode;
use octez_riscv_data::hash::DIGEST_SIZE;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Debug)]
enum CommitOperationType {
    Insert,
    Remove,
}

#[derive(Clone, Debug)]
struct CommitOperation {
    operation_type: CommitOperationType,
    data: Option<Vec<u8>>,
}

impl CommitOperation {
    fn insert_operation(data: Vec<u8>) -> Self {
        Self {
            operation_type: CommitOperationType::Insert,
            data: Some(data),
        }
    }

    fn remove_operation() -> Self {
        Self {
            operation_type: CommitOperationType::Remove,
            data: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct CommitOperationCollection {
    collection: HashMap<[u8; DIGEST_SIZE], CommitOperation>,
}

impl CommitOperationCollection {
    pub(crate) fn add_new_node_to_commit(&mut self, node: &Arc<MavlNode>) {
        let node_hash: [u8; 32] = *hash(node).as_bytes();
        let serialized_node = node.encode_to_vec();
        self.collection.insert(
            node_hash,
            CommitOperation::insert_operation(serialized_node),
        );
    }

    pub(crate) fn remove_node_from_commit(&mut self, node: &Arc<MavlNode>) {
        let node_hash: [u8; 32] = *hash(node).as_bytes();
        self.collection
            .insert(node_hash, CommitOperation::remove_operation());
    }
}
