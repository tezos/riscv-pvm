use std::array::TryFromSliceError;

use bincode::error::DecodeError;
use thiserror::Error;

use super::MerkleLayerError;
use crate::persistence_layer::PersistenceLayerError;

#[derive(Debug, Error)]
pub(crate) enum MavlNodeResolverError {
    #[error("The node is not resolved")]
    Unresolved,

    #[error("Node data missing in the key value store")]
    MissingInKeyValueStore(#[from] PersistenceLayerError),

    #[error("Cannot be resolved due to missing commit hash")]
    MissingCommitHash,

    #[error("Intermediate representation parsing failed")]
    FailedParsingIntermediateRepresentation(#[from] DecodeError),

    #[error("Child hash has the wrong format")]
    ChildHasWrongFormat(#[from] TryFromSliceError),

    #[error("The key has the wrong format")]
    KeyHasWrongFormat(#[from] MerkleLayerError),
}
