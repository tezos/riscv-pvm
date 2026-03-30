// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Durable storage interfaces

use bincode::Decode;
use bincode::Encode;
use bytes::Bytes;
use octez_riscv_data::clone::CloneState;
use octez_riscv_data::foldable::Fold;
use octez_riscv_data::foldable::Foldable;
use octez_riscv_data::foldable::NodeFold;
use octez_riscv_data::hash::Hash;
use octez_riscv_data::merkle_proof::DeserialiserNode;
use octez_riscv_data::merkle_proof::FromProof;
use octez_riscv_data::mode::Mode;
use octez_riscv_data::mode::Normal;
use octez_riscv_data::mode::Provable;
use octez_riscv_durable_storage::errors::Error as DurableError;
use octez_riscv_durable_storage::errors::InvalidArgumentError;
use octez_riscv_durable_storage::errors::OperationalError;
use octez_riscv_durable_storage::key::Key;
#[cfg(feature = "rocksdb")]
use octez_riscv_durable_storage::persistence_layer::PersistenceLayer;
use octez_riscv_durable_storage::registry::CloneRegistryMode;
use octez_riscv_durable_storage::registry::Registry;
use octez_riscv_durable_storage::storage::KeyValueStore;

/// Implementing types provide an interface for durable storage
pub trait DurableStorage<M: Mode>: Sized {
    fn try_clone(&self) -> Result<Self, OperationalError>
    where
        M: CloneRegistryMode;
}

/// Errors raised by the durable-storage ECALL runtime.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("durable storage host functions are not supported by this backend")]
    NotSupported,

    #[error(transparent)]
    Durable(#[from] DurableError),

    #[error(transparent)]
    InvalidArgument(#[from] InvalidArgumentError),

    #[error(transparent)]
    Operational(#[from] OperationalError),
}

/// Runtime durable-storage operations exposed to the Tezos ECALL handler.
pub trait RuntimeDurableStorage {
    fn registry_len(&self) -> Result<usize, RuntimeError>;

    fn registry_resize_tick(&mut self, new_size: usize) -> Result<(), RuntimeError>;

    fn registry_copy_database(
        &mut self,
        src_index: usize,
        dst_index: usize,
    ) -> Result<(), RuntimeError>;

    fn registry_move_database(
        &mut self,
        src_index: usize,
        dst_index: usize,
    ) -> Result<(), RuntimeError>;

    fn registry_clear_database(&mut self, index: usize) -> Result<(), RuntimeError>;

    fn database_exists(&self, index: usize, key: &[u8]) -> Result<bool, RuntimeError>;

    fn database_delete(&mut self, index: usize, key: &[u8]) -> Result<bool, RuntimeError>;

    fn database_value_length(&self, index: usize, key: &[u8]) -> Result<usize, RuntimeError>;

    fn database_read(
        &self,
        index: usize,
        key: &[u8],
        offset: usize,
        max_bytes: usize,
    ) -> Result<Vec<u8>, RuntimeError>;

    fn database_set(&mut self, index: usize, key: &[u8], data: &[u8]) -> Result<(), RuntimeError>;

    fn database_write(
        &mut self,
        index: usize,
        key: &[u8],
        offset: usize,
        data: &[u8],
    ) -> Result<usize, RuntimeError>;

    fn database_hash(&self, index: usize) -> Result<Hash, RuntimeError>;
}

impl<KV: KeyValueStore + Send + Sync + 'static, M: Mode> DurableStorage<M> for Registry<KV, M>
where
    KV::Repo: Clone,
{
    fn try_clone(&self) -> Result<Self, OperationalError>
    where
        M: CloneRegistryMode,
    {
        M::try_clone(self)
    }
}

#[cfg(feature = "rocksdb")]
impl RuntimeDurableStorage for Registry<PersistenceLayer, Normal> {
    fn registry_len(&self) -> Result<usize, RuntimeError> {
        Ok(self.len())
    }

    fn registry_resize_tick(&mut self, new_size: usize) -> Result<(), RuntimeError> {
        self.resize_tick(new_size)?;
        Ok(())
    }

    fn registry_copy_database(
        &mut self,
        src_index: usize,
        dst_index: usize,
    ) -> Result<(), RuntimeError> {
        self.copy_database(src_index, dst_index)?;
        Ok(())
    }

    fn registry_move_database(
        &mut self,
        src_index: usize,
        dst_index: usize,
    ) -> Result<(), RuntimeError> {
        self.move_database(src_index, dst_index)?;
        Ok(())
    }

    fn registry_clear_database(&mut self, index: usize) -> Result<(), RuntimeError> {
        self.clear_database(index)?;
        Ok(())
    }

    fn database_exists(&self, index: usize, key: &[u8]) -> Result<bool, RuntimeError> {
        let key = Key::new(key)?;
        Ok(self.database(index)?.exists(&key)?)
    }

    fn database_delete(&mut self, index: usize, key: &[u8]) -> Result<bool, RuntimeError> {
        let key = Key::new(key)?;
        let existed = self.database(index)?.exists(&key)?;
        self.database_mut(index)?.delete(key)?;
        Ok(existed)
    }

    fn database_value_length(&self, index: usize, key: &[u8]) -> Result<usize, RuntimeError> {
        let key = Key::new(key)?;
        Ok(self.database(index)?.value_length(&key)?)
    }

    fn database_read(
        &self,
        index: usize,
        key: &[u8],
        offset: usize,
        max_bytes: usize,
    ) -> Result<Vec<u8>, RuntimeError> {
        let key = Key::new(key)?;
        let mut buffer = vec![0u8; max_bytes];
        let bytes_read = self.database(index)?.read(&key, offset, &mut buffer[..])?;
        buffer.truncate(bytes_read);
        Ok(buffer)
    }

    fn database_set(&mut self, index: usize, key: &[u8], data: &[u8]) -> Result<(), RuntimeError> {
        let key = Key::new(key)?;
        self.database_mut(index)?
            .set(key, Bytes::copy_from_slice(data))?;
        Ok(())
    }

    fn database_write(
        &mut self,
        index: usize,
        key: &[u8],
        offset: usize,
        data: &[u8],
    ) -> Result<usize, RuntimeError> {
        let key = Key::new(key)?;
        Ok(self
            .database_mut(index)?
            .write(key, offset, Bytes::copy_from_slice(data))?)
    }

    fn database_hash(&self, index: usize) -> Result<Hash, RuntimeError> {
        Ok(self.database(index)?.hash()?)
    }
}

/// Dummy implementation for Durable Storage
///
/// This type's purpose is primarily to aid with the integration of the durable storage into the PVM.
/// The idea is that you can always use this type in place of the PVM's "durable storage" type
/// parameter, in order to make the entire code base compile.
///
/// A non-goal is to provide a run-time working implementation of a durable storage system.
#[derive(Debug, Default, Clone, Encode, Decode, PartialEq, Eq)]
pub struct DurableStorageDummy;

impl<M: Mode> DurableStorage<M> for DurableStorageDummy {
    fn try_clone(&self) -> Result<Self, OperationalError> {
        Ok(Self)
    }
}

impl RuntimeDurableStorage for DurableStorageDummy {
    fn registry_len(&self) -> Result<usize, RuntimeError> {
        Err(RuntimeError::NotSupported)
    }

    fn registry_resize_tick(&mut self, _new_size: usize) -> Result<(), RuntimeError> {
        Err(RuntimeError::NotSupported)
    }

    fn registry_copy_database(
        &mut self,
        _src_index: usize,
        _dst_index: usize,
    ) -> Result<(), RuntimeError> {
        Err(RuntimeError::NotSupported)
    }

    fn registry_move_database(
        &mut self,
        _src_index: usize,
        _dst_index: usize,
    ) -> Result<(), RuntimeError> {
        Err(RuntimeError::NotSupported)
    }

    fn registry_clear_database(&mut self, _index: usize) -> Result<(), RuntimeError> {
        Err(RuntimeError::NotSupported)
    }

    fn database_exists(&self, _index: usize, _key: &[u8]) -> Result<bool, RuntimeError> {
        Err(RuntimeError::NotSupported)
    }

    fn database_delete(&mut self, _index: usize, _key: &[u8]) -> Result<bool, RuntimeError> {
        Err(RuntimeError::NotSupported)
    }

    fn database_value_length(&self, _index: usize, _key: &[u8]) -> Result<usize, RuntimeError> {
        Err(RuntimeError::NotSupported)
    }

    fn database_read(
        &self,
        _index: usize,
        _key: &[u8],
        _offset: usize,
        _max_bytes: usize,
    ) -> Result<Vec<u8>, RuntimeError> {
        Err(RuntimeError::NotSupported)
    }

    fn database_set(
        &mut self,
        _index: usize,
        _key: &[u8],
        _data: &[u8],
    ) -> Result<(), RuntimeError> {
        Err(RuntimeError::NotSupported)
    }

    fn database_write(
        &mut self,
        _index: usize,
        _key: &[u8],
        _offset: usize,
        _data: &[u8],
    ) -> Result<usize, RuntimeError> {
        Err(RuntimeError::NotSupported)
    }

    fn database_hash(&self, _index: usize) -> Result<Hash, RuntimeError> {
        Err(RuntimeError::NotSupported)
    }
}

impl<F: Fold> Foldable<F> for DurableStorageDummy {
    fn fold(&self, builder: F) -> <F as Fold>::Folded {
        builder.into_node_fold().done()
    }
}

impl FromProof for DurableStorageDummy {
    fn from_proof<Proof: octez_riscv_data::merkle_proof::Deserialiser>(
        proof: Proof,
    ) -> octez_riscv_data::merkle_proof::SuspendedResult<Proof, Self> {
        let node = proof.into_node()?;
        node.done(Self)
    }
}

impl<'normal> Provable<'normal> for DurableStorageDummy {
    type Prover = Self;

    fn start_proof(&'normal self) -> Self::Prover {
        Self
    }
}

impl CloneState for DurableStorageDummy {
    fn clone_state(&self) -> Self {
        Self
    }
}
