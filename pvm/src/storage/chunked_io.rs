// SPDX-FileCopyrightText: 2024 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

use std::cmp;
use std::collections::VecDeque;
use std::io;
use std::io::Cursor;
use std::io::Write;

use octez_riscv_data::hash::Hash;
use octez_riscv_data::hash::HashedData;
use octez_riscv_data::serialisation::deserialise;
use octez_riscv_data::store::BlobStore;

use super::CHUNK_SIZE;
use super::StorageError;

/// Simple writer that stores data in chunks of size [`CHUNK_SIZE`] in any [`BlobStore`].
pub struct ChunkWriter<'a, BS> {
    store: &'a BS,
    hashes: Vec<Hash>,
    buffer: Vec<u8>,
}

impl<'a, BS: BlobStore> ChunkWriter<'a, BS> {
    /// Create a new writer that writes the chunks to the given [`BlobStore`].
    pub fn new(store: &'a BS) -> Self {
        Self {
            store,
            hashes: Vec::new(),
            buffer: Vec::with_capacity(CHUNK_SIZE),
        }
    }

    /// Finalise the writer by flushing any remaining chunks-in-progress to the store and returning
    /// the list of identifiers for each chunk that was written.
    pub fn finalise(mut self) -> Result<Vec<Hash>, StorageError> {
        if !self.buffer.is_empty() {
            self.flush_buffer()?;
        }

        Ok(self.hashes)
    }

    /// Write a chunk to the store.
    fn flush_buffer(&mut self) -> Result<(), StorageError> {
        let hashed = HashedData::from_data(&self.buffer);
        self.store.blob_set(&hashed)?;
        self.hashes.push(hashed.hash());
        self.buffer.clear();
        Ok(())
    }
}

impl<BS: BlobStore> Write for ChunkWriter<'_, BS> {
    fn write(&mut self, mut data: &[u8]) -> io::Result<usize> {
        let ret = data.len();

        while !data.is_empty() {
            let rem_buffer_len = CHUNK_SIZE - self.buffer.len();
            let new_data_len = cmp::min(rem_buffer_len, data.len());

            // Take the data from the input.
            let new_data = &data[..new_data_len];
            data = &data[new_data_len..];
            self.buffer.extend_from_slice(new_data);

            // If the buffer has been completely filled, flush it.
            if rem_buffer_len == new_data_len {
                self.flush_buffer().map_err(io::Error::other)?;
            }
        }

        Ok(ret)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Just like [`ChunkWriter`], but for reading.
pub struct ChunkedReader<'a, BS> {
    store: &'a BS,
    hashes: VecDeque<Hash>,
    buffer: Cursor<Vec<u8>>,
}

impl<'a, BS: BlobStore> ChunkedReader<'a, BS> {
    /// Create a new reader that pulls the chunks from the given [`BlobStore`].
    pub fn new(store: &'a BS, hash: &Hash) -> Result<Self, StorageError> {
        let raw_hashes = store.blob_get(*hash)?;
        let hashes = deserialise(raw_hashes.as_ref())?;
        Ok(Self {
            store,
            hashes,
            buffer: Cursor::new(Vec::with_capacity(CHUNK_SIZE)),
        })
    }

    /// Start reading the next chunk.
    fn next_chunk(&mut self) -> Result<(), StorageError> {
        let Some(hash) = self.hashes.pop_front() else {
            return Ok(());
        };

        let bytes = self.store.blob_get(hash)?;

        self.buffer.get_mut().clear();
        self.buffer.get_mut().extend_from_slice(bytes.as_ref());
        self.buffer.set_position(0);

        Ok(())
    }
}

impl<BS: BlobStore> io::Read for ChunkedReader<'_, BS> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.buffer.position() as usize >= self.buffer.get_ref().len() {
            self.next_chunk().map_err(io::Error::other)?;
        }

        self.buffer.read(buf)
    }
}
