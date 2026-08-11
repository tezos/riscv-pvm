// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Measurement of a committed database, for space accounting.
//!
//! These read a commit's column families directly: how many entries each holds, what they
//! occupy, and which SST files hold them. They are whole-column-family scans, so they belong
//! to the harnesses rather than to any path where performance matters — which is why they sit
//! here rather than beside the production code that opens the instance.

use std::collections::HashMap;

use super::BLOB_CF;
use super::KV_CF;
use super::ReadOnlyPersistenceLayer;
use super::blob_cf_of;

/// One SST file of a committed database.
///
/// A *sorted string table* is the immutable file RocksDB keeps key-value pairs in: a memtable is
/// flushed into a new SST at level zero, and compaction merges those into fewer, larger SSTs at
/// deeper levels. Everything a database holds that is not a log or a manifest sits in one of
/// these, so they are the unit disk usage is attributed in — and since a checkpoint hard-links
/// them rather than copying them, they are also the unit two commits share.
#[derive(Debug, Clone)]
pub struct SstFile {
    /// File name, as it appears in the commit directory.
    pub name: String,

    /// Which column family's data it holds.
    pub owner: SstOwner,

    /// LSM level it sits at.
    pub level: i32,

    /// Size in bytes, as RocksDB reports it.
    pub size: u64,

    /// Number of entries it holds.
    pub entries: u64,
}

/// Which column family an SST file holds data for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SstOwner {
    /// Merkle node bodies.
    Blob,

    /// Values.
    Value,
}

/// Byte totals for one column family, counted by scanning every entry in it.
///
/// These are the bytes as stored logically, before compression or any per-SST overhead. Compare
/// against [`ReadOnlyPersistenceLayer::sst_bytes`] for what the files actually occupy.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CfTotals {
    /// Number of key-value pairs.
    pub entries: u64,

    /// Total length of all keys.
    pub key_bytes: u64,

    /// Total length of all values.
    pub value_bytes: u64,
}

impl CfTotals {
    /// Total bytes stored, keys and values together.
    pub fn stored_bytes(&self) -> u64 {
        self.key_bytes + self.value_bytes
    }
}

/// Count the entries and bytes in the column family `cf` by iterating all of it.
///
/// A full scan rather than a RocksDB estimate, because the point of measuring is to know how much
/// of a column family is dead, and an estimate cannot be subtracted from an exact live figure.
fn scan_cf(
    db: &rocksdb::DB,
    cf: &rocksdb::ColumnFamily,
) -> Result<CfTotals, Box<dyn std::error::Error>> {
    let mut totals = CfTotals::default();
    let mut iter = db.raw_iterator_cf(cf);
    iter.seek_to_first();

    while iter.valid() {
        totals.entries += 1;
        totals.key_bytes += iter.key().map_or(0, |key| key.len() as u64);
        totals.value_bytes += iter.value().map_or(0, |value| value.len() as u64);
        iter.next();
    }

    // An iterator that stopped because of an error also reports `!valid()`, so the loop above
    // cannot tell a complete scan from a truncated one without this.
    iter.status()?;

    Ok(totals)
}

/// Which column family an SST file belongs to, from the name RocksDB reports for it.
///
/// A database has exactly these two families, so any other name means the instance is not one of
/// ours — and attributing its bytes to either family would be a guess.
fn sst_owner_of(column_family: &str) -> Result<SstOwner, Box<dyn std::error::Error>> {
    match column_family {
        BLOB_CF => Ok(SstOwner::Blob),
        KV_CF => Ok(SstOwner::Value),
        other => Err(format!("unexpected column family {other} in a database instance").into()),
    }
}

/// Measurement of a committed database, for space accounting.
///
/// Only available to the harnesses: these are whole-column-family scans, far too expensive to sit
/// on any path where performance matters.
impl ReadOnlyPersistenceLayer {
    /// Scan the column family holding Merkle node bodies.
    pub fn blob_totals(&self) -> Result<CfTotals, Box<dyn std::error::Error>> {
        scan_cf(&self.db_instance, blob_cf_of(&self.db_instance))
    }

    /// Scan the column family holding values.
    pub fn value_totals(&self) -> Result<CfTotals, Box<dyn std::error::Error>> {
        let cf = self
            .db_instance
            .cf_handle(KV_CF)
            .ok_or("the rocksdb instance should always contain the default cf")?;

        scan_cf(&self.db_instance, cf)
    }

    /// Every SST file of this instance, with the level and column family it belongs to.
    ///
    /// Comparing these sets between successive commits is how sharing is measured: a checkpoint
    /// hard-links the files live when it was taken, so a file present in two checkpoints of the
    /// same database is one file on disk, and a file present in only the later one was written by
    /// compaction or a flush in between.
    pub fn sst_files(&self) -> Result<Vec<SstFile>, Box<dyn std::error::Error>> {
        self.db_instance
            .live_files()?
            .into_iter()
            .map(|file| {
                Ok(SstFile {
                    owner: sst_owner_of(&file.column_family_name)?,
                    name: file.name.trim_start_matches('/').to_owned(),
                    level: file.level,
                    size: file.size as u64,
                    entries: file.num_entries,
                })
            })
            .collect()
    }

    /// Which column family each SST file of this instance belongs to, keyed by file name.
    ///
    /// The SST files of both column families sit in one directory with no distinguishing names, so
    /// attributing disk to one or the other has to come from RocksDB's own metadata. Any leading
    /// separator is stripped from the names so they match directory entries directly.
    pub fn sst_column_families(
        &self,
    ) -> Result<HashMap<String, SstOwner>, Box<dyn std::error::Error>> {
        let mut owners = HashMap::new();

        for file in self.db_instance.live_files()? {
            let owner = sst_owner_of(&file.column_family_name)?;

            owners.insert(file.name.trim_start_matches('/').to_owned(), owner);
        }

        Ok(owners)
    }

    /// Size of the SST files backing both column families, as RocksDB reports it.
    pub fn sst_bytes(&self) -> Result<u64, Box<dyn std::error::Error>> {
        let mut total = 0;

        for cf in [
            blob_cf_of(&self.db_instance),
            self.db_instance
                .cf_handle(KV_CF)
                .ok_or("the rocksdb instance should always contain the default cf")?,
        ] {
            total += self
                .db_instance
                .property_int_value_cf(cf, "rocksdb.total-sst-files-size")?
                .unwrap_or(0);
        }

        Ok(total)
    }
}
