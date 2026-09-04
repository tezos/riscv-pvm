// SPDX-FileCopyrightText: 2026 Trilitech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! The order in which registry commits were made.
//!
//! Garbage collection needs to know which commits are newer than the one it is asked to collect
//! at, and that cannot be read off the commits themselves: a commit id is the registry root hash,
//! so it says what a state contains and nothing about when it was reached. The journal records the
//! order beside the roots, leaving their identity untouched and commits idempotent.
//!
//! Retention follows insertion order rather than ancestry, so a state committed without a parent
//! is kept like any other. Collecting at a root keeps every root recorded at or after it.

use std::collections::HashMap;
use std::collections::HashSet;

use octez_riscv_data::hash::Hash;

use crate::commit::CommitId;
use crate::errors::OperationalError;

/// Position of a registry commit in the order the repository recorded them.
///
/// Sequence numbers increase with each commit recorded and are never reused. They are internal to
/// the repository: nothing outside it should persist one, since collection removes the entries it
/// drops.
#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct Seq(u64);

impl Seq {
    /// The sequence number given to the first commit a repository records.
    pub const FIRST: Self = Self(0);

    /// The sequence number after this one.
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }

    /// The underlying position.
    pub fn raw(self) -> u64 {
        self.0
    }

    /// The position numbered `raw`.
    ///
    /// For reading a sequence number back out of somewhere it was stored.
    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
}

/// A registry root together with the position at which it was recorded.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct JournalEntry {
    /// Where this commit falls in the recorded order.
    pub seq: Seq,

    /// The registry root committed at that point.
    pub root: CommitId,
}

/// Bytes used by one encoded [`JournalEntry`].
///
/// Entries are fixed width so that the journal can be appended to without rewriting it, and so
/// that a torn write at the end of the file is recognisable by length alone.
pub(crate) const ENTRY_BYTES: usize = SEQ_BYTES + Hash::DIGEST_SIZE;

/// Bytes used by the sequence number within an encoded entry.
const SEQ_BYTES: usize = size_of::<u64>();

impl JournalEntry {
    /// Encode this entry as [`ENTRY_BYTES`] bytes.
    pub(crate) fn encode(&self) -> [u8; ENTRY_BYTES] {
        let mut bytes = [0u8; ENTRY_BYTES];
        bytes[..SEQ_BYTES].copy_from_slice(&self.seq.raw().to_le_bytes());
        bytes[SEQ_BYTES..].copy_from_slice(self.root.as_hash().as_ref());
        bytes
    }

    /// Decode an entry previously written by [`JournalEntry::encode`].
    pub(crate) fn decode(bytes: &[u8; ENTRY_BYTES]) -> Self {
        let (seq, root) = bytes.split_at(SEQ_BYTES);

        let seq = u64::from_le_bytes(seq.try_into().expect("the split is a whole u64"));
        let root: [u8; Hash::DIGEST_SIZE] = root.try_into().expect("the split is a whole digest");

        Self {
            seq: Seq(seq),
            root: CommitId::from(Hash::from(root)),
        }
    }
}

/// Decode as many whole entries as `bytes` holds, ignoring a trailing partial one.
///
/// A partial entry is a write interrupted by a crash. The commit it belongs to was never returned
/// to whoever asked for it, so nothing can reference it and dropping it is what recovery wants.
pub(crate) fn decode_entries(bytes: &[u8]) -> Vec<JournalEntry> {
    bytes
        .chunks_exact(ENTRY_BYTES)
        .map(|chunk| {
            JournalEntry::decode(chunk.try_into().expect("chunks_exact yields whole entries"))
        })
        .collect()
}

/// The position each root was most recently recorded at.
///
/// A root committed more than once keeps its highest position, so re-committing a state can only
/// ever extend how long it is retained.
pub fn latest_positions(entries: &[JournalEntry]) -> HashMap<CommitId, Seq> {
    let mut latest = HashMap::with_capacity(entries.len());
    for entry in entries {
        latest
            .entry(entry.root)
            .and_modify(|seq: &mut Seq| *seq = (*seq).max(entry.seq))
            .or_insert(entry.seq);
    }
    latest
}

/// The roots to keep when collecting at `target`.
///
/// Every root recorded at or after `target` is retained, `target` itself included. Fails with
/// [`OperationalError::CollectionTargetNotRecorded`] if the journal holds no entry for `target`,
/// since without one there is no floor to compare against and collecting would drop everything.
pub fn roots_to_retain(
    entries: &[JournalEntry],
    target: &CommitId,
) -> Result<HashSet<CommitId>, OperationalError> {
    let latest = latest_positions(entries);

    let floor =
        *latest
            .get(target)
            .ok_or_else(|| OperationalError::CollectionTargetNotRecorded {
                target: target.hex_encode(),
            })?;

    Ok(latest
        .into_iter()
        .filter_map(|(root, seq)| (seq >= floor).then_some(root))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(byte: u8) -> CommitId {
        CommitId::from(Hash::from([byte; Hash::DIGEST_SIZE]))
    }

    fn entry(seq: u64, byte: u8) -> JournalEntry {
        JournalEntry {
            seq: Seq(seq),
            root: root(byte),
        }
    }

    // An encoded entry survives a round trip through the fixed-width representation.
    #[test]
    fn encoding_round_trips() {
        let original = entry(7, 0xab);
        assert_eq!(JournalEntry::decode(&original.encode()), original);
    }

    // Whole entries are decoded and a trailing partial one - a write a crash cut short - is
    // dropped rather than failing the read.
    #[test]
    fn decoding_ignores_a_torn_final_entry() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&entry(0, 1).encode());
        bytes.extend_from_slice(&entry(1, 2).encode());
        bytes.extend_from_slice(&entry(2, 3).encode()[..ENTRY_BYTES - 1]);

        assert_eq!(decode_entries(&bytes), vec![entry(0, 1), entry(1, 2)]);
    }

    // Retention takes every root at or after the target, and excludes older ones.
    #[test]
    fn retains_the_target_and_everything_after_it() {
        let entries = vec![entry(0, 1), entry(1, 2), entry(2, 3), entry(3, 4)];

        let retained = roots_to_retain(&entries, &root(3)).expect("the target is recorded");

        assert_eq!(retained, HashSet::from([root(3), root(4)]));
    }

    // A root committed twice is retained on its most recent position, so an old state that has
    // been re-committed survives collection at a target recorded between the two.
    #[test]
    fn a_recommitted_root_keeps_its_latest_position() {
        let entries = vec![entry(0, 1), entry(1, 2), entry(2, 1)];

        let retained = roots_to_retain(&entries, &root(2)).expect("the target is recorded");

        assert_eq!(retained, HashSet::from([root(1), root(2)]));
    }

    // Collecting at the oldest recorded root keeps everything.
    #[test]
    fn collecting_at_the_oldest_root_retains_all() {
        let entries = vec![entry(0, 1), entry(1, 2), entry(2, 3)];

        let retained = roots_to_retain(&entries, &root(1)).expect("the target is recorded");

        assert_eq!(retained, HashSet::from([root(1), root(2), root(3)]));
    }

    // An unrecorded target is refused rather than treated as a floor that retains nothing.
    #[test]
    fn an_unrecorded_target_is_refused() {
        let entries = vec![entry(0, 1)];

        assert!(matches!(
            roots_to_retain(&entries, &root(9)),
            Err(OperationalError::CollectionTargetNotRecorded { .. })
        ));
    }
}
