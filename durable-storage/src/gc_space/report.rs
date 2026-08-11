// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Helpers for converting samples into printed output.
//!
//! One line per sample while the run proceeds, then a summary of what the whole sequence showed.

use std::fs;
use std::io;
use std::io::BufWriter;
use std::io::Write;

use anyhow::Context;
use anyhow::Result;

use super::prune::PruneOutcome;
use super::sample::Sample;
use super::sample::Sharing;

/// Write a sample to the JSON output, if there is one.
pub(super) fn record(json: &mut Option<BufWriter<fs::File>>, sample: &Sample) -> Result<()> {
    let Some(json) = json else {
        return Ok(());
    };

    let line = serde_json::to_string(sample).context("encoding a sample")?;
    writeln!(json, "{line}").context("writing a sample")?;

    Ok(())
}

pub(super) fn report_header(out: &mut impl Write) -> io::Result<()> {
    writeln!(out)?;
    writeln!(
        out,
        "{:>7}  {:>10}  {:>10}  {:>10}  {:>7}  {:>10}  {:>10}  {:>9}  {:>11}  {:>10}  {:>8}  {:>11}",
        "commit",
        "blob MiB",
        "live MiB",
        "dead MiB",
        "dead %",
        "value MiB",
        "disk MiB",
        "commit ms",
        "commits MiB",
        "new MiB",
        "shared %",
        "rewrote MiB"
    )?;

    Ok(())
}

pub(super) fn report(out: &mut impl Write, sample: &Sample) -> io::Result<()> {
    // The last three columns are the sharing question: how many bytes this commit added that the
    // previous one could not share, what fraction it did manage to reuse, and how much of what the
    // previous checkpoint pinned this one no longer references. That last figure is only non-zero
    // when a compaction ran, and it is what explains a `shared %` that has collapsed: the content
    // is the same, but every file holding it has a new name, so none of it can be shared.
    let (new, shared, rewrote) = match &sample.sharing {
        Some(sharing) => (
            format!("{:.1}", mib(sharing.new_bytes)),
            format!("{:.1}%", sharing.carried_fraction() * 100.0),
            format!("{:.1}", mib(sharing.dropped_bytes)),
        ),
        None => ("-".to_owned(), "-".to_owned(), "-".to_owned()),
    };

    // `disk` is the whole repository, the live databases included, so a compaction shows there as
    // soon as it lands. `commits` is the retained history alone, which is the same population the
    // sharing columns compare: without it a row can show the repository growing by fifty megabytes
    // while claiming its checkpoint shared almost everything, because the bytes are not in a
    // checkpoint yet.
    writeln!(
        out,
        "{:>7}  {:>10.1}  {:>10.1}  {:>10.1}  {:>6.1}%  {:>10.1}  {:>10.1}  {:>9}  {:>11.1}  \
         {:>10}  {:>8}  {:>11}",
        sample.commit,
        mib(sample.blob.stored_bytes),
        mib(sample.blob.live_bytes),
        mib(sample.blob.dead_bytes()),
        sample.blob.dead_fraction() * 100.0,
        mib(sample.value_stored_bytes),
        mib(sample.disk.unique_bytes),
        sample.commit_ms,
        mib(sample.disk_commits.unique_bytes),
        new,
        shared,
        rewrote,
    )?;

    Ok(())
}

/// Report the headline figures: how much dead node data each commit leaves behind, and how much of
/// the storage that has become by the end.
pub(super) fn summarise(out: &mut impl Write, samples: &[Sample]) -> io::Result<()> {
    let (Some(first), Some(last)) = (samples.first(), samples.last()) else {
        return Ok(());
    };

    let commits = last.commit.saturating_sub(first.commit);

    writeln!(out)?;
    writeln!(out, "over {commits} commit(s):")?;

    if commits > 0 {
        let dead_growth = last
            .blob
            .dead_bytes()
            .saturating_sub(first.blob.dead_bytes());
        let blob_growth = last
            .blob
            .stored_bytes
            .saturating_sub(first.blob.stored_bytes);

        writeln!(
            out,
            "  dead node data grew by {:.1} MiB, {:.2} MiB per commit",
            mib(dead_growth),
            mib(dead_growth) / commits as f64,
        )?;
        writeln!(
            out,
            "  blob column family grew by {:.1} MiB, {:.2} MiB per commit",
            mib(blob_growth),
            mib(blob_growth) / commits as f64,
        )?;

        // The two figures a different shape can be extrapolated from: how many nodes a commit
        // rewrites (which follows the tree's depth) and what a node costs to store.
        let node_growth = last.blob.entries.saturating_sub(first.blob.entries);

        if node_growth > 0 {
            writeln!(
                out,
                "  {} node(s) written, {} per commit, averaging {} B stored per node",
                node_growth,
                node_growth / commits as u64,
                blob_growth / node_growth,
            )?;
        }

        // What retention costs, and which half of the storage it is spent holding. The `blob` part
        // is duplication a shared Merkle store would remove outright, as opposed to the dead node
        // data above, which only deleting the node keys can reclaim.
        let pinned = last.pinned;
        let pinned_growth = pinned.total().saturating_sub(first.pinned.total());

        writeln!(
            out,
            "  history pins {:.1} MiB ({:.1} MiB Merkle, {:.1} MiB values, {:.1} MiB other), \
             growing {:.1} MiB per commit",
            mib(pinned.total()),
            mib(pinned.blob),
            mib(pinned.value),
            mib(pinned.other),
            mib(pinned_growth) / commits as f64,
        )?;

        if pinned.total() > 0 {
            writeln!(
                out,
                "  {:.1}% of what the history pins is Merkle node data, which a shared store would \
                 stop duplicating per commit",
                pinned.blob as f64 / pinned.total() as f64 * 100.0,
            )?;
        }

        // Why retention costs what it does. If commits shared everything they could, `new` would be
        // about the size of what each commit changed; the gap is compaction rewriting untouched
        // data, which the previous checkpoint then pins in its old form.
        let shared: Vec<&Sharing> = samples.iter().filter_map(|s| s.sharing.as_ref()).collect();

        if !shared.is_empty() {
            let new_bytes: u64 = shared.iter().map(|s| s.new_bytes).sum();
            let new_files: u64 = shared.iter().map(|s| s.new_files).sum();
            let overall = pooled_sharing(shared.iter().copied());
            let quiet: Vec<&Sharing> = shared
                .iter()
                .copied()
                .filter(|s| s.dropped_files == 0)
                .collect();

            writeln!(
                out,
                "  sharing: {:.1}% of the bytes checkpoints pinned were already on disk \
                 ({:.1} MiB of {:.1} MiB), adding {:.1} MiB in {} new file(s) per commit",
                overall.fraction() * 100.0,
                mib(overall.carried),
                mib(overall.total()),
                mib(new_bytes) / shared.len() as f64,
                new_files / shared.len() as u64,
            )?;

            // What a compaction costs, as the gap between the two. Reporting only the figure that
            // includes the rewrites hides how much sharing there is to lose; reporting only the
            // one that excludes them hides that it was lost.
            if quiet.len() < shared.len() && !quiet.is_empty() {
                let steady = pooled_sharing(quiet.iter().copied());
                let dropped: u64 = shared.iter().map(|s| s.dropped_bytes).sum();

                writeln!(
                    out,
                    "  {} of {} measured commit(s) followed a compaction; without them sharing is \
                     {:.1}%, so compaction costs {:.1} points of it, rewriting {:.1} MiB the \
                     earlier checkpoints still pin",
                    shared.len() - quiet.len(),
                    shared.len(),
                    steady.fraction() * 100.0,
                    (steady.fraction() - overall.fraction()) * 100.0,
                    mib(dropped),
                )?;
            }
        }

        writeln!(out, "  levels at the last commit:")?;

        for level in &last.levels {
            writeln!(
                out,
                "    L{}: {} file(s), {:.1} MiB",
                level.level,
                level.files,
                mib(level.bytes)
            )?;
        }
    }

    writeln!(
        out,
        "  {:.1}% of the blob column family is now dead ({:.1} MiB of {:.1} MiB)",
        last.blob.dead_fraction() * 100.0,
        mib(last.blob.dead_bytes()),
        mib(last.blob.stored_bytes),
    )?;
    writeln!(
        out,
        "  repository occupies {:.1} MiB across {} commit directories ({:.1} MiB apparent, and \
         hard links save {:.1} MiB)",
        mib(last.disk.unique_bytes),
        last.commit_dirs,
        mib(last.disk.apparent_bytes),
        mib(last.disk.shared_bytes()),
    )?;

    Ok(())
}

/// Report what a directory-level collection reclaimed, and what it could not.
///
/// The second figure is the one that matters: whatever dead node data remains after this is data no
/// directory deletion can ever reach, because the retained commit still references the files it
/// sits in.
pub(super) fn report_prune(
    out: &mut impl Write,
    outcome: &PruneOutcome,
    last: Option<&Sample>,
) -> io::Result<()> {
    writeln!(out)?;
    writeln!(
        out,
        "directory-level collection: removed {} database commit(s) and {} manifest(s)",
        outcome.databases_removed, outcome.registries_removed,
    )?;
    writeln!(
        out,
        "  repository went from {:.1} MiB to {:.1} MiB, freeing {:.1} MiB",
        mib(outcome.before.unique_bytes),
        mib(outcome.after.unique_bytes),
        mib(outcome.freed_bytes()),
    )?;
    writeln!(
        out,
        "  retained history now occupies {:.1} MiB",
        mib(outcome.after_commits.unique_bytes),
    )?;

    let Some(last) = last else {
        return Ok(());
    };

    writeln!(
        out,
        "  still dead and now unreachable by any directory deletion: {:.1} MiB of node data \
         ({:.1}% of the surviving blob column family)",
        mib(last.blob.dead_bytes()),
        last.blob.dead_fraction() * 100.0,
    )?;

    Ok(())
}

/// Carried and new bytes summed over a set of commits, for one sharing figure over all of them.
#[derive(Default)]
struct PooledSharing {
    carried: u64,
    new: u64,
}

impl PooledSharing {
    fn total(&self) -> u64 {
        self.carried + self.new
    }

    fn fraction(&self) -> f64 {
        if self.total() == 0 {
            return 0.0;
        }

        self.carried as f64 / self.total() as f64
    }
}

/// Sum the commits' bytes and take one ratio, rather than averaging their per-commit ratios.
///
/// An average of ratios weights a commit that pinned forty megabytes the same as one that pinned
/// eighty. Summing first weights each commit by what it actually pinned, which is what makes the
/// figure comparable between runs of different lengths — and it is the same statistic whether or
/// not the compaction commits are included, so the two can be subtracted.
fn pooled_sharing<'a>(sharing: impl Iterator<Item = &'a Sharing>) -> PooledSharing {
    let mut pooled = PooledSharing::default();

    for one in sharing {
        pooled.carried += one.carried_bytes;
        pooled.new += one.new_bytes;
    }

    pooled
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}
