// SPDX-FileCopyrightText: 2026 Nomadic Labs <contact@nomadic-labs.com>
//
// SPDX-License-Identifier: MIT

//! Helpers for converting samples into printed output.
//!
//! One line per sample while the run proceeds, then a summary of what the whole sequence showed.

use std::io;
use std::io::Write;

use super::sample::Sample;
use super::sample::Sharing;

pub(super) fn report_header(out: &mut impl Write) -> io::Result<()> {
    writeln!(out)?;
    writeln!(
        out,
        "{:>7}  {:>10}  {:>10}  {:>10}  {:>7}  {:>10}  {:>10}  {:>9}  {:>10}  {:>7}",
        "commit",
        "blob MiB",
        "live MiB",
        "dead MiB",
        "dead %",
        "value MiB",
        "disk MiB",
        "commit ms",
        "new MiB",
        "shared %"
    )?;

    Ok(())
}

pub(super) fn report(out: &mut impl Write, sample: &Sample) -> io::Result<()> {
    // The last two columns are the sharing question: how many bytes this commit added that the
    // previous one could not share, and what fraction it did manage to reuse.
    let (new, shared) = match &sample.sharing {
        Some(sharing) => (
            format!("{:.1}", mib(sharing.new_bytes)),
            format!("{:.1}%", sharing.carried_fraction() * 100.0),
        ),
        None => ("-".to_owned(), "-".to_owned()),
    };

    writeln!(
        out,
        "{:>7}  {:>10.1}  {:>10.1}  {:>10.1}  {:>6.1}%  {:>10.1}  {:>10.1}  {:>9}  {:>10}  {:>7}",
        sample.commit,
        mib(sample.blob.stored_bytes),
        mib(sample.blob.live_bytes),
        mib(sample.blob.dead_bytes()),
        sample.blob.dead_fraction() * 100.0,
        mib(sample.value_stored_bytes),
        mib(sample.disk.unique_bytes),
        sample.commit_ms,
        new,
        shared,
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
            let mean_carried =
                shared.iter().map(|s| s.carried_fraction()).sum::<f64>() / shared.len() as f64;

            writeln!(
                out,
                "  sharing: each measured commit reused {:.1}% of the previous commit's bytes, \
                 adding {:.1} MiB in {} new file(s) on average",
                mean_carried * 100.0,
                mib(new_bytes) / shared.len() as f64,
                new_files / shared.len() as u64,
            )?;
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
        "  repository occupies {:.1} MiB across {} commit directories ({:.1} MiB apparent, so \
         hard links save {:.1} MiB)",
        mib(last.disk.unique_bytes),
        last.commit_dirs,
        mib(last.disk.apparent_bytes),
        mib(last
            .disk
            .apparent_bytes
            .saturating_sub(last.disk.unique_bytes)),
    )?;

    Ok(())
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}
