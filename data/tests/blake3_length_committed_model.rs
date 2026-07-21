// SPDX-FileCopyrightText: 2026 TriliTech <contact@trili.tech>
// SPDX-License-Identifier: MIT
//! Reference model validating the LENGTH-COMMITTED design
//! (`data/docs/blake3-bytes-length-committed.md`): confirms it REJECTS the campaign
//! forgeries, round-trips, and gives O(depth) position-independent proofs with free blinding.

use std::ops::Range;

use blake3::CHUNK_LEN;
use blake3::hazmat::HasherExt;
use blake3::hazmat::Mode as B3;
use blake3::hazmat::left_subtree_len;
use blake3::hazmat::merge_subtrees_non_root;
use blake3::hazmat::merge_subtrees_root;
use proptest::prelude::*;

#[derive(Clone, Debug)]
enum Tree {
    Chunk(Vec<u8>),
    Blind([u8; 32]),
    Node(Box<Tree>, Box<Tree>),
}

#[derive(Clone, Debug)]
struct Proof {
    length: usize,
    data: Tree,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Err_ {
    Shape,
    ChunkTooLarge,
}

// ----- §2 committed value hash -----
fn h_len(n: usize) -> [u8; 32] {
    *blake3::hash(&(n as u64).to_le_bytes()).as_bytes()
}
fn h_data(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}
fn combine(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(a);
    h.update(b);
    *h.finalize().as_bytes()
}
fn value_hash(bytes: &[u8]) -> [u8; 32] {
    combine(&h_len(bytes.len()), &h_data(bytes))
}

fn canonical_cv(data: &[u8], offset: u64) -> [u8; 32] {
    if data.len() <= CHUNK_LEN {
        return blake3::Hasher::new()
            .set_input_offset(offset)
            .update(data)
            .finalize_non_root();
    }
    let ll = left_subtree_len(data.len() as u64) as usize;
    let (l, r) = data.split_at(ll);
    merge_subtrees_non_root(
        &canonical_cv(l, offset),
        &canonical_cv(r, offset + ll as u64),
        B3::Hash,
    )
}

// ----- §4.1 prove (FREE blinding: any untouched canonical subtree collapses) -----
fn touched(offset: usize, span: usize, acc: &[Range<usize>]) -> bool {
    let end = offset + span;
    acc.iter().any(|r| r.start < end && offset < r.end)
}
fn prove(data: &[u8], acc: &[Range<usize>]) -> Proof {
    Proof {
        length: data.len(),
        data: build(data, 0, data.len(), acc, true),
    }
}
fn build(data: &[u8], offset: usize, span: usize, acc: &[Range<usize>], is_root: bool) -> Tree {
    let t = is_root && !acc.is_empty() || (!is_root && touched(offset, span, acc));
    if !t {
        return if is_root {
            Tree::Blind(h_data(data)) // whole value untouched -> H_data (ROOT output)
        } else {
            Tree::Blind(canonical_cv(&data[offset..offset + span], offset as u64))
        };
    }
    if span <= CHUNK_LEN {
        Tree::Chunk(data[offset..offset + span].to_vec())
    } else {
        let ll = left_subtree_len(span as u64) as usize;
        Tree::Node(
            Box::new(build(data, offset, ll, acc, false)),
            Box::new(build(data, offset + ll, span - ll, acc, false)),
        )
    }
}

// ----- §4.2 verify -----
fn verify(p: &Proof) -> Result<[u8; 32], Err_> {
    let hd = data_root(&p.data, p.length)?;
    Ok(combine(&h_len(p.length), &hd))
}
fn data_root(t: &Tree, length: usize) -> Result<[u8; 32], Err_> {
    match t {
        Tree::Blind(h) => Ok(*h), // top-level blind IS H_data
        Tree::Chunk(b) => {
            if b.len() > CHUNK_LEN {
                return Err(Err_::ChunkTooLarge);
            }
            if length > CHUNK_LEN || b.len() != length {
                return Err(Err_::Shape);
            }
            Ok(h_data(b))
        }
        Tree::Node(l, r) => {
            if length <= CHUNK_LEN {
                return Err(Err_::Shape);
            }
            let ll = left_subtree_len(length as u64) as usize;
            Ok(
                *merge_subtrees_root(&cv(l, 0, ll)?, &cv(r, ll as u64, length - ll)?, B3::Hash)
                    .as_bytes(),
            )
        }
    }
}
fn cv(t: &Tree, offset: u64, span: usize) -> Result<[u8; 32], Err_> {
    match t {
        Tree::Chunk(b) => {
            if b.len() > CHUNK_LEN {
                return Err(Err_::ChunkTooLarge);
            }
            if span > CHUNK_LEN || b.len() != span {
                return Err(Err_::Shape);
            }
            Ok(blake3::Hasher::new()
                .set_input_offset(offset)
                .update(b)
                .finalize_non_root())
        }
        Tree::Blind(cv) => Ok(*cv), // no blindability check needed (length pins the shape)
        Tree::Node(l, r) => {
            if span <= CHUNK_LEN {
                return Err(Err_::Shape);
            }
            let ll = left_subtree_len(span as u64) as usize;
            Ok(merge_subtrees_non_root(
                &cv(l, offset, ll)?,
                &cv(r, offset + ll as u64, span - ll)?,
                B3::Hash,
            ))
        }
    }
}

// ----- strongest attacker: build canonical(L') filled with honest cross-span CVs -----
fn forge(honest: &[u8], l_prime: usize) -> Proof {
    Proof {
        length: l_prime,
        data: fb(honest, 0, l_prime, true),
    }
}
fn fb(honest: &[u8], offset: usize, span: usize, is_root: bool) -> Tree {
    // Attacker blinds everything it can with genuine CVs of the true value (clamped).
    if !is_root {
        let mut buf = vec![0u8; span];
        let end = (offset + span).min(honest.len());
        if offset < end {
            buf[..end - offset].copy_from_slice(&honest[offset..end]);
        }
        return Tree::Blind(canonical_cv(&buf, offset as u64));
    }
    if span <= CHUNK_LEN {
        let mut buf = vec![0u8; span];
        let end = (offset + span).min(honest.len());
        if offset < end {
            buf[..end - offset].copy_from_slice(&honest[offset..end]);
        }
        Tree::Chunk(buf)
    } else {
        let ll = left_subtree_len(span as u64) as usize;
        Tree::Node(
            Box::new(fb(honest, offset, ll, false)),
            Box::new(fb(honest, offset + ll, span - ll, false)),
        )
    }
}

fn sizes(t: &Tree) -> (usize, usize) {
    match t {
        Tree::Chunk(_) => (1, 0),
        Tree::Blind(_) => (0, 1),
        Tree::Node(l, r) => {
            let a = sizes(l);
            let b = sizes(r);
            (a.0 + b.0, a.1 + b.1)
        }
    }
}

// ============================ campaign forgeries must be REJECTED ============================

#[test]
fn graft_3_to_4_rejected() {
    let data = vec![0xABu8; 3 * CHUNK_LEN];
    let committed = value_hash(&data);
    // Attacker's best 4096-byte proof reproducing the honest DATA root:
    let f = forge(&data, 4 * CHUNK_LEN);
    match verify(&f) {
        Err(_) => {}
        Ok(h) => assert_ne!(
            h, committed,
            "3->4 length forgery accepted under length-committed!"
        ),
    }
}

#[test]
fn graft_1025_to_2048_rejected() {
    let mut data = vec![0u8; CHUNK_LEN + 1];
    data[CHUNK_LEN] = 0x7F;
    let committed = value_hash(&data);
    let f = forge(&data, 2 * CHUNK_LEN);
    match verify(&f) {
        Err(_) => {}
        Ok(h) => assert_ne!(h, committed, "1025->2048 length forgery accepted!"),
    }
}

// ============================ round-trip, tamper, size (proptest) ============================

fn interesting_len() -> impl Strategy<Value = usize> {
    prop_oneof![
        Just(0usize),
        1usize..CHUNK_LEN,
        Just(CHUNK_LEN),
        Just(CHUNK_LEN + 1),
        Just(3 * CHUNK_LEN),
        Just(5 * CHUNK_LEN + 7),
        (0usize..=16 * CHUNK_LEN),
    ]
}
fn data_access() -> impl Strategy<Value = (Vec<u8>, Vec<Range<usize>>)> {
    interesting_len().prop_flat_map(|len| {
        let data = proptest::collection::vec(any::<u8>(), len..=len);
        let acc = if len == 0 {
            Just(Vec::new()).boxed()
        } else {
            proptest::collection::vec(
                (0usize..len)
                    .prop_flat_map(move |a| (Just(a), (a + 1)..=len))
                    .prop_map(|(a, b)| a..b),
                0..=4,
            )
            .boxed()
        };
        (data, acc)
    })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 8000, ..ProptestConfig::default() })]

    #[test]
    fn roundtrip((data, acc) in data_access()) {
        prop_assert_eq!(verify(&prove(&data, &acc)).unwrap(), value_hash(&data));
    }

    /// Any forged length (honest-tree reinterpretation OR constructed cross-span graft) fails.
    #[test]
    fn any_forged_length_rejected((data, acc) in data_access(), k in 1usize..=20, boundary in any::<bool>()) {
        let committed = value_hash(&data);
        // (a) reinterpret the honest tree at a wrong length
        let mut p = prove(&data, &acc);
        p.length = data.len().wrapping_add(k * CHUNK_LEN);
        match verify(&p) { Err(_) => {}, Ok(h) => prop_assert_ne!(h, committed) }
        // (b) strongest constructed graft at a wrong length
        let tail = if boundary { 0 } else { data.len() % CHUNK_LEN };
        let mut lp = k * CHUNK_LEN + tail;
        if lp == data.len() { lp += CHUNK_LEN; }
        let f = forge(&data, lp);
        match verify(&f) { Err(_) => {}, Ok(h) => prop_assert_ne!(h, committed, "graft @ L'={}", lp) }
    }

    #[test]
    fn content_tamper(data in proptest::collection::vec(any::<u8>(), 1..=8 * CHUNK_LEN), idx in 0usize..64) {
        let committed = value_hash(&data);
        let mut d2 = data.clone();
        let i = idx % d2.len();
        d2[i] ^= 0xFF;
        // same length, different content -> different committed hash (blinded content cannot forge)
        prop_assert_ne!(value_hash(&d2), committed);
    }
}

#[test]
fn size_is_o_depth_and_position_independent() {
    println!();
    for mib in [1usize, 16, 64] {
        let n = mib << 20;
        let depth = (n / CHUNK_LEN).next_power_of_two().trailing_zeros() as usize;
        let data = vec![0x5Au8; n];
        for (name, acc) in [
            ("start 2KiB", 0..2048),
            ("middle 2KiB", n / 2..n / 2 + 2048),
            ("end 2KiB", n - 2048..n),
        ] {
            let p = prove(&data, &[acc]);
            assert_eq!(verify(&p).unwrap(), value_hash(&data));
            let (pc, bl) = sizes(&p.data);
            let wire = pc * CHUNK_LEN + bl * 32 + 8;
            println!(
                "{mib:>3}MiB d{depth:>2} {name:<12} present {pc} chunk  blinds {bl:>2}  ~wire {:>5.1} KiB",
                wire as f64 / 1024.0
            );
            assert!(pc <= 3 + 2, "present {pc}");
            assert!(bl <= 2 * depth + 4, "blinds {bl} (depth {depth})");
        }
    }
}
