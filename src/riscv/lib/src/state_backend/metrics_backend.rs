use std::cell::RefCell;

use crate::state_backend::ManagerAlloc;
use crate::state_backend::ManagerBase;
use crate::state_backend::ManagerRead;
use crate::state_backend::ManagerReadWrite;
use crate::state_backend::ManagerWrite;
use crate::state_backend::metrics_backend::merkle_tracker::MerkleTracker;
use crate::state_backend::owned_backend::Owned;

mod merkle_tracker;

pub struct Metrics;

thread_local! {
    /// Static variable used by [`block_metrics`] as a place to record metrics.
    static MERKLE_METRICS: RefCell<MerkleTracker> = Default::default();
}

fn log_read(path: &Vec<u8>) {
    MERKLE_METRICS.with_borrow_mut(|m| m.log_read(path));
}

fn log_write(path: &Vec<u8>) {
    MERKLE_METRICS.with_borrow_mut(|m| m.log_write(path));
}

pub struct MeteredRegion<E: 'static, const LEN: usize> {
    region: <Owned as ManagerBase>::Region<E, LEN>,
    merkle_path: Vec<u8>,
}

pub struct MeteredDynRegion<const LEN: usize> {
    region: <Owned as ManagerBase>::DynRegion<LEN>,
    merkle_path: Vec<u8>,
}

pub struct MeteredEnrichedCell<V: super::EnrichedValue> {
    underlying: MeteredRegion<V::E, 1>,
}

impl ManagerBase for Metrics {
    type Region<E: 'static, const LEN: usize> = MeteredRegion<E, LEN>;

    type DynRegion<const LEN: usize> = MeteredDynRegion<LEN>;

    type EnrichedCell<V: super::EnrichedValue> = MeteredEnrichedCell<V>;

    type ManagerRoot = Self;

    fn enrich_cell<V: super::EnrichedValueLinked>(
        underlying: Self::Region<V::E, 1>,
    ) -> Self::EnrichedCell<V> {
        Self::EnrichedCell { underlying }
    }

    fn as_devalued_cell<V: super::EnrichedValue>(
        cell: &Self::EnrichedCell<V>,
    ) -> &Self::Region<V::E, 1> {
        &cell.underlying
    }
}

impl ManagerRead for Metrics {
    fn region_read<E: Copy, const LEN: usize>(region: &Self::Region<E, LEN>, index: usize) -> E {
        log_read(&region.merkle_path);
        Owned::region_read(&region.region, index)
    }

    fn region_ref<E: 'static, const LEN: usize>(region: &Self::Region<E, LEN>, index: usize) -> &E {
        log_read(&region.merkle_path);
        Owned::region_ref(&region.region, index)
    }

    fn region_read_all<E: Copy, const LEN: usize>(region: &Self::Region<E, LEN>) -> Vec<E> {
        log_read(&region.merkle_path);
        Owned::region_read_all(&region.region)
    }

    fn dyn_region_read<E: super::Elem, const LEN: usize>(
        region: &Self::DynRegion<LEN>,
        address: usize,
    ) -> E {
        todo!("Compute path through dyn region")
    }

    fn dyn_region_read_all<E: super::Elem, const LEN: usize>(
        region: &Self::DynRegion<LEN>,
        address: usize,
        values: &mut [E],
    ) {
        todo!("Compute path through dyn region")
    }

    fn enriched_cell_read_stored<V>(cell: &Self::EnrichedCell<V>) -> V::E
    where
        V: super::EnrichedValue,
        V::E: Copy,
    {
        log_read(&cell.underlying.merkle_path);
        Owned::region_read(&cell.underlying.region, 0)
    }

    fn enriched_cell_read_derived<V>(cell: &Self::EnrichedCell<V>) -> V::D
    where
        V: super::EnrichedValueLinked,
        V::D: Copy,
    {
        V::derive(Self::enriched_cell_ref_stored(cell))
    }

    fn enriched_cell_ref_stored<V>(cell: &Self::EnrichedCell<V>) -> &V::E
    where
        V: super::EnrichedValue,
    {
        Self::region_ref(&cell.underlying, 0)
    }
}

impl ManagerWrite for Metrics {
    fn region_write<E: 'static, const LEN: usize>(
        region: &mut Self::Region<E, LEN>,
        index: usize,
        value: E,
    ) {
        log_write(&region.merkle_path);
        Owned::region_write(&mut region.region, index, value);
    }

    fn region_write_all<E: Copy, const LEN: usize>(region: &mut Self::Region<E, LEN>, value: &[E]) {
        log_write(&region.merkle_path);
        Owned::region_write_all(&mut region.region, value);
    }

    fn dyn_region_write<E: super::Elem, const LEN: usize>(
        region: &mut Self::DynRegion<LEN>,
        address: usize,
        value: E,
    ) {
        todo!("Compute path through dyn region")
    }

    fn dyn_region_write_all<E: super::Elem + Copy, const LEN: usize>(
        region: &mut Self::DynRegion<LEN>,
        address: usize,
        values: &[E],
    ) {
        todo!("Compute path through dyn region")
    }

    fn enriched_cell_write<V>(cell: &mut Self::EnrichedCell<V>, value: V::E)
    where
        V: super::EnrichedValueLinked,
    {
        Self::region_write(&mut cell.underlying, 0, value);
    }
}

impl ManagerReadWrite for Metrics {
    fn region_replace<E: Copy, const LEN: usize>(
        region: &mut Self::Region<E, LEN>,
        index: usize,
        value: E,
    ) -> E {
        let old = Self::region_read(region, index);
        Self::region_write(region, index, value);
        old
    }
}

impl ManagerAlloc for Metrics {
    fn allocate_region<E, const LEN: usize>(init_value: [E; LEN]) -> Self::Region<E, LEN> {
        Self::Region {
            region: Owned::allocate_region(init_value),
            merkle_path: vec![],
        }
    }

    fn allocate_dyn_region<const LEN: usize>() -> Self::DynRegion<LEN> {
        Self::DynRegion {
            region: Owned::allocate_dyn_region::<LEN>(),
            merkle_path: vec![],
        }
    }
}
