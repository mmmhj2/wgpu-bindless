use super::lowest_integer::{LowestIntegerCache, LIC_BIT_PER_DWORD};

/// A slot-indexed resource table backed by [`LowestIntegerCache`].
///
/// Slot indices are stable across removals. Freed slots are reused by the next
/// allocation, which keeps resource indices dense for typical workloads while
/// still allowing individual entries to be released.
pub(crate) struct SlotTable<T, const BITFIELD_WORDS: usize> {
    slots: Vec<Option<T>>,
    free_list: LowestIntegerCache<BITFIELD_WORDS>,
    high_water: usize,
}

impl<T, const BITFIELD_WORDS: usize> SlotTable<T, BITFIELD_WORDS> {
    pub fn new() -> Self {
        let capacity = BITFIELD_WORDS * LIC_BIT_PER_DWORD;
        Self {
            slots: (0..capacity).map(|_| None).collect(),
            free_list: LowestIntegerCache::new(),
            high_water: 0
        }
    }

    /// Insert a value and return its stable slot index.
    ///
    /// Returns `None` when the table is full.
    pub fn insert(&mut self, value: T) -> Option<usize> {
        let index = self.free_list.allocate()?;
        self.slots[index] = Some(value);
        self.high_water = self.high_water.max(index + 1);
        Some(index)
    }

    /// Remove the value stored at `index`, releasing the slot for reuse.
    pub fn remove(&mut self, index: usize) -> Option<T> {
        let value = self.slots.get_mut(index)?.take();
        if value.is_some() {
            self.free_list.free(index);

            while self.high_water > 0 && !self.free_list.is_allocated(self.high_water - 1) {
                self.high_water -= 1;
            }
        }
        value
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.slots.get(index)?.as_ref()
    }

    /// Highest slot index ever touched by an allocation, plus one.
    ///
    /// This is the length needed to build a dense GPU array without dropping
    /// stable indices. Slots below it may be `None` after a removal.
    pub fn high_water(&self) -> usize { self.high_water }

    pub fn len(&self) -> usize { self.free_list.allocated_count() }

    #[cfg(test)]
    pub fn capacity(&self) -> usize { self.free_list.capacity() }
}

impl<T, const BITFIELD_WORDS: usize> Default for SlotTable<T, BITFIELD_WORDS> {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn reuses_lowest_freed_slot() {
        let mut table: SlotTable<u32, 1> = SlotTable::new();

        assert_eq!(table.insert(10), Some(0));
        assert_eq!(table.insert(20), Some(1));
        assert_eq!(table.insert(30), Some(2));

        assert_eq!(table.remove(1), Some(20));
        assert_eq!(table.insert(40), Some(1));
        assert_eq!(table.get(1), Some(&40));
        assert_eq!(table.len(), 3);
    }

    #[test]
    fn high_water_tracks_tail_removals() {
        let mut table: SlotTable<u32, 1> = SlotTable::new();

        table.insert(10);
        table.insert(20);
        table.insert(30);
        assert_eq!(table.high_water(), 3);

        table.remove(2);
        assert_eq!(table.high_water(), 2);
        table.remove(1);
        assert_eq!(table.high_water(), 1);
        table.remove(0);
        assert_eq!(table.high_water(), 0);
    }

    #[test]
    fn holes_below_high_water_preserve_indices() {
        let mut table: SlotTable<u32, 1> = SlotTable::new();

        table.insert(10);
        table.insert(20);
        table.insert(30);
        table.remove(1);

        assert_eq!(table.high_water(), 3);
        assert_eq!(table.get(0), Some(&10));
        assert_eq!(table.get(1), None);
        assert_eq!(table.get(2), Some(&30));
    }

    #[test]
    fn capacity_is_bitfield_words_times_dword_bits() {
        let table: SlotTable<u32, 2> = SlotTable::new();
        assert_eq!(table.capacity(), 2 * LIC_BIT_PER_DWORD);
    }
}

