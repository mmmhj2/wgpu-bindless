use cgmath::num_traits::Euclid;

pub(crate) const LIC_BIT_PER_DWORD: usize = 64;

/// A lowest integer cache using the same idea as Linux file descriptor allocation.
/// 
/// Implemented via a bitset, it has a linear complexity with low constant overhead, 
/// and is efficient for low capacity (< 1000) usages.
/// 
/// **Note:** Due to Rust disallowing const expressions, the `BITFIELD_CAPACITY`
/// controls the size of the underlying bitfield of the cache, rather than the
/// actual integer capacity of the cache. Actual capacity can be caculated by
/// *multiplying it by `LIC_BIT_PER_DWORD` (64)*.
#[derive(Debug)]
pub(crate) struct LowestIntegerCache <const BITFIELD_CAPACITY: usize> {
    watermark   : usize,
    // Maybe we can use BitSet instead.
    bitfield    : [u64; BITFIELD_CAPACITY]
}

impl <const CAPACITY: usize> LowestIntegerCache<CAPACITY> {
    pub fn new() -> Self {
        Self {
            watermark: 0,
            bitfield: [0; CAPACITY]
        }
    }

    /// Get the lowest available unallocated integer in the cache.
    /// Returns `None` if the cache is full and an allocation cannot be performed.
    pub fn get_lowest_integer(& self) -> Option<usize> {
        for idx in (self.watermark/LIC_BIT_PER_DWORD)..(CAPACITY) {
            let v = self.bitfield[idx];
            if !v == 0 {
                continue;
            }
            if v == 0 {
                return Some(idx * LIC_BIT_PER_DWORD);
            }

            let lowest_zero = (!v) & (!v).wrapping_neg();
            return Some(lowest_zero.trailing_zeros() as usize + idx * LIC_BIT_PER_DWORD);
        }
        return None;
    }

    /// Allocate from the cache.
    /// Returns `None` if the cache is full and an allocation cannot be performed.
    pub fn allocate(&mut self) -> Option<usize> {
        let ret = self.get_lowest_integer();
        
        if let Some(v) = ret {
            self.watermark = v + 1;
            let (q, r) = v.div_rem_euclid(&LIC_BIT_PER_DWORD);
            self.bitfield[q] |= (1 << r);
        }
        ret
    }

    /// Free the integer.
    /// Does not check whether the returned value is actually allocated.
    pub fn free(&mut self, x: usize) {
        assert!(x < CAPACITY * LIC_BIT_PER_DWORD);

        self.watermark = std::cmp::min(x, self.watermark);
        let (q, r) = x.div_rem_euclid(&LIC_BIT_PER_DWORD);
        self.bitfield[q] &= !(1 << r);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    const TEST_CAPACITY: usize = 4;
    #[test]
    fn test_lowest_integer_cache_uniform() {
        let mut t = LowestIntegerCache::<TEST_CAPACITY>::new();
        
        for i in 0..(TEST_CAPACITY*LIC_BIT_PER_DWORD) {
            assert_eq!(t.get_lowest_integer().unwrap(), i);
            t.allocate();
        }
        assert_eq!(t.get_lowest_integer(), None);

        for i in (0..(TEST_CAPACITY*LIC_BIT_PER_DWORD)).rev() {
            t.free(i);
            assert_eq!(t.get_lowest_integer(), Some(i));
        }
    }

    #[test]
    fn test_lowest_integer_cache_sparse() {
        let mut t = LowestIntegerCache::<TEST_CAPACITY>::new();
        for _ in 0..(TEST_CAPACITY*LIC_BIT_PER_DWORD) {
            t.allocate();
        }

        for i in (0..20).rev() {
            t.free(i * 10);
            assert_eq!(t.get_lowest_integer(), Some(i * 10));
        }
    }
}
