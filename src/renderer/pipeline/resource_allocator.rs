
#[derive(Debug, Clone, Copy)]
pub enum LinearResourceAllocatorError {
    BadAllocation,
    OutOfBound,
    NotAllocated
}

pub struct LinearResourceAllocator<T, const SIZE: usize> {
    occupied    : usize,
    buffer      : [Option<T>; SIZE]
}

impl<T, const SIZE: usize> LinearResourceAllocator<T, SIZE> {
    pub fn new() -> Self {
        Self { occupied: 0, buffer: std::array::from_fn(|_| None) }
    }

    pub fn count(&self) -> usize { self.occupied }

    pub fn get(&self, idx: usize) -> Result<&T, LinearResourceAllocatorError> {
        if idx >= SIZE {
            Err(LinearResourceAllocatorError::OutOfBound)
        } else {
            self.buffer[idx].as_ref().ok_or(LinearResourceAllocatorError::NotAllocated)
        }
    }

    pub fn push_back(&mut self, v: T) -> Result<usize, LinearResourceAllocatorError> {
        if self.occupied >= SIZE {
            Err(LinearResourceAllocatorError::BadAllocation)
        } else {
            let old_idx = self.occupied;
            self.occupied += 1;
            self.buffer[old_idx] = Some(v);
            Ok(old_idx)
        }
    }

    pub fn get_allocated_slice(&self) -> &[Option<T>] {
        &self.buffer[0..self.occupied]
    }

    pub fn clear(&mut self) -> () {
        self.occupied = 0;
    }
}
