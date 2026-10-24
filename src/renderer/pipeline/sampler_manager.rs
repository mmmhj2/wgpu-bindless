use std::collections::{HashMap, hash_map::Entry};

use crate::{
    renderer::pipeline::sampler::SamplerDescription,
    util::resource_allocator::{LinearResourceAllocator, LinearResourceAllocatorError},
};

pub const MAX_SAMPLER_SLOTS : usize = 128;

/// Owns the GPU samplers used by the bindless system.
///
/// Samplers are kept separate from the texture resources: this manager
/// allocates sampler slots, deduplicates them by description, and exposes the
/// array needed to build the bindless bind group.
pub struct SamplerManager {
    sampler_allocator   : LinearResourceAllocator<wgpu::Sampler, MAX_SAMPLER_SLOTS>,
    hashed_samplers     : HashMap<SamplerDescription, usize>,
    default_sampler_idx : usize,
}

impl SamplerManager {

    /// Create a sampler manager, registering the default sampler at index 0.
    pub fn new(device: &wgpu::Device) -> Self {
        let mut ret = Self {
            sampler_allocator: LinearResourceAllocator::new(),
            hashed_samplers: HashMap::new(),
            default_sampler_idx: 0,
        };
        ret.default_sampler_idx = ret.push_sampler(device, Default::default())
            .expect("Failed to create default sampler.").0;
        ret
    }

    pub fn get_default_sampler_id(&self) -> usize { self.default_sampler_idx }

    /// Get the default sampler along with its index.
    /// 
    /// Panic
    /// ---
    /// Default sampler is assumed to be always present. If it is not, the
    /// invariant is broken and the program will panic.
    pub fn get_default_sampler(&self) -> (usize, &wgpu::Sampler) {
        let idx = self.default_sampler_idx;
        (idx, self.get_sampler(idx).expect("default sampler should always present"))
    }

    /// Get the sampler stored at the given index.
    pub fn get_sampler(&self, idx: usize) -> Result<&wgpu::Sampler, LinearResourceAllocatorError> {
        self.sampler_allocator.get(idx)
    }

    /// Number of samplers currently registered.
    pub fn count(&self) -> usize { self.sampler_allocator.count() }

    /// Push a sampler into the manager, returning its index and the sampler itself.
    ///
    /// Descriptions are hashed and stored into a Hash Map.
    /// Samplers with the same description will therefore be reused and share
    /// the same index.
    pub fn push_sampler(&mut self, d: &wgpu::Device, desc: SamplerDescription) -> Result<(usize, &wgpu::Sampler), LinearResourceAllocatorError> {
        match self.hashed_samplers.entry(desc) {
            Entry::Occupied(s) => {
                Ok((*s.get(), self.sampler_allocator.get(*s.get()).expect("Allocated sampler disappeared.")))
            },
            Entry::Vacant(s) => {
                let sampler = d.create_sampler(&wgpu::SamplerDescriptor::from(&desc));
                let idx = self.sampler_allocator.push_back(sampler)?;
                s.insert(idx);
                Ok((idx, self.sampler_allocator.get(idx).expect("Allocated sampler disappeared.")))
            }
        }
    }

    /// Collect the currently allocated samplers for bind group construction.
    pub(crate) fn get_allocated_samplers(&self) -> Vec<&wgpu::Sampler> {
        Vec::from_iter(
            self.sampler_allocator.get_allocated_slice().iter().map(
                |x| x.as_ref().expect("Allocated sampler disappeared.")
            )
        )
    }
}
