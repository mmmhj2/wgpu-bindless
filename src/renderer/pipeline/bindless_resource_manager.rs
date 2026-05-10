use std::collections::{HashMap, hash_map::Entry};


/// Wrapper around wgpu::SamplerDescriptor.
/// 
/// This wrapper does not contain any floating point members, allowing
/// it to be stored within a hash map.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerDescription {
    pub address_mode    : [wgpu::AddressMode; 3],
    pub mag_filter      : wgpu::FilterMode,
    pub min_filter      : wgpu::FilterMode,
    pub mipmap_filter   : wgpu::MipmapFilterMode,
    pub compare         : Option<wgpu::CompareFunction>,
    pub anisotropy_clamp: u16,
    pub border_color    : Option<wgpu::SamplerBorderColor>  
}

impl From<&SamplerDescription> for wgpu::SamplerDescriptor<'_> {
    fn from(value: &SamplerDescription) -> Self {
        Self { 
            label: Some("Hashed sampler"),
            address_mode_u: value.address_mode[0],
            address_mode_v: value.address_mode[1],
            address_mode_w: value.address_mode[2],
            mag_filter: value.mag_filter,
            min_filter: value.min_filter,
            mipmap_filter: value.mipmap_filter,
            lod_min_clamp: 0.0f32,
            lod_max_clamp: 32.0f32,
            compare: value.compare,
            anisotropy_clamp: value.anisotropy_clamp,
            border_color: value.border_color
        }
    }
}

pub const MAX_TEXTURE_SLOTS : usize = 512;
pub const MAX_SAMPLER_SLOTS : usize = 128;

#[derive(Debug, Clone, Copy)]
enum LinearResourceAllocatorError {
    BadAllocation,
    OutOfBound,
    NotAllocated
}

struct LinearResourceAllocator<T, const SIZE: usize> {
    occupied    : usize,
    buffer      : [Option<T>; SIZE]
}

impl<T, const SIZE: usize> LinearResourceAllocator<T, SIZE> {
    pub fn new() -> Self {
        Self { occupied: 0, buffer: std::array::from_fn(|_| None) }
    }

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

    pub fn clear(&mut self) -> () {
        self.occupied = 0;
    }
}

pub struct BindlessResourceManager<'a> {
    device              : &'a wgpu::Device,

    textures            : [(); MAX_TEXTURE_SLOTS],

    sampler_allocator   : LinearResourceAllocator<wgpu::Sampler, MAX_SAMPLER_SLOTS>,
    hashed_samplers     : HashMap<SamplerDescription, usize>
}

impl<'a> BindlessResourceManager<'a> {
    pub fn new (d: &'a wgpu::Device) -> Self {
        Self {
            device: d,
            textures: [(); MAX_TEXTURE_SLOTS],
            sampler_allocator: LinearResourceAllocator::new(),
            hashed_samplers: HashMap::new()
        }
    }

    pub fn query_sampler (&self, desc: &SamplerDescription) -> Option<&wgpu::Sampler> {
        let idx = self.hashed_samplers.get(desc)?;
        if let Ok(s) = self.sampler_allocator.get(*idx) {
            Some(s)
        } else {
            None
        }
    }

    pub fn get_sampler (&mut self, desc: SamplerDescription) -> &wgpu::Sampler {
        match self.hashed_samplers.entry(desc) {
            Entry::Occupied(s) => {
                self.sampler_allocator.get(*s.get()).expect("Stored sampler disappeared.")
            },
            Entry::Vacant(s) => {
                let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor::from(&desc));
                let idx = self.sampler_allocator.push_back(sampler).expect("Failed to allocate new sampler");
                s.insert(idx);
                self.sampler_allocator.get(idx).expect("Stored sampler disappeared.")
            }
        }
    }
}
