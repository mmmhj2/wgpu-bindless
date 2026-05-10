use std::{collections::{HashMap, hash_map::Entry}, num::NonZeroU32};


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

pub struct BindlessResourceManager {
    bind_group_layout   : wgpu::BindGroupLayout,

    textures            : [(); MAX_TEXTURE_SLOTS],

    sampler_allocator   : LinearResourceAllocator<wgpu::Sampler, MAX_SAMPLER_SLOTS>,
    hashed_samplers     : HashMap<SamplerDescription, usize>,

    /// WGPU disallows zero sized bind groups, so we need a dummy texture and dummy sampler.
    dummy_texture_view  : wgpu::TextureView,
    dummy_sampler       : wgpu::Sampler
}

impl BindlessResourceManager {

    pub const BGL_BINDLESS: [wgpu::BindGroupLayoutEntry; 2] = [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false
            },
            count: Some(NonZeroU32::new(MAX_TEXTURE_SLOTS as u32).unwrap())
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: Some(NonZeroU32::new(MAX_SAMPLER_SLOTS as u32).unwrap()) 
        }
    ];

    pub const BGLD_BINDLESS: wgpu::BindGroupLayoutDescriptor<'static> = wgpu::BindGroupLayoutDescriptor{
        label: Some("Bindless resource descriptor set layout"),
        entries: &Self::BGL_BINDLESS
    };

    pub fn new (d: &wgpu::Device) -> Self {

        let dummy_texture_view = d.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }).create_view(&Default::default());
        let dummy_sampler  =d.create_sampler(&Default::default());

        Self {
            bind_group_layout: d.create_bind_group_layout(&Self::BGLD_BINDLESS),
            textures: [(); MAX_TEXTURE_SLOTS],
            sampler_allocator: LinearResourceAllocator::new(),
            hashed_samplers: HashMap::new(),
            dummy_texture_view,
            dummy_sampler
        }
    }

    /// Query the index of a sampler in the bindless resource manager.
    pub fn query_sampler (&self, desc: &SamplerDescription) -> Option<usize> {
        self.hashed_samplers.get(desc).copied()
    }

    /// Obtain the index of a given sampler in the manager.
    /// If the sampler does not exist, a new one will be created.
    pub fn get_sampler (&mut self, d: &wgpu::Device, desc: SamplerDescription) -> usize {
        match self.hashed_samplers.entry(desc) {
            Entry::Occupied(s) => {
                *s.get()
            },
            Entry::Vacant(s) => {
                let sampler = d.create_sampler(&wgpu::SamplerDescriptor::from(&desc));
                let idx = self.sampler_allocator.push_back(sampler).expect("Failed to allocate new sampler");
                s.insert(idx);
                idx
            }
        }
    }

    /// Get the bind group layout associated with the current bindless resource manager.
    pub fn get_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get the bind group representing all current registered bindless resources.
    /// 
    /// A new bind group will be created on the device specified when creating the manager.
    pub fn get_bind_group(&self, d: &wgpu::Device) -> wgpu::BindGroup {
        let sampler_array = if self.sampler_allocator.count() > 0 { 
            Vec::from_iter(
                self.sampler_allocator.get_allocated_slice().iter().map(
                    |x| x.as_ref().expect("Allocated sampler disappeared.")
                )
            )
        } else {
            vec![&self.dummy_sampler]
        };

        let entries= &[
            wgpu::BindGroupEntry{
                binding: 0,
                resource: wgpu::BindingResource::TextureViewArray(&[&self.dummy_texture_view])
            },
            wgpu::BindGroupEntry{
                binding: 1,
                resource: wgpu::BindingResource::SamplerArray(
                    &sampler_array
                )
            }
        ];
        d.create_bind_group(
            &wgpu::BindGroupDescriptor{
                label: None,
                layout: self.get_bind_group_layout(),
                entries
            }
        )
    }
}
