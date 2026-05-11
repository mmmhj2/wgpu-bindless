use std::{collections::{HashMap, hash_map::Entry}, num::NonZeroU32};
use crate::renderer::pipeline::{resource_allocator::LinearResourceAllocator, sampler::SamplerDescription};

pub const MAX_TEXTURE_SLOTS : usize = 512;
pub const MAX_SAMPLER_SLOTS : usize = 128;

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
                label: Some("Dummy texture for bindless"),
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
            }).create_view(&wgpu::TextureViewDescriptor{
                label: Some("Dummy texture view for bindless"), ..Default::default()
            });
        let dummy_sampler  =d.create_sampler(
            &wgpu::SamplerDescriptor{ label: Some("Dummy sampler for bindless"), ..Default::default() }
        );

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
