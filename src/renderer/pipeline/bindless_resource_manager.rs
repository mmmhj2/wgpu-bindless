use std::{collections::{HashMap, hash_map::Entry}, num::NonZeroU32};
use wgpu::TexelCopyBufferInfo;

use crate::renderer::{device_interface::DeviceInterface, pipeline::{resource_allocator::LinearResourceAllocator, sampler::SamplerDescription}};

use super::resource_allocator::LinearResourceAllocatorError;

pub const MAX_TEXTURE_SLOTS : usize = 512;
pub const MAX_SAMPLER_SLOTS : usize = 128;

pub struct BindlessResourceManager {
    bind_group_layout       : wgpu::BindGroupLayout,
    texture_view_allocator  : LinearResourceAllocator<wgpu::TextureView, MAX_TEXTURE_SLOTS>,
    sampler_allocator       : LinearResourceAllocator<wgpu::Sampler, MAX_SAMPLER_SLOTS>,
    hashed_samplers         : HashMap<SamplerDescription, usize>,

    default_sampler_idx     : usize,
    white_txv_idx           : usize,
    default_bump_txv_idx    : usize
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

    pub fn new (d: &DeviceInterface) -> Self {
        let mut ret = Self {
            bind_group_layout: d.get_device().create_bind_group_layout(&Self::BGLD_BINDLESS),
            texture_view_allocator: LinearResourceAllocator::new(),
            sampler_allocator: LinearResourceAllocator::new(),
            hashed_samplers: HashMap::new(),
            default_sampler_idx: 0,
            white_txv_idx: 0,
            default_bump_txv_idx: 0
        };

        ret.default_sampler_idx = ret.push_sampler(d.get_device(), Default::default()).expect("Failed to create default sampler.");

        let packed_layout = wgpu::TexelCopyBufferLayout{
            offset: 0,
            bytes_per_row: None,
            rows_per_image: None
        };
        let one_extent = wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1
        };

        let white_texture = d.get_device().create_texture(&wgpu::TextureDescriptor {
            label: Some("White texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        d.get_queue().write_texture(
            wgpu::TexelCopyTextureInfo{
                aspect: wgpu::TextureAspect::All,
                mip_level: 0,
                texture: &white_texture,
                origin: wgpu::Origin3d{x: 0, y: 0, z: 0}
            },
            &[255, 255, 255, 255],
            packed_layout.clone(),
            one_extent.clone()
        );
        ret.white_txv_idx = ret.push_texture(white_texture.create_view(&Default::default())).expect("Failed to create white texture");

        let default_normal_texture = d.get_device().create_texture(&wgpu::TextureDescriptor {
            label: Some("Default normal texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        d.get_queue().write_texture(
            wgpu::TexelCopyTextureInfo{
                aspect: wgpu::TextureAspect::All,
                mip_level: 0,
                texture: &default_normal_texture,
                origin: wgpu::Origin3d{x: 0, y: 0, z: 0}
            },
            &[127, 127, 127, 0],
            packed_layout.clone(),
            one_extent.clone()
        );
        ret.default_bump_txv_idx = ret.push_texture(default_normal_texture.create_view(&Default::default())).expect("Failed to create default normal texture");

        return ret;
    }

    pub fn get_default_sampler(&self) -> usize { self.default_sampler_idx }
    pub fn get_white_texture(&self) -> usize { self.white_txv_idx }
    pub fn get_default_bump_texture(&self) -> usize { self.default_bump_txv_idx }

    /// Push a sampler into the manager, returning its index.
    ///
    /// Descriptions are hashed and stored into a Hash Map.
    /// Samplers with the same description will therefore be reused and shares
    /// the same index.
    pub fn push_sampler(&mut self, d: &wgpu::Device, desc: SamplerDescription) -> Result<usize, LinearResourceAllocatorError> {
        match self.hashed_samplers.entry(desc) {
            Entry::Occupied(s) => {
                Ok(*s.get())
            },
            Entry::Vacant(s) => {
                let sampler = d.create_sampler(&wgpu::SamplerDescriptor::from(&desc));
                let idx = self.sampler_allocator.push_back(sampler)?;
                s.insert(idx);
                Ok(idx)
            }
        }
    }

    /// Push a texture into the manager, returning its index.
    pub fn push_texture(&mut self, t: wgpu::TextureView) -> Result<usize, LinearResourceAllocatorError> {
        self.texture_view_allocator.push_back(t)
    }

    /// Get the bind group layout associated with the current bindless resource manager.
    pub fn get_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get the bind group representing all current registered bindless resources.
    /// 
    /// A new bind group will be created on the device specified when creating the manager.
    pub fn get_bind_group(&self, d: &wgpu::Device) -> wgpu::BindGroup {

        assert!(self.sampler_allocator.count() > 0);
        assert!(self.texture_view_allocator.count() > 0);

        let sampler_array = Vec::from_iter(
                self.sampler_allocator.get_allocated_slice().iter().map(
                    |x| x.as_ref().expect("Allocated sampler disappeared.")
                )
            );
        let texture_view_array = Vec::from_iter(
                self.texture_view_allocator.get_allocated_slice().iter().map(
                    |x| x.as_ref().expect("Allocated texture view disappeared.")
                )
            );

        let entries= &[
            wgpu::BindGroupEntry{
                binding: 0,
                resource: wgpu::BindingResource::TextureViewArray(&texture_view_array)
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
