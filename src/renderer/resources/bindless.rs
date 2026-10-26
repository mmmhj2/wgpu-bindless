use std::{collections::HashMap, num::NonZeroU32};

use crate::{
    asset::asset_types::{
        GpuAssetType,
        texture_asset::{TextureAsset, TextureDimension, TextureFormat},
    },
    renderer::resources::sampler::SamplerDescription,
    util::slot_table::SlotTable,
};

pub const MAX_TEXTURE_SLOTS: usize = 512;
pub const MAX_SAMPLER_SLOTS: usize = 128;

const TEXTURE_BITFIELD_WORDS: usize = MAX_TEXTURE_SLOTS / 64;
const SAMPLER_BITFIELD_WORDS: usize = MAX_SAMPLER_SLOTS / 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SampledTextureHandle(u32);

impl SampledTextureHandle {
    pub fn index(self) -> usize { self.0 as usize }
    fn from_index(index: usize) -> Self { Self(index as u32) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerHandle(u32);

impl SamplerHandle {
    pub fn index(self) -> usize { self.0 as usize }
    fn from_index(index: usize) -> Self { Self(index as u32) }
}

/// Reserved for future bindless SSBO support.
///
/// No storage buffer binding is part of the bindless layout yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StorageBufferHandle {
    _private: (),
}

#[derive(Debug, thiserror::Error)]
pub enum BindlessError {
    #[error("bindless slot table is full")]
    TableFull,
    #[error("requested bindless resource is not allocated")]
    NotAllocated,
}

/// Owns the read-only sampled textures and samplers exposed to shaders.
///
/// Render targets must not be inserted here. Use the post-processing
/// framebuffers (or another pass-owned target) for anything that is written to.
pub struct BindlessResources {
    sampled_textures: SlotTable<wgpu::TextureView, TEXTURE_BITFIELD_WORDS>,
    samplers: SlotTable<wgpu::Sampler, SAMPLER_BITFIELD_WORDS>,
    sampler_by_desc: HashMap<SamplerDescription, SamplerHandle>,

    white_texture: SampledTextureHandle,
    default_bump_texture: SampledTextureHandle,
    default_mrao_texture: SampledTextureHandle,
    default_sampler: SamplerHandle,

    bind_group_layout: wgpu::BindGroupLayout,
}

impl BindlessResources {
    pub const BGL_BINDLESS: [wgpu::BindGroupLayoutEntry; 2] = [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: Some(NonZeroU32::new(MAX_TEXTURE_SLOTS as u32).unwrap()),
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: Some(NonZeroU32::new(MAX_SAMPLER_SLOTS as u32).unwrap()),
        },
    ];

    pub const BGLD_BINDLESS: wgpu::BindGroupLayoutDescriptor<'static> = wgpu::BindGroupLayoutDescriptor {
        label: Some("Bindless resource descriptor set layout"),
        entries: &Self::BGL_BINDLESS,
    };

    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let mut ret = Self {
            sampled_textures: SlotTable::new(),
            samplers: SlotTable::new(),
            sampler_by_desc: HashMap::new(),
            white_texture: SampledTextureHandle(0),
            default_bump_texture: SampledTextureHandle(0),
            default_mrao_texture: SampledTextureHandle(0),
            default_sampler: SamplerHandle(0),
            bind_group_layout: device.create_bind_group_layout(&Self::BGLD_BINDLESS),
        };

        let white_texture = TextureAsset::from_single_color_texel(
            vec![255u8, 255, 255, 255],
            TextureFormat::Rgba8Unorm,
            TextureDimension::D2,
        )
        .upload("White texture", device, queue);
        ret.white_texture = ret
            .create_sampled_texture(white_texture.create_view(&Default::default()))
            .expect("failed to allocate the white bindless texture");

        let default_normal_texture = TextureAsset::from_single_color_texel(
            // Texels are interpreted as signed 8-bit integers.
            vec![128u8, 128, 255, 0],
            TextureFormat::Rgba8Unorm,
            TextureDimension::D2,
        )
        .upload("Default normal texture", device, queue);
        ret.default_bump_texture = ret
            .create_sampled_texture(default_normal_texture.create_view(&Default::default()))
            .expect("failed to allocate the default normal bindless texture");

        let default_mrao_texture = TextureAsset::from_single_color_texel(
            vec![0u8, 0, 0, 255],
            TextureFormat::Rgba8Unorm,
            TextureDimension::D2,
        )
        .upload("Default MRAO texture", device, queue);
        ret.default_mrao_texture = ret
            .create_sampled_texture(default_mrao_texture.create_view(&Default::default()))
            .expect("failed to allocate the default MRAO bindless texture");

        ret.default_sampler = ret
            .create_sampler(device, Default::default())
            .expect("failed to allocate the default bindless sampler");

        ret
    }

    pub fn get_white_texture(&self) -> SampledTextureHandle { self.white_texture }
    pub fn get_default_bump_texture(&self) -> SampledTextureHandle { self.default_bump_texture }
    pub fn get_default_mrao_texture(&self) -> SampledTextureHandle { self.default_mrao_texture }
    pub fn get_default_sampler(&self) -> SamplerHandle { self.default_sampler }

    /// Register a texture that will only ever be sampled.
    ///
    /// Do not pass views belonging to render targets.
    pub fn create_sampled_texture(
        &mut self,
        view: wgpu::TextureView,
    ) -> Result<SampledTextureHandle, BindlessError> {
        let index = self.sampled_textures.insert(view).ok_or(BindlessError::TableFull)?;
        Ok(SampledTextureHandle::from_index(index))
    }

    pub fn free_sampled_texture(&mut self, handle: SampledTextureHandle) {
        self.sampled_textures.remove(handle.index());
    }

    pub fn create_sampler(
        &mut self,
        device: &wgpu::Device,
        desc: SamplerDescription,
    ) -> Result<SamplerHandle, BindlessError> {
        if let Some(handle) = self.sampler_by_desc.get(&desc) {
            return Ok(*handle);
        }

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::from(&desc));
        let index = self.samplers.insert(sampler).ok_or(BindlessError::TableFull)?;
        let handle = SamplerHandle::from_index(index);
        self.sampler_by_desc.insert(desc, handle);
        Ok(handle)
    }

    pub fn free_sampler(&mut self, handle: SamplerHandle) {
        self.samplers.remove(handle.index());
        self.sampler_by_desc.retain(|_, v| *v != handle);
    }

    pub fn sampled_texture(&self, handle: SampledTextureHandle) -> Result<&wgpu::TextureView, BindlessError> {
        self.sampled_textures.get(handle.index()).ok_or(BindlessError::NotAllocated)
    }

    pub fn sampler(&self, handle: SamplerHandle) -> Result<&wgpu::Sampler, BindlessError> {
        self.samplers.get(handle.index()).ok_or(BindlessError::NotAllocated)
    }

    pub fn sampled_texture_count(&self) -> usize { self.sampled_textures.len() }
    pub fn sampler_count(&self) -> usize { self.samplers.len() }

    pub fn get_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Build the bind group for the read-only texture/sampler arrays.
    ///
    /// Freed slots below the high-water mark are filled with the default white
    /// texture / default sampler so existing resource indices stay stable.
    pub fn get_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        assert!(self.sampled_textures.len() > 0);
        assert!(self.samplers.len() > 0);

        let texture_fallback = self
            .sampled_textures
            .get(self.white_texture.index())
            .expect("white bindless texture disappeared");
        let sampler_fallback = self
            .samplers
            .get(self.default_sampler.index())
            .expect("default bindless sampler disappeared");

        let texture_views: Vec<&wgpu::TextureView> = (0..self.sampled_textures.high_water())
            .map(|i| self.sampled_textures.get(i).unwrap_or(texture_fallback))
            .collect();
        let sampler_views: Vec<&wgpu::Sampler> = (0..self.samplers.high_water())
            .map(|i| self.samplers.get(i).unwrap_or(sampler_fallback))
            .collect();

        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureViewArray(&texture_views),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::SamplerArray(&sampler_views),
            },
        ];

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: self.get_bind_group_layout(),
            entries: &entries,
        })
    }

    /// Reserved for future bindless SSBO support.
    ///
    /// This intentionally does not add a storage binding to the bind group
    /// layout yet.
    pub fn create_storage_buffer(&mut self, _buffer: wgpu::Buffer) -> StorageBufferHandle {
        todo!("bindless storage buffers are not implemented yet")
    }
}
