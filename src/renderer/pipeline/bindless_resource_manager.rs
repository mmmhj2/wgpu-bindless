use std::num::NonZeroU32;
use wgpu::TexelCopyBufferInfo;

use crate::{
    asset::asset_types::{
        GpuAssetType,
        texture_asset::{TextureAsset, TextureDimension, TextureFormat}
    },
    renderer::{
        device_interface::DeviceInterface,
        pipeline::sampler_manager::{SamplerManager, MAX_SAMPLER_SLOTS}
    },
    util::resource_allocator::{
        LinearResourceAllocator, LinearResourceAllocatorError
    }
};

pub const MAX_TEXTURE_SLOTS : usize = 512;

pub struct BindlessResourceManager {
    bind_group_layout       : wgpu::BindGroupLayout,
    texture_view_allocator  : LinearResourceAllocator<wgpu::TextureView, MAX_TEXTURE_SLOTS>,
    samplers                : SamplerManager,

    white_txv_idx           : usize,
    default_bump_txv_idx    : usize,
    default_mrao_txv_idx    : usize
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
            samplers: SamplerManager::new(d.get_device()),
            white_txv_idx: 0,
            default_bump_txv_idx: 0,
            default_mrao_txv_idx: 0
        };

        let white_texture = TextureAsset::from_single_color_texel(
            vec![255u8, 255, 255, 255],
            TextureFormat::Rgba8Unorm,
            TextureDimension::D2
        ).upload("White texture", d.get_device(), d.get_queue());
        ret.white_txv_idx = ret.push_texture(
            white_texture.create_view(&Default::default())
        ).expect("Failed to create white texture");

        let default_normal_texture = TextureAsset::from_single_color_texel(
            // Texels are interpreted as signed 8-bit integers.
            vec![128u8, 128, 255, 0],
            TextureFormat::Rgba8Unorm,
            TextureDimension::D2
        ).upload("Default normal texture", d.get_device(), d.get_queue());
        ret.default_bump_txv_idx = ret.push_texture(
            default_normal_texture.create_view(&Default::default())
        ).expect("Failed to create default normal texture");

        let default_mrao_texture = TextureAsset::from_single_color_texel(
            vec![0u8, 0, 0, 255],
            TextureFormat::Rgba8Unorm,
            TextureDimension::D2
        ).upload("Default MRAO texture", d.get_device(), d.get_queue());
        ret.default_mrao_txv_idx = ret.push_texture(
            default_mrao_texture.create_view(&Default::default())
        ).expect("Failed to create default MRAO texture");

        return ret;
    }

    /// Acquire the sampler manager owning all samplers registered with this
    /// bindless resource manager.
    pub fn get_samplers(&self) -> &SamplerManager { &self.samplers }
    pub fn get_samplers_mut(&mut self) -> &mut SamplerManager { &mut self.samplers }

    pub fn get_white_texture(&self) -> usize { self.white_txv_idx }

    /// Acquire the default normal map texture.
    /// 
    /// This texture has three channels of data, representing tangent-space normals
    /// under the following specification:
    /// - R: tangent space, X+, \[0  , 1\]
    /// - G: tangent space, Y+, \[0  , 1\]
    /// - B: tangent space, Z+, \]0.5, 1\]
    /// 
    /// The default normal vector is (0.5, 0.5, 1.0), pointing up.
    /// It has the same representation as GLTF spec, meaning that you have to
    /// manually renormalize the normal vector in the shader code.
    pub fn get_default_bump_texture(&self) -> usize { self.default_bump_txv_idx }

    /// Acquire the default MRAO texture.
    /// 
    /// This texture has three channels of valid data in range [0, 1]:
    /// - R: Ambient occlusion = 0;
    /// - G: Metallic = 0;
    /// - B: Roughness = 0.
    /// 
    /// It therefore shares the same layout as GLTF spec.
    pub fn get_default_mrao_texture(&self) -> usize { self.default_mrao_txv_idx }

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

        assert!(self.samplers.count() > 0);
        assert!(self.texture_view_allocator.count() > 0);

        let sampler_array = self.samplers.get_allocated_samplers();
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
