use std::num::NonZero;

use serde::{Deserialize, Serialize};

use crate::asset::asset_types::{ConcreteAssetType, GpuAssetType};


#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    Rgba8Unorm,
    Rgba8Srgb,
    Rgba16Unorm,
    Rgba32Float,
    R8Unorm,
    Rg8Unorm,
    R16Unorm,
    Rg16Unorm
}

impl Into<wgpu::TextureFormat> for TextureFormat {
    fn into(self) -> wgpu::TextureFormat {
        match self {
            TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8Srgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Rgba16Unorm => wgpu::TextureFormat::Rgba16Unorm,
            TextureFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
            TextureFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
            TextureFormat::Rg8Unorm => wgpu::TextureFormat::Rg8Unorm,
            TextureFormat::R16Unorm => wgpu::TextureFormat::R16Unorm,
            TextureFormat::Rg16Unorm => wgpu::TextureFormat::Rg16Unorm,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum AddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
    ClampToBorder
}

impl Into<wgpu::AddressMode> for AddressMode {
    fn into(self) -> wgpu::AddressMode {
        match self {
            AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            AddressMode::Repeat => wgpu::AddressMode::Repeat,
            AddressMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
            AddressMode::ClampToBorder => todo!("Clamp to border is currently unsupported."),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum FilterMode {
    Nearest,
    Linear
}

impl Into<wgpu::FilterMode> for FilterMode {
    fn into(self) -> wgpu::FilterMode {
        match self {
            FilterMode::Nearest => wgpu::FilterMode::Nearest,
            FilterMode::Linear => wgpu::FilterMode::Linear,
        }
    }
}

impl Into<wgpu::MipmapFilterMode> for FilterMode {
    fn into(self) -> wgpu::MipmapFilterMode {
        match self {
            FilterMode::Nearest => wgpu::MipmapFilterMode::Nearest,
            FilterMode::Linear => wgpu::MipmapFilterMode::Linear,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Sampler {
    /// Address mode for u,v,w coordinates
    pub address_mode: (AddressMode, AddressMode, AddressMode),
    /// Filter mode for magnification, minification and mipmap
    pub filter      : (FilterMode, FilterMode, FilterMode)
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            address_mode: (AddressMode::ClampToEdge, AddressMode::ClampToEdge, AddressMode::ClampToEdge),
            filter: (FilterMode::Nearest, FilterMode::Nearest, FilterMode::Nearest)
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum TextureDimension {
    D1,
    D2,
    D2Array,
    D3
}

impl Into<wgpu::TextureDimension> for TextureDimension {
    fn into(self) -> wgpu::TextureDimension {
        match self {
            TextureDimension::D1 => wgpu::TextureDimension::D1,
            TextureDimension::D2 => wgpu::TextureDimension::D2,
            TextureDimension::D2Array => wgpu::TextureDimension::D2,
            TextureDimension::D3 => wgpu::TextureDimension::D3,
        }
    }
}

/// serde_bytes does not support array of arrays of u8, this struct works around it.
#[derive(Serialize, Deserialize, Debug)]
pub struct MipmapLayerData {
    #[serde(with = "serde_bytes")]
    texels: Box<[u8]>
}

#[derive(Serialize, Deserialize, Debug)]
pub enum TexelData {
    #[serde(with = "serde_bytes")]
    Plain(Box<[u8]>),
    /// For mipmapped textures with block compression, the data is assumed to
    /// be *correctly padded*. Which means that each mipmap level should contain
    /// at least one block of texels.
    Mipmapped(Box<[MipmapLayerData]>)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TextureAsset {
    pub format: TextureFormat,
    pub dimension: TextureDimension,
    pub width: NonZero<u32>,
    pub height: Option<NonZero<u32>>,
    pub depth_or_array_slice: Option<NonZero<u32>>,
    pub sampler: Sampler,
    pub texels: TexelData,
}

impl TextureAsset {
    fn determine_upload_data_layout(format: wgpu::TextureFormat, size: wgpu::Extent3d) -> (u32, u32) {
        let block_copy_size = format.block_copy_size(None).expect("designated format should have a block copy size");
        let block_dimensions = format.block_dimensions();

        let bytes_per_row = block_copy_size * size.width.div_ceil(block_dimensions.0);
        let rows_per_image = size.height.div_ceil(block_dimensions.1);
        return (bytes_per_row, rows_per_image)
    }

    /// Build a texture asset that holds a single texel.
    ///
    /// This is useful for providing default fallback textures (white, flat normal, etc.).
    /// The sampler defaults to `Sampler::default()`.
    pub fn from_single_color_texel(
        texel: impl Into<Box<[u8]>>,
        format: TextureFormat,
        dimension: TextureDimension
    ) -> Self {
        let one = NonZero::new(1u32).unwrap();
        let (height, depth_or_array_slice) = match dimension {
            TextureDimension::D1 => (None, None),
            TextureDimension::D2 => (Some(one), None),
            TextureDimension::D2Array => (Some(one), Some(one)),
            TextureDimension::D3 => (Some(one), Some(one)),
        };

        Self {
            format,
            dimension,
            width: one,
            height,
            depth_or_array_slice,
            sampler: Sampler::default(),
            texels: TexelData::Plain(texel.into()),
        }
    }
}

impl ConcreteAssetType for TextureAsset {
    const SCHEMA_VERSION: u32 = 1;
}

impl GpuAssetType for TextureAsset {
    type GpuObjectType = wgpu::Texture;

    /// Upload the texture asset onto GPU device.
    fn upload(&self, name: &str, device: &wgpu::Device, queue: &wgpu::Queue) -> Self::GpuObjectType {
        let wgpu_format = self.format.into();
        let wgpu_size = wgpu::Extent3d {
            width: self.width.into(),
            height: self.height.unwrap_or(1.try_into().unwrap()).into(),
            depth_or_array_layers: self.depth_or_array_slice.unwrap_or(1.try_into().unwrap()).into()
        };

        let mipmap_level = match &self.texels {
            TexelData::Plain(_t) => 1,
            TexelData::Mipmapped(m) => m.len(),
        } as u32;

        let descriptor = wgpu::TextureDescriptor{
            label: Some(name),
            size: wgpu_size,
            mip_level_count: mipmap_level,
            sample_count: 1,
            dimension: self.dimension.into(),
            format: wgpu_format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[]
        };
        let texture = device.create_texture(&descriptor);

        match &self.texels {
            TexelData::Plain(texels) => {
                let (bytes_per_row, rows_per_image) = Self::determine_upload_data_layout(wgpu_format, wgpu_size);
                let expected_total_texel_size = bytes_per_row * rows_per_image * wgpu_size.depth_or_array_layers;
                assert_eq!(expected_total_texel_size as usize, texels.len());

                queue.write_texture(
                    wgpu::TexelCopyTextureInfoBase {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                        aspect: wgpu::TextureAspect::All
                    },
                    texels,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(rows_per_image)
                    },
                    wgpu_size
                );
            },
            TexelData::Mipmapped(miplevels) => {
                // Not tested!
                let mut miplevel_size = wgpu_size;
                for (index, data) in miplevels.iter().enumerate() {
                    assert!(miplevel_size.width > 0 && miplevel_size.height > 0 && miplevel_size.depth_or_array_layers > 0);

                    let (bytes_per_row, rows_per_image) = Self::determine_upload_data_layout(wgpu_format, miplevel_size);
                    let expected_total_texel_size = bytes_per_row * rows_per_image * miplevel_size.depth_or_array_layers;
                    assert_eq!(expected_total_texel_size as usize, data.texels.len());

                    queue.write_texture(
                        wgpu::TexelCopyTextureInfoBase {
                            texture: &texture,
                            mip_level: index as u32,
                            origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                            aspect: wgpu::TextureAspect::All
                        },
                        &data.texels,
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(bytes_per_row),
                            rows_per_image: Some(rows_per_image)
                        },
                        miplevel_size
                    );

                    miplevel_size.width >>= 1;
                    miplevel_size.height >>= 1;
                    // 2D texture array does not generate mipmap.
                    if descriptor.dimension == wgpu::TextureDimension::D3 {
                        miplevel_size.depth_or_array_layers >>= 1;
                    }
                }
            },
        }

        return texture;
    }
}
