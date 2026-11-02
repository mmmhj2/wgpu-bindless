//! Conversions from the serialized texture description to `wgpu` enums.

use crate::asset::asset_types::texture_asset::{
    AddressMode,
    FilterMode,
    TextureDimension,
    TextureFormat,
};

impl From<TextureFormat> for wgpu::TextureFormat {
    fn from(value: TextureFormat) -> Self {
        match value {
            TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8Srgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Rgba16Unorm => wgpu::TextureFormat::Rgba16Unorm,
            TextureFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
            TextureFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
            TextureFormat::Rg8Unorm => wgpu::TextureFormat::Rg8Unorm,
            TextureFormat::R16Unorm => wgpu::TextureFormat::R16Unorm,
            TextureFormat::Rg16Unorm => wgpu::TextureFormat::Rg16Unorm,
            TextureFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        }
    }
}

impl From<AddressMode> for wgpu::AddressMode {
    fn from(value: AddressMode) -> Self {
        match value {
            AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            AddressMode::Repeat => wgpu::AddressMode::Repeat,
            AddressMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
            // TODO: figure out how WGPU handles custom border color, and determine a hashing algo.
            AddressMode::ClampToBorder => todo!("Clamp to border is currently unsupported."),
        }
    }
}

impl From<FilterMode> for wgpu::FilterMode {
    fn from(value: FilterMode) -> Self {
        match value {
            FilterMode::Nearest => wgpu::FilterMode::Nearest,
            FilterMode::Linear => wgpu::FilterMode::Linear,
        }
    }
}

impl From<FilterMode> for wgpu::MipmapFilterMode {
    fn from(value: FilterMode) -> Self {
        match value {
            FilterMode::Nearest => wgpu::MipmapFilterMode::Nearest,
            FilterMode::Linear => wgpu::MipmapFilterMode::Linear,
        }
    }
}

impl From<TextureDimension> for wgpu::TextureDimension {
    fn from(value: TextureDimension) -> Self {
        match value {
            TextureDimension::D1 => wgpu::TextureDimension::D1,
            TextureDimension::D2 => wgpu::TextureDimension::D2,
            TextureDimension::D2Array => wgpu::TextureDimension::D2,
            TextureDimension::D3 => wgpu::TextureDimension::D3,
        }
    }
}
