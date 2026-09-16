use serde::{Deserialize, Serialize};

use crate::asset::asset_types::ConcreteAssetType;


#[derive(Serialize, Deserialize, Debug)]
pub enum TextureFormat {
    Rgba8Unorm,
    Rgba8Srgb,
    Rgba16Unorm,
    Rgba32Float
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

#[derive(Serialize, Deserialize, Debug)]
pub struct Sampler {
    /// Address mode for u,v,w coordinates
    pub address_mode: (AddressMode, AddressMode, AddressMode),
    /// Filter mode for magnification, minification and mipmap
    pub filter      : (FilterMode, FilterMode, FilterMode)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TextureAsset {
    pub format: TextureFormat,
    pub dimension: u8,
    pub size: (u32, u32, u32),
    pub sampler: Sampler,
    pub texels: Vec<u8>,
}

impl ConcreteAssetType for TextureAsset {
}

impl TextureAsset {
}
