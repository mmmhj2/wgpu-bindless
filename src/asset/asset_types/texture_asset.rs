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

impl Into<AddressMode> for gltf::texture::WrappingMode {
    fn into(self) -> AddressMode {
        match self {
            gltf::texture::WrappingMode::ClampToEdge => AddressMode::ClampToEdge,
            gltf::texture::WrappingMode::MirroredRepeat => AddressMode::MirrorRepeat,
            gltf::texture::WrappingMode::Repeat => AddressMode::Repeat,
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

impl FilterMode {
    pub fn import_from_gltf_sampler(sampler: &gltf::texture::Sampler) -> (Self, Self, Self) {
        let mipmap = match sampler.min_filter() {
            Some(x) => match x {
                gltf::texture::MinFilter::Nearest
                 | gltf::texture::MinFilter::Linear
                 | gltf::texture::MinFilter::NearestMipmapNearest
                 | gltf::texture::MinFilter::LinearMipmapNearest  => Self::Nearest,
                _ => Self::Linear,
            },
            None => Self::Nearest,
        };

        let mag = match sampler.mag_filter() {
            Some(x) => match x {
                gltf::texture::MagFilter::Nearest => Self::Nearest,
                gltf::texture::MagFilter::Linear => Self::Linear,
            },
            None => Self::Nearest,
        };

        let min = match sampler.min_filter() {
            Some(x) => match x {
                gltf::texture::MinFilter::Nearest
                 | gltf::texture::MinFilter::NearestMipmapNearest
                 | gltf::texture::MinFilter::NearestMipmapLinear => Self::Nearest,
                _ => Self::Linear
            },
            None => Self::Nearest,
        };

        (mag, min, mipmap)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Sampler {
    /// Address mode for u,v,w coordinates
    pub address_mode: (AddressMode, AddressMode, AddressMode),
    /// Filter mode for magnification, minification and mipmap
    pub filter      : (FilterMode, FilterMode, FilterMode)
}

impl Sampler {
    fn import_from_gltf_sampler(sampler: &gltf::texture::Sampler) -> Self {
        Self { 
            address_mode: (sampler.wrap_s().into(), sampler.wrap_t().into(), AddressMode::ClampToEdge),
            filter: FilterMode::import_from_gltf_sampler(sampler)
        }
    }
}


#[derive(Serialize, Deserialize, Debug)]
pub struct TextureAsset {
    format: TextureFormat,
    dimension: u8,
    size: (u32, u32, u32),
    sampler: Sampler,
    texels: Vec<u8>,
}

impl ConcreteAssetType for TextureAsset {
}

impl TextureAsset {
    fn match_texture_format(f: gltf::image::Format, is_srgb: bool) -> TextureFormat {
        match f {
            gltf::image::Format::R8G8B8A8 => if is_srgb {TextureFormat::Rgba8Srgb} else {TextureFormat::Rgba8Unorm},
            gltf::image::Format::R16G16B16A16 => TextureFormat::Rgba16Unorm,
            gltf::image::Format::R32G32B32A32FLOAT => TextureFormat::Rgba32Float,
            _ => panic!("Unsupported GLTF texture format")
        }
    }

    pub fn import_from_gltf_texture(
        texture: &gltf::Texture,
        images: &Vec<gltf::image::Data>,
        is_srgb: bool
    ) -> Result<TextureAsset, ()> {
        let image = &images[texture.index()];

        let converted_format = Self::match_texture_format(image.format, is_srgb);
        let image = &images[texture.source().index()];

        let asset = Self {
            format: converted_format,
            dimension: 2,
            size: (image.width, image.height, 1),
            sampler: Sampler::import_from_gltf_sampler(&texture.sampler()),
            // One gltf image can correspond to multiple textures with different sampler.
            // So we must clone its data for safety.
            texels: image.pixels.clone(),
        };

        Ok(asset)
    }
}
