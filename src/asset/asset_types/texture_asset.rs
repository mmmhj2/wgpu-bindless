use std::num::NonZero;

use serde::{Deserialize, Serialize};

use crate::asset::asset_types::ConcreteAssetType;


#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    Rgba8Unorm,
    Rgba8Srgb,
    Rgba16Unorm,
    Rgba32Float,
    R8Unorm,
    Rg8Unorm,
    R16Unorm,
    Rg16Unorm,
    Rgba16Float
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum AddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
    ClampToBorder
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum FilterMode {
    Nearest,
    Linear
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

/// serde_bytes does not support array of arrays of u8, this struct works around it.
#[derive(Serialize, Deserialize, Debug)]
pub struct MipmapLayerData {
    #[serde(with = "serde_bytes")]
    texels: Box<[u8]>
}

impl MipmapLayerData {
    /// Raw texel bytes of this mip level.
    ///
    /// The renderer needs this to upload the payload; the field stays private so
    /// the invariant (serde_bytes layout) is only enforced here.
    pub fn texels(&self) -> &[u8] { &self.texels }
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
    /// 2: added [`TextureFormat::Rgba16Float`].
    const SCHEMA_VERSION: u32 = 2;
}

