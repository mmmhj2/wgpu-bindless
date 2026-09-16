use serde::{Deserialize, Serialize};

use crate::asset::asset_types::ConcreteAssetType;


#[derive(Serialize, Deserialize, Debug)]
pub enum TextureFormat {
    Rgba8Unorm,
    Rgba8Srgb,
    Rgba16Unorm,
    Rgba32Float
}


#[derive(Serialize, Deserialize, Debug)]
pub struct TextureAsset {
    format: TextureFormat,
    dimensions: (u32, u32, u32),
    texels: Vec<u8>,
}

impl ConcreteAssetType for TextureAsset {
}
