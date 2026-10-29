use serde::{Deserialize, Serialize};

use crate::asset::asset_types::{ConcreteAssetType, GpuAssetType};


#[derive(Debug, Serialize, Deserialize)]
pub struct AnimationAsset {
}

impl ConcreteAssetType for AnimationAsset {
    const SCHEMA_VERSION: u32 = 1;
}

impl GpuAssetType for AnimationAsset {
    type GpuObjectType = wgpu::Texture;
    
    /// Upload the animation onto the GPU.
    /// 
    /// Joint animations should be baked into a 2D texture. This leverages GPU
    /// hardware to automatically do linear interpolation over time.
    /// 
    /// It is possible to either:
    /// - directly bake vertex positions into the texture which is known as Vertex Attribute Texture (VAT),
    /// - or just bake the transform sequences (e.g. quaternions, translation
    /// vectors or matrices).
    /// 
    /// Needs to decide which to use.
    fn upload(&self, name: &str, device: &wgpu::Device, queue: &wgpu::Queue) -> Self::GpuObjectType {
        todo!()
    }
}
