use serde::{Deserialize, Serialize};

use crate::asset::asset_types::{ConcreteAssetType, GpuAssetType};


#[derive(Debug, Serialize, Deserialize)]
pub struct SkinnedAnimationAsset {
    /// Name of the skin asset that this animation applies to.
    pub skin: String
}

impl ConcreteAssetType for SkinnedAnimationAsset {
    const SCHEMA_VERSION: u32 = todo!();
}

impl GpuAssetType for SkinnedAnimationAsset {
    type GpuObjectType = wgpu::Texture;
    
    /// Upload the animation onto the GPU.
    /// 
    /// Joint animations should be baked into a 2D texture. This leverages GPU
    /// hardware to automatically do linear interpolation over time.
    /// 
    /// It is possible to either:
    /// - directly bake vertex positions into the texture which is known as
    /// Vertex Attribute Texture (VAT),
    /// - or just bake the transform sequences (i.e. matrices).
    /// 
    /// Needs to decide which to use. The latter one is often implemented with
    /// texel buffers, but that is not available for WebGPU.
    /// 
    /// Whichever route is chosen, all transform data uploaded onto the GPU must
    /// be *in model space*, to be directly used in the compute shader.
    fn upload(&self, name: &str, device: &wgpu::Device, queue: &wgpu::Queue) -> Self::GpuObjectType {
        todo!()
    }
}
