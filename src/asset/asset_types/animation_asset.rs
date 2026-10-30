use serde::{Deserialize, Serialize};

use crate::asset::asset_types::{ConcreteAssetType, GpuAssetType, texture_asset::TextureAsset};


#[derive(Debug, Serialize, Deserialize)]
pub struct SkinnedAnimationAsset {
    /// Name of the skin asset that this animation applies to.
    pub skin: String,
    /// Number of baked frames, i.e. the height of [`Self::texture`].
    pub frames: u32,
    /// Playback rate the frames were baked at.
    pub fps: f32,
    /// Baked joint matrices.
    ///
    /// `width == joint_count * 4` and `height == frames`. Each joint owns four
    /// consecutive texels holding the four *columns* of its object-space
    /// skinning matrix, so a joint matrix can be rebuilt with four `textureLoad`
    /// calls.
    pub texture: TextureAsset
}

impl ConcreteAssetType for SkinnedAnimationAsset {
    const SCHEMA_VERSION: u32 = 1;
}

impl GpuAssetType for SkinnedAnimationAsset {
    type GpuObjectType = wgpu::Texture;
    
    /// Upload the baked animation texture onto the GPU.
    /// 
    /// Joint animations are baked into a 2D texture at import time, one row per
    /// frame. The skinning compute shader reads the row selected by the `frame`
    /// immediate with `textureLoad`, so no sampling or filtering is involved.
    /// 
    /// The texture has the format of RGBA16 by default. It should provide more
    /// than enough precision. If bandwidth is a problem, RGBA8 may be used in
    /// the future.
    /// 
    /// All transform data is baked *in model space*, to be directly used in the
    /// compute shader.
    fn upload(&self, name: &str, device: &wgpu::Device, queue: &wgpu::Queue) -> Self::GpuObjectType {
        self.texture.upload(name, device, queue)
    }
}
