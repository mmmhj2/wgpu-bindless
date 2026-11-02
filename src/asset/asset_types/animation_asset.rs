use serde::{Deserialize, Serialize};

use crate::asset::asset_types::{ConcreteAssetType, texture_asset::TextureAsset};


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
