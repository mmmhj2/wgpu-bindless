use serde::{Deserialize, Serialize};

use crate::asset::asset_types::{ConcreteAssetType, texture_asset::TextureAsset};


#[derive(Debug, Serialize, Deserialize)]
pub struct SkinnedAnimationAsset {
    /// Name of the skin asset that this animation applies to.
    pub skin: String,
    /// Number of unique baked frames.
    ///
    /// [`Self::texture`] is one row taller for wrapping in the compute shader.
    pub frames: u32,
    /// Playback rate the frames were baked at.
    pub fps: f32,
    /// Baked joint matrices.
    ///
    /// `width == joint_count * 4` and `height == frames + 1`. Each joint owns four
    /// consecutive texels holding the four *columns* of its object-space
    /// skinning matrix, so a joint matrix can be rebuilt with four `textureLoad`
    /// calls.
    pub texture: TextureAsset
}

impl ConcreteAssetType for SkinnedAnimationAsset {
    const SCHEMA_VERSION: u32 = 1;
}

/// Data storage for morph targets of a mesh.
/// 
/// glTF spec mandates support for TEXCOORD_n and COLOR_n, which we are currently
/// not supporting for simplicity.
#[derive(Debug, Serialize, Deserialize)]
pub enum VertexMorphTargetData {
    Position(Vec<[f32; 3]>),
    Normal(Vec<[f32; 3]>),
    // glTF spec says that tangets must have 4 components.
    // However I don't believe any morph targets can change handiness,
    // which is a transform that can definitely not be interpolated linearly.
    // Correction: morph target for tangents has 3 components only.
    Tangent(Vec<[f32; 3]>)
}

/// Asset of containing all morph targets to a single skinned mesh.
/// 
/// Vertex morph targets should be uploaded to GPU as storage buffers to be
/// used in compute shaders.
#[derive(Debug, Serialize, Deserialize)]
pub struct VertexMorphTargetAsset {
    /// Skinned mesh that the morph target is attached to.
    pub mesh: String,
    /// Channels of morph targets.
    /// 
    /// Up to 8 channels can be specified according to glTF spec.
    pub data: Vec<VertexMorphTargetData>
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VertexMorphAnimationAsset {
    /// Mesh that the animation is attached to.
    /// 
    /// Contrary to skeletal animations, vertex morph targets does not need
    /// a skin to work. They are instead attached to primitives, which we
    /// process as skinned mesh assets.
    pub mesh: String,
    pub frames: u32,
    /// Playback rate the frames were baked at.
    pub fps: f32,
    /// Baked weight texture.
    /// 
    /// The texture has `height == frames + 1` and `width = floor(channels/4)`,
    /// meaning weights of up to four channels can share a single texel per
    /// frame. Compute shader should load morph target weights from the texture
    /// and manipulate vertex attributes accordingly.
    /// 
    /// Uses RGBA16 format by default.
    pub texture: TextureAsset
}

