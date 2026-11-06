use serde::{Deserialize, Serialize};

use crate::asset::asset_types::{ConcreteAssetType, texture_asset::TextureAsset};


#[derive(Debug, Serialize, Deserialize)]
pub struct SkinnedAnimationAsset {
    /// Name of the skin asset that this animation applies to.
    pub skin: String,
    /// Import source this clip was baked from, as `scene{scene}/anim{animation}`.
    ///
    /// Both the skeletal track and the morph-weight track of the same glTF
    /// animation share this identifier, so they can be merged into one clip.
    pub source: String,
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
    // 2: Added `source` field.
    const SCHEMA_VERSION: u32 = 2;
}

/// Upper bound on morph targets per mesh primitive, matching the glTF limit.
///
/// The GPU layout reserves this many mask slots, so raising it changes the
/// layout word count.
pub const MAX_MORPH_TARGETS: usize = 8;

/// One glTF morph target.
///
/// The index of a target in [`VertexMorphTargetAsset::targets`] is its index in
/// the glTF `weights` array, so every attribute delta stored here shares one
/// weight. Missing attributes are `None` and occupy no GPU space.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MorphTarget {
    /// Per-vertex position deltas.
    pub position : Option<Vec<[f32; 3]>>,
    /// Per-vertex normal deltas. Renormalized after accumulation.
    pub normal   : Option<Vec<[f32; 3]>>,
    /// Per-vertex tangent deltas.
    /// The handedness (`w`) does not take part in morphing.
    pub tangent  : Option<Vec<[f32; 3]>>,
    /// Per-vertex UV deltas.
    /// Note: Reserved. The importer does not produce them yet.
    pub texcoord0: Option<Vec<[f32; 2]>>,
    /// Per-vertex color deltas.
    /// 
    /// A `VEC3` source yields an alpha delta of zero.
    /// Note: Reserved.  The importer does not produce them yet.
    pub color0   : Option<Vec<[f32; 4]>>,
}

impl MorphTarget {
    /// Number of attribute slots the GPU layout and shaders reserve per target.
    ///
    /// The slot order is exactly the field order above.
    /// Growing this count is a layout change the shader and the layout buffer
    /// must be updated together.
    pub const CHANNEL_COUNT: usize = 5;

    /// Deltas of one attribute by slot index, in [`Self::CHANNEL_COUNT`] order.
    /// 
    /// TODO: I suppose all target should have the same size according to glTF spec.
    /// Needs confirmation.
    pub fn attribute_vertex_count(&self, channel: usize) -> Option<usize> {
        match channel {
            0 => self.position.as_ref().map(Vec::len),
            1 => self.normal.as_ref().map(Vec::len),
            2 => self.tangent.as_ref().map(Vec::len),
            3 => self.texcoord0.as_ref().map(Vec::len),
            4 => self.color0.as_ref().map(Vec::len),
            _ => None,
        }
    }
}

/// Asset of containing all morph targets to a single skinned mesh.
///
/// Vertex morph targets should be uploaded to GPU as storage buffers to be
/// used in compute shaders.
#[derive(Debug, Serialize, Deserialize)]
pub struct VertexMorphTargetAsset {
    /// Skinned mesh that the morph target is attached to.
    pub mesh: String,
    /// Morph targets of the mesh, indexed like the glTF `weights` array.
    ///
    /// Up to 8 targets can be specified according to glTF spec.
    pub targets: Vec<MorphTarget>,
    /// `mesh.weights` of the source mesh.
    ///
    /// Acts as the base shape when the playing clip has no weights track, i.e
    /// as the neutral input of the morphing stage. `None` or all-zero means the
    /// mesh has no static deformation.
    pub default_weights: Option<Vec<f32>>,
}

impl ConcreteAssetType for VertexMorphTargetAsset {
    const SCHEMA_VERSION: u32 = 1;
}

/// Asset for vertex morph animation.
///
/// Vertex morphs are attached to meshes with morph targets, instead of skins.
/// So they are managed separately.
#[derive(Debug, Serialize, Deserialize)]
pub struct VertexMorphAnimationAsset {
    /// Skinned mesh that the animation is attached to.
    ///
    /// Contrary to skeletal animations, vertex morph targets does not need
    /// a skin to work. They are instead attached to primitives, which we
    /// process as skinned mesh assets.
    pub mesh: String,
    /// Morph target asset of this animation.
    pub target: String,
    /// Import source this clip was baked from, as `scene{scene}/anim{animation}`.
    ///
    /// Shared with the skeletal track of the same glTF animation. Together with
    /// [`Self::mesh`] it identifies the clip the weights belong to.
    pub source: String,
    /// Number of unique baked frames.
    ///
    /// Computed from the same animation duration as the skeletal track, so both
    /// tracks of one clip have an equal frame count.
    pub frames: u32,
    /// Playback rate the frames were baked at.
    pub fps: f32,
    /// Baked weight texture.
    ///
    /// The texture has `height == frames + 1` and `width = ceil(targets/4)`,
    /// meaning weights of up to four targets can share a single texel per
    /// frame. Compute shader should load morph target weights from the texture
    /// and manipulate vertex attributes accordingly.
    ///
    /// Uses RGBA16F format, dimension 2D.
    pub texture: TextureAsset
}

impl ConcreteAssetType for VertexMorphAnimationAsset {
    const SCHEMA_VERSION: u32 = 1;
}
