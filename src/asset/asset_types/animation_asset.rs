use serde::{Deserialize, Serialize};

use crate::asset::asset_types::{ConcreteAssetType, texture_asset::TextureAsset};

/// One baked track of an animation.
///
/// A baked texture plus the identity of what it drives.
/// 
/// The list stays open-ended because an animation is a set of channels over
/// different targets, so a future track kind (displacement, events) belongs here
/// rather than in a new asset kind.
#[derive(Debug, Serialize, Deserialize)]
pub enum BakedTrack {
    /// Joint matrices driving one skin.
    Joints {
        /// Skin asset this track animates.
        skin: String,
        /// Baked joint matrices.
        ///
        /// `width == joint_count * 4` and `height == frames + 1`. Each joint owns
        /// four consecutive texels holding the four *columns* of its object-space
        /// skinning matrix, so a joint matrix can be rebuilt with four
        /// `textureLoad` calls.
        texture: TextureAsset,
    },
    /// Morph weights driving one deformable mesh.
    MorphWeights {
        /// Deformable mesh asset this track animates.
        mesh: String,
        /// Morph target asset of that mesh.
        target: String,
        /// Baked weights.
        ///
        /// `width == ceil(targets / 4)` and `height == frames + 1`. Weights of up
        /// to four targets share one texel per frame.
        texture: TextureAsset,
    },
}

impl BakedTrack {
    /// The baked texture of this track.
    pub fn texture(&self) -> &TextureAsset {
        match self {
            Self::Joints { texture, .. } | Self::MorphWeights { texture, .. } => texture,
        }
    }
}

/// One baked animation.
#[derive(Debug, Serialize, Deserialize)]
pub struct AnimationAsset {
    /// Import source this clip was baked from, as `scene{scene}/anim{animation}`.
    ///
    /// This is the clip identity: it names the clip in the runtime manager and is
    /// the only identifier shared by every track of the animation.
    pub source: String,
    /// Every track the animation drives.
    ///
    /// At most one [`BakedTrack::Joints`] and any number of
    /// [`BakedTrack::MorphWeights`], one per morphing primitive.
    pub tracks: Vec<BakedTrack>,
    /// Number of unique baked frames, shared by every track.
    ///
    /// Each track texture is one row taller, repeating frame 0 so looping needs
    /// no modulo in the compute shader.
    pub frames: u32,
    /// Playback rate the frames were baked at.
    pub fps: f32,
}

impl AnimationAsset {
    /// The joint track driving `skin`, if this animation carries one.
    pub fn joint_track(&self, skin: &str) -> Option<&BakedTrack> {
        self.tracks.iter().find(|track| match track {
            BakedTrack::Joints { skin: track_skin, .. } => track_skin == skin,
            BakedTrack::MorphWeights { .. } => false,
        })
    }

    /// The weight track driving `mesh`, if this animation carries one.
    pub fn weight_track(&self, mesh: &str) -> Option<&BakedTrack> {
        self.tracks.iter().find(|track| match track {
            BakedTrack::MorphWeights { mesh: track_mesh, .. } => track_mesh == mesh,
            BakedTrack::Joints { .. } => false,
        })
    }
}

impl ConcreteAssetType for AnimationAsset {
    /// 1: initial. Replaces the split skeletal and morph-weight animation assets,
    ///    so both tracks of one glTF animation share one clock.
    const SCHEMA_VERSION: u32 = 1;
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
    /// TODO: since gltF channels have identical size, as the importer already
    /// enforces, there is no need to pass the channel argument. Maybe two
    /// methods (one for counts, one to query whether the channel exists)
    /// are better.
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
