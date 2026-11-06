use serde::{Serialize, Deserialize};

use crate::asset::asset_types::{ConcreteAssetType, static_mesh_asset::StaticMeshAsset, vertex::VertexBufferSkinning};

/// One geometry that animation can deform.
///
/// This is what the animation system calls a **rig**: the thing that can be
/// driven, not the skeleton. A rig names the skin it is bound to and the morph
/// target it carries (either may be absent) and owns the rest-pose geometry
/// together with the per-vertex skinning attributes.
///
/// Two primitives of one mesh are two rigs even when they share a skin, because
/// the rest pose, the morph target and the skinning stream are all
/// per-primitive.
#[derive(Serialize, Deserialize, Debug)]
pub struct DeformableMeshAsset {
    /// Name of the skin asset that this mesh is bound to.
    ///
    /// `None` for a morph-only mesh. Vertex morph targets do not need a skin,but
    /// such meshes still use this asset so that one asset kind covers every
    /// deformable mesh.
    pub skin: Option<String>,
    /// Name of the morph target asset of this mesh, if it has morph targets.
    ///
    /// The target asset is shared by every instance of the mesh. The morph
    /// weights come either from a clip or from the target's static base weights.
    pub morph_target: Option<String>,
    /// Rest pose geometry, in the object space of the skinned mesh.
    pub restpose: StaticMeshAsset,
    /// Per-vertex joint indices and weights, indexed like the rest pose vertex
    /// buffers.
    /// 
    /// Consumed by the skinning compute shader as a storage buffer.
    ///
    /// `None` for a morph-only mesh. Such a mesh never dispatches the skinning
    /// stage, so it has no skinning stream at all.
    pub skinning: Option<Vec<VertexBufferSkinning>>,
}

impl ConcreteAssetType for DeformableMeshAsset {
    /// 2: Added morph target.
    /// 3: `animation: Vec` became `skinning: Option<Vec>`; morph-only meshes
    ///    store `None` instead of an empty sentinel stream.
    const SCHEMA_VERSION: u32 = 3;
}
