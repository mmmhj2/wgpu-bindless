use serde::{Serialize, Deserialize};

use crate::asset::asset_types::{ConcreteAssetType, static_mesh_asset::StaticMeshAsset, vertex::VertexBufferSkinning};

/// TODO: rename to `AnimatedMeshAsset`
#[derive(Serialize, Deserialize, Debug)]
pub struct SkinnedMeshAsset {
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
    /// Empty for a morph-only mesh.
    pub animation : Vec<VertexBufferSkinning>,
}

impl ConcreteAssetType for SkinnedMeshAsset {
    /// 2: Added morph target
    const SCHEMA_VERSION: u32 = 2;
}
