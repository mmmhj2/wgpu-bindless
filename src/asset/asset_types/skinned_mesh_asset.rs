use serde::{Serialize, Deserialize};

use crate::asset::asset_types::{ConcreteAssetType, static_mesh_asset::StaticMeshAsset, vertex::VertexBufferSkinning};

#[derive(Serialize, Deserialize, Debug)]
pub struct SkinnedMeshAsset {
    /// Name of the skin asset that this mesh is bound to.
    pub skin: String,
    /// Rest pose geometry, in the object space of the skinned mesh.
    pub restpose: StaticMeshAsset,
    /// Per-vertex joint indices and weights, indexed like the rest pose vertex
    /// buffers. Consumed by the skinning compute shader as a storage buffer.
    pub animation : Vec<VertexBufferSkinning>,
}

impl ConcreteAssetType for SkinnedMeshAsset {
    const SCHEMA_VERSION: u32 = 1;
}

