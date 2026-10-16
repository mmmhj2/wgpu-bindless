use serde::{Deserialize, Serialize};

use crate::asset::asset_types::ConcreteAssetType;

/// A single mesh instance baked into a scene asset.
///
/// The transform is flattened at import time so the renderer does not need to
/// re-parse the source glTF to walk the scene graph.
#[derive(Serialize, Deserialize, Debug)]
pub struct SceneMeshInstanceAsset {
    /// World matrix in column-major order, matching cgmath and WGSL conventions.
    pub world_matrix: [f32; 16],
    /// Subasset name of the referenced static mesh (primitive granularity).
    pub mesh_subasset_name: String,
}

/// A pre-flattened scene: a flat list of mesh instances with baked world
/// transforms.
#[derive(Serialize, Deserialize, Debug)]
pub struct SceneAsset {
    pub instances: Vec<SceneMeshInstanceAsset>,
}

impl ConcreteAssetType for SceneAsset {
    const SCHEMA_VERSION: u32 = 1;
}
