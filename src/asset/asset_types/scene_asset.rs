use serde::{Deserialize, Serialize};

use crate::asset::asset_types::ConcreteAssetType;

/// A single node of a scene's transform hierarchy.
///
/// Nodes are stored in an arena (`SceneAsset::nodes`); hierarchy is encoded by
/// `children` indices. Each node keeps its *local* transform so the runtime can
/// rebuild the hierarchy without re-reading the source scene. World transforms
/// are derived at runtime by `renderer::scene::graph::SceneGraph`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SceneNodeAsset {
    /// Optional node name from the source scene, useful for debugging.
    pub name: Option<String>,
    /// Local transform relative to the parent, column-major.
    pub local_matrix: [[f32; 4]; 4],
    /// Subasset names of the static meshes attached to this node.
    ///
    /// One entry per primitive: a node may reference a mesh made of several
    /// primitives, and each primitive is an independent renderable mesh.
    pub mesh_subasset_names: Vec<String>,
    /// Child node indices into [`SceneAsset::nodes`].
    pub children: Vec<u32>,
}

/// A scene represented as a hierarchy of nodes with local transforms.
///
/// The source hierarchy is preserved: no world matrix is baked at import time.
/// `roots` lists the node indices that have no parent.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SceneAsset {
    /// Node arena. Hierarchy is encoded by [`SceneNodeAsset::children`].
    pub nodes: Vec<SceneNodeAsset>,
    /// Indices of root nodes in [`SceneAsset::nodes`].
    pub roots: Vec<u32>,
}

impl ConcreteAssetType for SceneAsset {
    const SCHEMA_VERSION: u32 = 2;
}
