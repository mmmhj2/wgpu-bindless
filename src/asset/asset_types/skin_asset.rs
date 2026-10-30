use serde::{Deserialize, Serialize};

use crate::asset::asset_types::ConcreteAssetType;


/// Skin joint modeled after glTF specification.
#[derive(Debug, Serialize, Deserialize)]
pub struct SkinJoint {
    pub children: Vec<u32>,
    /// Rest pose matrix in column-major (as in cgmath or OpenGL) of this bone.
    /// 
    /// This transform is relative to its parent (or origin if it is the root).
    pub rest_pose_matrix: [[f32; 4]; 4],
    /// Inverse bind matrix in column-major (as in cgmath or OpenGL).
    /// 
    /// Inverse bind matrix is actually the inverse of the rest post matrix in
    /// the global space. This is used to help translate vertex coordinates into
    /// the joint space. glTF spec mandates that it must be accessed from a buffer
    /// view, but maybe we need to calculate it ourselves for other formats.
    pub inverse_bind_matrix: [[f32; 4]; 4]
}

/// Skin asset which corresponds to glTF `skins` element.

#[derive(Debug, Serialize, Deserialize)]
pub struct SkinAsset {
    pub joints: Vec<SkinJoint>,
}

impl ConcreteAssetType for SkinAsset {
    const SCHEMA_VERSION: u32 = todo!();
}
