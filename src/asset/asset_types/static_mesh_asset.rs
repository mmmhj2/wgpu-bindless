use serde::{Deserialize, Serialize};

use crate::asset::{asset_types::{ConcreteAssetType, vertex::{VertexBufferOthers, VertexBufferPosition}}, importer::vertex_reconditioner::{NormalRecalculator, TangentRecalculator, VertexColorApplyScale, VertexReconditionable, VertexReconditionableAttributeWrite}};

#[derive(Serialize, Deserialize, Debug)]
pub struct StaticMeshMaterial {
    pub diffuse_texture_name: Option<String>,
    pub normal_texture_name: Option<String>,
    pub mrao_texture_name: Option<String>
}

#[derive(Serialize, Deserialize, Debug)]
pub struct StaticMeshAsset {
    pub vp  : Vec<VertexBufferPosition>,
    pub va  : Vec<VertexBufferOthers>,
    pub vi  : Option<Vec<u32>>,
    pub material : StaticMeshMaterial,
    pub vertex_draw_count : u32,
    pub vertex_color_scale: [f32; 4]
}

impl ConcreteAssetType for StaticMeshAsset {
    const SCHEMA_VERSION: u32 = 1;
}

impl StaticMeshAsset {
    /// Recondition the vertex data in place so that it is ready for a
    /// zero-copy GPU upload.
    ///
    /// Importers are expected to call this once before the asset is saved.
    /// Tangent recalculation is performed separately by the importer when the
    /// source lacked tangents (see `GltfImporter`); this applies the configured
    /// vertex color scale.
    pub fn recondition(&mut self) {
        self.rescale_vertex_color(self.vertex_color_scale);
    }
}

impl VertexReconditionable for StaticMeshAsset {
    fn get_position_buffer(&self) -> &Vec<VertexBufferPosition> { &self.vp }
    fn get_attribute_buffer(&self) -> &Vec<VertexBufferOthers> { &self.va }
    fn get_index_buffer(&self) -> Option<&Vec<u32>> { self.vi.as_ref() }
}
impl VertexReconditionableAttributeWrite for StaticMeshAsset {
    fn get_attribute_buffer_mut(&mut self) -> &mut Vec<VertexBufferOthers> { &mut self.va }
}
impl TangentRecalculator for StaticMeshAsset {}
impl NormalRecalculator for StaticMeshAsset {}
impl VertexColorApplyScale for StaticMeshAsset {}
