use std::{fs::File, io::Write};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{asset::{asset_types::ConcreteAssetType, importer::{ImporterContext, gltf_importer::GltfImporter, vertex_reconditioner::{TangentRecalculator, VertexColorApplyScale, VertexReconditionable, VertexReconditionableAttributeWrite}}}, renderer::mesh::vertex_types::{VertexBufferOthers, VertexBufferPosition}};

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
}

impl StaticMeshAsset {
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
impl VertexColorApplyScale for StaticMeshAsset {}
