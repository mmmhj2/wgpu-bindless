use std::{fs::File, io::Write};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{asset::{asset_types::ConcreteAssetType, importer::{ImporterContext, gltf_importer::GltfImporter, vertex_reconditioner::{TangentRecalculator, VertexColorApplyScale, VertexReconditionable, VertexReconditionableAttributeWrite}}}, renderer::mesh::vertex_types::{VertexBufferOthers, VertexBufferPosition}};

#[derive(Serialize, Deserialize, Debug)]
struct StaticMeshMaterial {
    pub diffuse_texture_name: Option<String>,
    pub normal_texture_name: Option<String>,
    pub mrao_texture_name: Option<String>
}

#[derive(Serialize, Deserialize, Debug)]
pub struct StaticMeshAsset {
    vp  : Vec<VertexBufferPosition>,
    va  : Vec<VertexBufferOthers>,
    vi  : Option<Vec<u32>>,
    material : StaticMeshMaterial,
    vertex_draw_count : u32,
    vertex_color_scale: [f32; 4]
}

impl ConcreteAssetType for StaticMeshAsset {
}

impl StaticMeshAsset {
    fn construct_position_buffer (
        primitive: &gltf::Primitive,
        buffers: &Vec<gltf::buffer::Data>
    ) -> Vec<VertexBufferPosition> {
        let mut position_buffer: Vec<VertexBufferPosition> = Vec::new();

        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
        if let Some(iter) = reader.read_positions() {
           for vertex_position in iter {
               position_buffer.push(VertexBufferPosition { position: vertex_position });
           }
       }

        position_buffer
    }

    fn construct_attribute_buffer (
        primitive: &gltf::Primitive,
        buffers: &Vec<gltf::buffer::Data>,
        vertices: usize
    ) -> (Vec<VertexBufferOthers>, bool) {
        let mut attribute_buffer = Vec::new();
        attribute_buffer.resize(
            vertices,
            VertexBufferOthers { color: [1.0, 1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0], tangent: [0.0, 0.0, 0.0, 1.0], uv0: [0.0, 0.0] }
        );

        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
        if let Some(iter) = reader.read_colors(0) {
            for (i, c) in iter.into_rgba_f32().enumerate() {
                attribute_buffer[i].color = c;
            }
        } else {
            log::warn!("Imported mesh does not have vertex color, or the vertex color is not located at set 0.")
        }

        if let Some(iter) = reader.read_normals() {
            for (i, n) in iter.enumerate() {
                attribute_buffer[i].normal = n;
            }
        } else {
            log::warn!("Imported mesh does not vertex have normal.")
        }

        if let Some(iter) = reader.read_tex_coords(0) {
            for (i, uv) in iter.into_f32().enumerate() {
                attribute_buffer[i].uv0 = uv;
            }
        } else {
            log::warn!("Imported mesh does not have texture coordinate, or the texcoord is not located at set 0.")
        }

        if let Some(iter) = reader.read_tangents() {
            for (i, n) in iter.enumerate() {
                attribute_buffer[i].tangent = n;
            }

            (attribute_buffer, false)
        } else {
            log::info!("Imported mesh does not vertex have tangent. It will be automatically calculated on upload.");
            (attribute_buffer, true)
        }
    }

    fn construct_index_buffer (
        primitive: &gltf::Primitive,
        buffers: &Vec<gltf::buffer::Data>
    ) -> Option<Vec<u32>> {
        let mut indices = Vec::new();

        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
        let iter = reader.read_indices()?;
        for i in iter.into_u32() {
            indices.push(i);
        }

        Some(indices)
    }


    pub(crate) fn import_from_gltf_primitive(
        context: &ImporterContext,
        primitive: &gltf::Primitive,
        buffers: &Vec<gltf::buffer::Data>
    ) -> Self {
        let position_buffer = Self::construct_position_buffer(primitive, buffers);
        let (attribute_buffer, calculate_tangent) = Self::construct_attribute_buffer(primitive, buffers, position_buffer.len());
        let index_buffer = Self::construct_index_buffer(&primitive, buffers);
        let vertex_draw_count = match index_buffer.as_ref() {
            Some(b) => b.len(),
            None => position_buffer.len()
        } as u32;

        let mut ret = Self {
            vp: position_buffer,
            va: attribute_buffer,
            vi: index_buffer,
            material: StaticMeshMaterial {
                diffuse_texture_name: None,
                normal_texture_name: None,
                mrao_texture_name: None
            },
            vertex_draw_count,
            vertex_color_scale: [1.0; 4]
        };

        // Process the materials
        let pbr_material = primitive.material().pbr_metallic_roughness();

        ret.material.diffuse_texture_name = if let Some(base_color_texture) = pbr_material.base_color_texture() {
            Some(
                GltfImporter::generate_subasset_name(
                    &context.imported_file_path,
                    base_color_texture.texture().index(),
                    crate::asset::importer::gltf_importer::SubAssetType::Texture
                ).into_string().expect("diffuse texture name should contain UTF-8 chars only")
            )
        } else {
            None
        };

        ret.material.normal_texture_name = if let Some(normal_texture) = primitive.material().normal_texture() {
            Some(
                GltfImporter::generate_subasset_name(
                    &context.imported_file_path,
                    normal_texture.texture().index(),
                    crate::asset::importer::gltf_importer::SubAssetType::Texture
                ).into_string().expect("normal texture name should contain UTF-8 chars only")
            )
        } else {
            None
        };

        let should_have_mrao_texture = primitive.material().pbr_metallic_roughness().metallic_roughness_texture().is_some() || primitive.material().occlusion_texture().is_some();
        if should_have_mrao_texture {
            todo!("MRAO texture is not implemented.")
        };
        
        if calculate_tangent {
            ret.recalculate_tangents();
        }

        return ret;
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
impl VertexColorApplyScale for StaticMeshAsset {}
