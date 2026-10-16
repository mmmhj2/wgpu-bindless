use std::{fs::File, io::Write};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::asset::{asset_types::{ConcreteAssetType, GpuAssetType, vertex::{VertexBufferOthers, VertexBufferPosition}}, importer::vertex_reconditioner::{TangentRecalculator, VertexColorApplyScale, VertexReconditionable, VertexReconditionableAttributeWrite}};

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

/// GPU-resident geometry produced by uploading a [`StaticMeshAsset`].
///
/// The vertex buffers are ordered to match the renderer's vertex buffer
/// layouts (`renderer::mesh::vertex_types`): index 0 is the position stream,
/// index 1 is the attribute stream.
pub struct GpuStaticMesh {
    pub vertex_buffers: Vec<wgpu::Buffer>,
    pub index_buffer: Option<wgpu::Buffer>,
    pub vertex_draw_count: u32
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

impl GpuAssetType for StaticMeshAsset {
    type GpuObjectType = GpuStaticMesh;

    /// Upload the vertex and index buffers onto the GPU device.
    ///
    /// Vertex data is written directly from the asset's own slices, so no
    /// intermediate copy of the whole vertex buffer is made. The vertex data is
    /// expected to be final (see [`StaticMeshAsset::recondition`]), which is why
    /// `&self` suffices for a zero-copy upload.
    fn upload(&self, name: &str, device: &wgpu::Device, queue: &wgpu::Queue) -> Self::GpuObjectType {
        assert_eq!(self.vp.len(), self.va.len(), "position and attribute buffers must have the same length.");

        let position_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some(&format!("{name}.position")),
            size: std::mem::size_of_val(self.vp.as_slice()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&position_buffer, 0, bytemuck::cast_slice(&self.vp));

        let attribute_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some(&format!("{name}.attribute")),
            size: std::mem::size_of_val(self.va.as_slice()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&attribute_buffer, 0, bytemuck::cast_slice(&self.va));

        let index_buffer = self.vi.as_ref().map(|indices| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor{
                label: Some(&format!("{name}.index")),
                size: std::mem::size_of_val(indices.as_slice()) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(indices));
            buffer
        });

        GpuStaticMesh {
            vertex_buffers: vec![position_buffer, attribute_buffer],
            index_buffer,
            vertex_draw_count: self.vertex_draw_count
        }
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
