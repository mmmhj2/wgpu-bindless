use std::cell::OnceCell;

use crate::{asset::asset_types::skinned_mesh_asset::GpuSkinnedMesh, renderer::{drawable::{drawable_mesh_traits::{DrawableMesh, Mesh}, vertex_types::VertexType}, pipeline::pbr_material::PBRMaterial}};

/// Skinned animated mesh.
/// 
/// Skinned meshes store both a rest-pose reference mesh in `geometry`, and the
/// intermediate vertex attributes that are produced by the compute shader.
/// 
/// Contrary to static meshes, skinned meshes are *never* instanced.
#[derive(Debug)]
pub struct SkinnedMesh {
    geometry: GpuSkinnedMesh,
    deformed_position: wgpu::Buffer,
    material: PBRMaterial,
    draw_buffers: OnceCell<Vec<wgpu::Buffer>>
}

impl SkinnedMesh {
    pub(crate) fn new(device: &wgpu::Device, geometry: GpuSkinnedMesh, material: PBRMaterial) -> Self {
        let deformed_position = device.create_buffer(&wgpu::BufferDescriptor{
            label: None,
            size: todo!(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false
        });
        Self { geometry, deformed_position, material, draw_buffers: OnceCell::new() }
    }

    /// Perform skinning to produce the immediate vertex buffer used for rendering.
    pub fn skin_with(&self, anim_texture: u32, frame: u32) {
        todo!()
    }

    pub fn get_material(&self) -> &PBRMaterial { &self.material }
}

impl Mesh for SkinnedMesh {
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer] {
        self.draw_buffers.get_or_init(|| {
            // Just swap out the position buffer is sufficient now.
            let mut buffers: Vec<wgpu::Buffer> = self.geometry.restpose_vertex_buffer.iter().cloned().collect();
            buffers[0] = self.deformed_position.clone();
            buffers
        })
    }
    fn get_vertex_draw_count(&self) -> u32 { self.geometry.vertex_draw_count }
    fn get_index_buffer(&self) -> Option<&wgpu::Buffer> { self.geometry.index_buffer.as_ref() }
    fn get_vertex_type(&self) -> VertexType { VertexType::Basic }
}

impl DrawableMesh for SkinnedMesh {
    fn get_model_matrix_bind_group(&self) -> &wgpu::BindGroup {
        todo!()
    }
    fn get_instance_range(&self) -> std::ops::Range<u32> { 0..1 }
    fn get_material(&self) -> &PBRMaterial { &self.material }
}
