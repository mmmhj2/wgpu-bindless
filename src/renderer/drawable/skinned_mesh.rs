use std::cell::OnceCell;

use crate::{asset::asset_types::skinned_mesh_asset::GpuSkinnedMesh, renderer::{drawable::{drawable_mesh_traits::{DrawableMesh, Mesh}, vertex_types::VertexType}, pipeline::pbr_material::PBRMaterial, resources::asset_cache::MeshUploadContext}};

/// Skinned animated mesh.
/// 
/// Skinned meshes store both a rest-pose reference mesh in `geometry`, and the
/// intermediate vertex attributes that are produced by the compute shader.
/// 
/// Contrary to static meshes, skinned meshes are *never* instanced.
#[derive(Debug)]
pub struct SkinnedMesh {
    geometry: GpuSkinnedMesh,
    temp_buffer: Vec<wgpu::Buffer>,
    material: PBRMaterial,

    model_matrix_buffer: wgpu::Buffer,
    model_matrix_bg: wgpu::BindGroup,
    skinning_dispatch_bg: wgpu::BindGroup
}

impl SkinnedMesh {

    pub(crate) fn new(ctx: &mut MeshUploadContext<'_>, geometry: GpuSkinnedMesh, material: PBRMaterial) -> Self {
        let deformed_position = ctx.device.create_buffer(&wgpu::BufferDescriptor{
            label: None,
            size: todo!(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false
        });
        let deformed_attribute = ctx.device.create_buffer(&wgpu::BufferDescriptor{
            label: None,
            size: todo!(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false
        });
        let temp_buffer = vec![deformed_position, deformed_attribute];

        Self {
            geometry,
            temp_buffer,
            material,
            model_matrix_buffer: todo!(),
            model_matrix_bg: todo!(),
            skinning_dispatch_bg: todo!()
        }
    }

    /// Perform skinning to produce the immediate vertex buffer used for rendering.
    /// 
    /// This method assumes that the compute pass is set up correctly:
    /// - Bindless bind group (group 0) is correctly bound to the pipeline;
    /// - Corresponding compute pipeline is correctly setup.
    /// 
    /// This method will mutate the first bind group for input to the compute shader.
    /// 
    /// TODO: revisit this API design after Skinning animation is actually running.
    pub fn skin_with(&self, cp: &mut wgpu::ComputePass, anim_texture: u32, frame: u32) {
        todo!()
    }

    pub fn get_material(&self) -> &PBRMaterial { &self.material }
}

impl Mesh for SkinnedMesh {
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer] { &self.temp_buffer }
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
