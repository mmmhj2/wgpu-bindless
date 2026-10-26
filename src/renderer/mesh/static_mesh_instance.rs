use std::{num::NonZero, sync::Arc};
use crate::{
    asset::{asset_manager::AssetManager, asset_types::{AssetData, scene_asset::SceneAsset}},
    renderer::{mesh::{Mesh, drawable_mesh_traits::DrawableMesh, mesh_manager::MeshManager, static_mesh::StaticMesh}, pipeline::{pbr_material::PBRMaterial, pipeline_state::PipelineStates}}
};

/// Instances of instanced mesh.
/// Holds unique data for each instance such as model matrix, and a reference to the underlying mesh.
#[derive(Clone)]
pub struct StaticMeshInstance {
    mesh            : Arc<StaticMesh>,
    model_matrix    : wgpu::Buffer,
    model_matrix_bg : wgpu::BindGroup
}

impl StaticMeshInstance {
    fn generate_bind_group(device: &wgpu::Device, mmgr: &MeshManager, buffer: &wgpu::Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("model matrix buffer bind group"),
            layout: mmgr.get_bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(
                        wgpu::BufferBinding { buffer, offset: 0, size: NonZero::new(buffer.size()) }
                    )
                }
            ]
        })
    }

    /// Create instances from an imported scene asset.
    ///
    /// The scene asset is typically imported via the asset importer, and
    /// contains a list of mesh instances with their world matrices.
    /// 
    /// The model matrix buffers for each instance will be created and uploaded to the GPU.
    /// If the asset has not been uploaded to the GPU, it will be uploaded via
    /// [`static_mesh_from_asset()`] during this process.
    pub fn create_from_scene_asset(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline_states: &mut PipelineStates,
        scene: &SceneAsset,
        assets: &AssetManager
    ) -> Vec<Self> {
        let mut ret = Vec::new();

        for instance in &scene.instances {
            let name = instance.mesh_subasset_name.as_str();

            let asset = assets.get_asset(name).unwrap_or_else(|| panic!("static mesh asset '{name}' not found."));
            let asset = asset.read().expect("failed to acquire read lock for static mesh asset.");
            let mesh_asset = match asset.get_data() {
                AssetData::StaticMeshAssetType(m) => m,
                _ => panic!("asset '{name}' is not a static mesh asset."),
            };

            let buffer = device.create_buffer(&wgpu::BufferDescriptor{
                label: Some("model matrix buffer"),
                size: (std::mem::size_of::<f32>() * 16) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = Self::generate_bind_group(device, pipeline_states.get_mesh_manager(), &buffer);
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&instance.world_matrix));

            ret.push(
                Self{
                    mesh: pipeline_states.get_asset_registry_mut().static_mesh_from_asset(device, queue, assets, name, mesh_asset),
                    model_matrix: buffer,
                    model_matrix_bg: bg
                }
            );
        }

        return ret;
    }
}

impl Mesh for StaticMeshInstance {
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer] { self.mesh.get_vertex_buffer() }
    fn get_vertex_draw_count(&self) -> u32 { self.mesh.get_vertex_draw_count() }
    fn get_index_buffer(&self) -> Option<&wgpu::Buffer> { self.mesh.get_index_buffer() }
    fn get_vertex_type(&self) -> super::vertex_types::VertexType { self.mesh.get_vertex_type() }
}

impl DrawableMesh for StaticMeshInstance {
    fn get_material(&self) -> &PBRMaterial { self.mesh.get_material() }
    fn get_model_matrix_bind_group(&self) -> &wgpu::BindGroup { &self.model_matrix_bg }
    fn get_instance_range(&self) -> std::ops::Range<u32> { 0..1 }
}
