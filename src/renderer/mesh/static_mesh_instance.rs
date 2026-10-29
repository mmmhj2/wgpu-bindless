use std::{num::NonZero, sync::Arc};

use crate::{
    asset::asset_types::{AssetData, scene_asset::SceneAsset},
    renderer::{
        mesh::{Mesh, drawable_mesh_traits::DrawableMesh, static_mesh::StaticMesh},
        pipeline::pbr_material::PBRMaterial,
        resources::{
            asset_cache::MeshUploadContext,
            mesh_layout::MeshLayout,
        },
        scene::graph::SceneGraph,
    },
};

/// One renderable instance of a static mesh.
///
/// Each instance owns a single model-matrix buffer.
/// 
/// TODO: Implement proper instancing
#[derive(Clone)]
pub struct StaticMeshInstance {
    mesh            : Arc<StaticMesh>,
    model_matrix    : wgpu::Buffer,
    model_matrix_bg : wgpu::BindGroup
}

impl StaticMeshInstance {
    fn generate_bind_group(device: &wgpu::Device, mmgr: &MeshLayout, buffer: &wgpu::Buffer) -> wgpu::BindGroup {
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
    /// Convenience wrapper that builds a read-only [`SceneGraph`] and delegates
    /// to [`StaticMeshInstance::create_from_scene_graph`]. Callers that need to
    /// inspect or retain the hierarchy should build the graph themselves.
    pub fn create_from_scene_asset(
        ctx: &mut MeshUploadContext<'_>,
        scene: &SceneAsset,
    ) -> Vec<Self> {
        let graph = SceneGraph::from_asset(scene);
        Self::create_from_scene_graph(ctx, &graph)
    }

    /// Create one instance per mesh primitive in `graph`.
    ///
    /// The model matrix buffer for each instance is created and uploaded here.
    /// If a mesh has not been uploaded to the GPU yet, it is uploaded via
    /// [`MeshUploadContext::static_mesh_from_asset()`] during this process.
    pub fn create_from_scene_graph(
        ctx: &mut MeshUploadContext<'_>,
        graph: &SceneGraph,
    ) -> Vec<Self> {
        let mut ret = Vec::with_capacity(graph.mesh_instance_count());

        for instance in graph.mesh_instances() {
            let name = instance.mesh_subasset_name;

            let asset = ctx.assets.get_asset(name).unwrap_or_else(|| panic!("static mesh asset '{name}' not found."));
            let asset = asset.read().expect("failed to acquire read lock for static mesh asset.");
            let mesh_asset = match asset.get_data() {
                AssetData::StaticMeshAssetType(m) => m,
                _ => panic!("asset '{name}' is not a static mesh asset."),
            };

            let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor{
                label: Some("model matrix buffer"),
                size: std::mem::size_of::<[[f32; 4]; 4]>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = Self::generate_bind_group(ctx.device, ctx.mesh_layout, &buffer);
            ctx.queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&instance.world_matrix));

            ret.push(
                Self{
                    mesh: ctx.static_mesh_from_asset(name, mesh_asset),
                    model_matrix: buffer,
                    model_matrix_bg: bg
                }
            );
        }

        ret
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
