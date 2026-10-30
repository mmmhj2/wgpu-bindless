use std::{collections::HashMap, num::NonZero, sync::Arc};

use crate::{
    asset::asset_types::{AssetData, scene_asset::SceneAsset}, renderer::{
        drawable::{drawable_mesh_traits::{Mesh, DrawableMesh}, static_mesh::StaticMesh}, pipeline::pbr_material::PBRMaterial, resources::{asset_cache::MeshUploadContext, mesh_layout::MeshLayout}, scene::graph::SceneGraph,
    },
};

/// Size of one column-major `mat4x4<f32>` entry in the model-matrix buffer.
const MODEL_MATRIX_SIZE: u64 = std::mem::size_of::<[[f32; 4]; 4]>() as u64;

/// A single draw call covering one or more instances of the same static-mesh
/// primitive.
///
/// All instances share the geometry and material baked into the primitive's
/// [`StaticMesh`] asset. Their model matrices live in one read-only storage
/// buffer and are indexed by `@builtin(instance_index)`.
///
/// Only opaque and alpha-mask primitives may use this type. Alpha-blended
/// transparent instances must not be grouped into a batch and must be sorted
/// independently.
#[derive(Debug, Clone)]
pub struct StaticMeshBatch {
    mesh: Arc<StaticMesh>,
    // Owned explicitly so the buffer outlives its bind group. The bind group
    // also keeps the resource alive, but keeping the handle makes ownership
    // and future per-frame updates straightforward.
    #[allow(dead_code)]
    model_matrix_buffer: wgpu::Buffer,
    model_matrix_bind_group: wgpu::BindGroup,
    instance_count: u32,
}

impl StaticMeshBatch {
    fn generate_bind_group(
        device: &wgpu::Device,
        mmgr: &MeshLayout,
        buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model matrix batch bind group"),
            layout: mmgr.get_bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: 0,
                        size: NonZero::new(buffer.size()),
                    }),
                },
            ],
        })
    }

    /// Create a batch from explicit world matrices.
    ///
    /// Matrices are column-major, matching both `cgmath` and WGSL. They must
    /// not be empty. Callers are responsible for only passing opaque or
    /// alpha-mask instances; alpha-blended instances must use a separate
    /// transparent path.
    ///
    /// Panics if `world_matrices` is empty, if the instance count or buffer size
    /// exceeds device limits, or if the mesh asset is missing / not a static
    /// mesh asset.
    pub fn create_from_matrices(
        ctx: &mut MeshUploadContext<'_>,
        mesh_subasset_name: &str,
        world_matrices: &[[[f32; 4]; 4]],
    ) -> Self {
        assert!(
            !world_matrices.is_empty(),
            "cannot create a static mesh batch with zero instances for '{mesh_subasset_name}'"
        );

        let instance_count = u32::try_from(world_matrices.len())
            .expect("static mesh batch instance count exceeds u32::MAX");
        let buffer_size = MODEL_MATRIX_SIZE * world_matrices.len() as u64;
        let max_binding_size = u64::from(ctx.device.limits().max_storage_buffer_binding_size);
        assert!(
            buffer_size <= max_binding_size,
            "static mesh batch for '{mesh_subasset_name}' requires {buffer_size} bytes, \
             exceeding max_storage_buffer_binding_size ({max_binding_size})"
        );

        let asset = ctx
            .assets
            .get_asset(mesh_subasset_name)
            .unwrap_or_else(|| panic!("static mesh asset '{mesh_subasset_name}' not found."));
        let asset = asset
            .read()
            .expect("failed to acquire read lock for static mesh asset.");
        let mesh_asset = match asset.get_data() {
            AssetData::StaticMeshAssetType(m) => m,
            _ => panic!("asset '{mesh_subasset_name}' is not a static mesh asset."),
        };

        let model_matrix_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("model matrix batch buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let model_matrix_bind_group =
            Self::generate_bind_group(ctx.device, ctx.mesh_layout, &model_matrix_buffer);
        ctx.queue.write_buffer(
            &model_matrix_buffer,
            0,
            bytemuck::cast_slice(world_matrices),
        );

        Self {
            mesh: ctx.static_mesh_from_asset(mesh_subasset_name, mesh_asset),
            model_matrix_buffer,
            model_matrix_bind_group,
            instance_count,
        }
    }

    /// Number of instances covered by this batch. Always at least one.
    pub fn instance_count(&self) -> u32 {
        self.instance_count
    }
}

impl Mesh for StaticMeshBatch {
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer] {
        self.mesh.get_vertex_buffer()
    }
    fn get_vertex_draw_count(&self) -> u32 {
        self.mesh.get_vertex_draw_count()
    }
    fn get_index_buffer(&self) -> Option<&wgpu::Buffer> {
        self.mesh.get_index_buffer()
    }
    fn get_vertex_type(&self) -> super::vertex_types::VertexType {
        self.mesh.get_vertex_type()
    }
}

impl DrawableMesh for StaticMeshBatch {
    fn get_material(&self) -> &PBRMaterial {
        self.mesh.get_material()
    }
    fn get_model_matrix_bind_group(&self) -> &wgpu::BindGroup {
        &self.model_matrix_bind_group
    }
    fn get_instance_range(&self) -> std::ops::Range<u32> {
        0..self.instance_count
    }
}
