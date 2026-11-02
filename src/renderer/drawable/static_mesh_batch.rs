use std::{num::NonZero, sync::Arc};

use crate::renderer::{
    drawable::{
        drawable_mesh_traits::{DrawableMesh, Mesh},
        static_mesh::StaticMesh,
    },
    pipeline::pbr_material::PBRMaterial,
    resources::mesh_layout::MeshLayout,
    upload::{buffer::upload_buffer, error::UploadError},
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

    /// Create a batch from an already-uploaded mesh and explicit world matrices.
    ///
    /// Matrices are column-major, matching both `cgmath` and WGSL. They must
    /// not be empty. 
    /// 
    /// TODO: figure out how to handle alpha-blended instances, which must use a
    /// separate transparent path.
    ///
    /// Errors
    /// ---
    ///
    /// [`UploadError::EmptyBuffer`] for an empty instance list,
    /// [`UploadError::StorageBindingTooLarge`] when the model-matrix buffer would
    /// exceed `max_storage_buffer_binding_size`, and [`UploadError::BufferTooLarge`]
    /// for the corresponding device-wide limit.
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh_layout: &MeshLayout,
        mesh: Arc<StaticMesh>,
        world_matrices: &[[[f32; 4]; 4]],
    ) -> Result<Self, UploadError> {
        let instance_count = u32::try_from(world_matrices.len())
            .map_err(|_| UploadError::TooManyInstances { count: world_matrices.len() })?;
        let buffer_size = MODEL_MATRIX_SIZE * world_matrices.len() as u64;
        let max_binding_size = u64::from(device.limits().max_storage_buffer_binding_size);
        if buffer_size > max_binding_size {
            return Err(UploadError::StorageBindingTooLarge {
                label: "model matrix batch buffer".to_string(),
                needed: buffer_size,
                limit: max_binding_size,
            });
        }

        let model_matrix_buffer = upload_buffer(
            device,
            queue,
            "model matrix batch buffer",
            wgpu::BufferUsages::STORAGE,
            world_matrices,
        )?;
        let model_matrix_bind_group =
            Self::generate_bind_group(device, mesh_layout, &model_matrix_buffer);

        Ok(Self {
            mesh,
            model_matrix_buffer,
            model_matrix_bind_group,
            instance_count,
        })
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
