use bytemuck::Zeroable;

use crate::{
    asset::asset_types::{skinned_mesh_asset::SkinnedMeshAsset, static_mesh_asset::StaticMeshAsset, vertex::VertexBufferSkinning}, renderer::upload::{
        GpuSkinnedMesh, context::UploadContext, error::UploadError, gpu_mesh::GpuStaticMesh, gpu_type::GpuAssetType,
    },
};

impl GpuAssetType for StaticMeshAsset {
    type GpuObjectType = GpuStaticMesh;

    /// Upload the vertex and index buffers onto the GPU device.
    ///
    /// Vertex data is written directly from the asset's own slices, so no
    /// intermediate copy of the whole vertex buffer is made. The vertex data is
    /// expected to be final (see [`StaticMeshAsset::recondition`]), and the
    /// asset will not be further altered.
    fn upload(
        &self,
        name: &str,
        ctx: &mut UploadContext<'_>,
    ) -> Result<Self::GpuObjectType, UploadError> {
        if self.vp.len() != self.va.len() {
            return Err(UploadError::VertexCountMismatch {
                name: name.to_string(),
                positions: self.vp.len(),
                attributes: self.va.len(),
            });
        }

        let position_buffer = ctx.upload_buffer(
            &format!("{name}.position"),
            wgpu::BufferUsages::VERTEX,
            self.vp.as_slice(),
        )?;
        let attribute_buffer = ctx.upload_buffer(
            &format!("{name}.attribute"),
            wgpu::BufferUsages::VERTEX,
            self.va.as_slice(),
        )?;

        let index_buffer = self
            .vi
            .as_ref()
            .map(|indices| {
                ctx.upload_buffer(
                    &format!("{name}.index"),
                    wgpu::BufferUsages::INDEX,
                    indices.as_slice(),
                )
            })
            .transpose()?;

        Ok(GpuStaticMesh {
            vertex_buffers: vec![position_buffer, attribute_buffer],
            index_buffer,
            vertex_draw_count: self.vertex_draw_count,
        })
    }
}

impl GpuAssetType for SkinnedMeshAsset {
    type GpuObjectType = GpuSkinnedMesh;

    /// Upload the rest-pose vertex buffers and the skinning attributes.
    ///
    /// In case of an animated mesh, each instance of the mesh must allocate
    /// separate buffers to store the intermediate vertex attributes calculated
    /// by the compute shader. Those are managed by [`SkinnedMesh`] instead.
    ///
    /// Rest pose vertex buffers are additionally flagged as `STORAGE` so the
    /// skinning compute shader can read them as typed storage buffers.
    /// 
    /// [`SkinnedMesh`]: crate::renderer::drawable::skinned_mesh::SkinnedMesh
    fn upload(
        &self,
        name: &str,
        ctx: &mut UploadContext<'_>,
    ) -> Result<Self::GpuObjectType, UploadError> {
        let restpose = &self.restpose;

        if restpose.vp.len() != restpose.va.len() {
            return Err(UploadError::VertexCountMismatch {
                name: name.to_string(),
                positions: restpose.vp.len(),
                attributes: restpose.va.len(),
            });
        }
        // An empty skinning stream means a morph-only mesh, which never
        // dispatches the skinning pass. Only a mesh that does claim a skin has to
        // carry one entry per vertex.
        // TODO: clean up morph and skinning entanglement after all features can run.
        if !self.animation.is_empty() && restpose.vp.len() != self.animation.len() {
            return Err(UploadError::SkinningCountMismatch {
                name: name.to_string(),
                vertices: restpose.vp.len(),
                skinning: self.animation.len(),
            });
        }

        let position_buffer = ctx.upload_buffer(
            &format!("{name}.restpose.position"),
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            restpose.vp.as_slice(),
        )?;
        let attribute_buffer = ctx.upload_buffer(
            &format!("{name}.restpose.attribute"),
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            restpose.va.as_slice(),
        )?;

        // A zero-sized buffer cannot be created. Fall back to a single zeroed
        // entry so the bind group stays valid for malformed assets.
        let skinning_label = format!("{name}.skinning");
        let skinning_buffer = if self.animation.is_empty() {
            if self.skin.is_some() {
                log::warn!("Missing armature data for skinned mesh {name}. The asset may be malformed.");
            } else {
                // Morph-only meshes still bind this buffer to satisfy the shared
                // bind group layout. The skinning shader never reads it.
                log::info!("Mesh {name} has no skinning attributes; it is a morph-only mesh.");
            }
            ctx.upload_buffer(
                &skinning_label,
                wgpu::BufferUsages::STORAGE,
                &[VertexBufferSkinning::zeroed()],
            )?
        } else {
            ctx.upload_buffer(
                &skinning_label,
                wgpu::BufferUsages::STORAGE,
                self.animation.as_slice(),
            )?
        };

        let index_buffer = restpose
            .vi
            .as_ref()
            .map(|indices| {
                ctx.upload_buffer(
                    &format!("{name}.index"),
                    wgpu::BufferUsages::INDEX,
                    indices.as_slice(),
                )
            })
            .transpose()?;

        Ok(GpuSkinnedMesh {
            restpose_vertex_buffer: vec![position_buffer, attribute_buffer],
            skinning_buffer,
            index_buffer,
            vertex_count: restpose.vp.len() as u32,
            vertex_draw_count: restpose.vertex_draw_count,
        })
    }
}
