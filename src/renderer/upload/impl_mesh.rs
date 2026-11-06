use crate::{
    asset::asset_types::{skinned_mesh_asset::SkinnedMeshAsset, static_mesh_asset::StaticMeshAsset}, renderer::upload::{
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
    /// by the compute shader. Those are managed by [`AnimatedMesh`] instead.
    ///
    /// Rest pose vertex buffers are additionally flagged as `STORAGE` so the
    /// skinning compute shader can read them as typed storage buffers.
    /// 
    /// [`AnimatedMesh`]: crate::renderer::drawable::skinned_mesh::AnimatedMesh
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
        // A morph-only mesh carries no skinning stream at all. A mesh that does
        // carry one must have exactly one entry per rest-pose vertex.
        if let Some(skinning) = &self.skinning {
            if restpose.vp.len() != skinning.len() {
                return Err(UploadError::SkinningCountMismatch {
                    name: name.to_string(),
                    vertices: restpose.vp.len(),
                    skinning: skinning.len(),
                });
            }
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

        // A morph-only mesh has no skinning stream and no buffer to bind.
        let skinning_buffer = match &self.skinning {
            Some(skinning) => Some(ctx.upload_buffer(
                &format!("{name}.skinning"),
                wgpu::BufferUsages::STORAGE,
                skinning.as_slice(),
            )?),
            None => {
                if self.skin.is_some() {
                    log::warn!(
                        "Skinned mesh {name} has no skinning attributes; the asset is malformed and skinning is disabled."
                    );
                } else {
                    log::info!("Mesh {name} has no skinning attributes; it is a morph-only mesh.");
                }
                None
            }
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
