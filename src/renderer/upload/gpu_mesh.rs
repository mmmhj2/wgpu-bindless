/// GPU-resident geometry produced by uploading a
/// [`crate::asset::asset_types::static_mesh_asset::StaticMeshAsset`].
///
/// The vertex buffers are ordered to match the renderer's vertex buffer layouts
/// (`renderer::drawable::vertex_types`): index 0 is the position stream, index 1
/// is the attribute stream.
#[derive(Debug)]
pub struct GpuStaticMesh {
    pub vertex_buffers: Vec<wgpu::Buffer>,
    pub index_buffer: Option<wgpu::Buffer>,
    pub vertex_draw_count: u32,
}

/// GPU-resident geometry produced by uploading a
/// [`crate::asset::asset_types::skinned_mesh_asset::SkinnedMeshAsset`].
///
/// The `restpose_vertex_buffer` streams are fed to the skinning compute shader and
/// then consumed by the rasterizer, so they carry storage (UAV) and vertex usages.
#[derive(Debug)]
pub struct GpuSkinnedMesh {
    /// Rest pose vertex streams, ordered like the renderer's vertex buffer
    /// layouts (position, then attributes).
    pub restpose_vertex_buffer: Vec<wgpu::Buffer>,
    /// Joint indices and weights, one
    /// [`crate::asset::asset_types::vertex::VertexBufferSkinning`] per vertex.
    pub skinning_buffer: wgpu::Buffer,
    pub index_buffer: Option<wgpu::Buffer>,
    /// Number of vertices in the rest pose. Drives the compute dispatch size,
    /// which must not be confused with `vertex_draw_count` (an index count).
    pub vertex_count: u32,
    pub vertex_draw_count: u32,
}
