use std::num::NonZero;

use crate::{
    asset::asset_types::animation_asset::{MorphTarget, MAX_MORPH_TARGETS},
    renderer::resources::bindless::SampledTextureHandle,
};

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

/// Number of morph target channels the GPU layout reserves.
///
/// Alias of [`MAX_MORPH_TARGETS`] under the upload-side name.
pub const MAX_MORPH_CHANNELS: usize = MAX_MORPH_TARGETS;

/// Number of `u32` words in the morph layout buffer.
///
/// `[0]` channel count, `[1..9]` per-channel attribute masks, `[9..49]`
/// per-channel, per-attribute offsets into [`GpuMorphTargets::deltas`].
pub const MORPH_LAYOUT_WORDS: usize = 1 + MAX_MORPH_TARGETS + MAX_MORPH_TARGETS * MorphTarget::CHANNEL_COUNT;

/// GPU-resident morph targets of one primitive.
///
/// One asset produces one of these, shared by every instance of the mesh. The
/// layout and the deltas are immutable, only the weights vary per draw.
///
/// The layout is data-driven so that adding an attribute never changes the
/// bindings or the immediate encoding. Masks say which attributes a channel
/// carries, offsets say where its block starts in [`Self::deltas`].
#[derive(Debug)]
pub struct GpuMorphTargets {
    /// `[u32; MORPH_LAYOUT_WORDS]`: `[0]` channel count, `[1..9]` masks,
    /// `[9..49]` offsets (unit: `f32` index into [`Self::deltas`]).
    pub layout: wgpu::Buffer,
    /// Attribute deltas of every channel, packed back to back.
    ///
    /// Buffer should never be zero sized.
    pub deltas: wgpu::Buffer,
    /// One-row weight texture holding the mesh's static `mesh.weights`.
    ///
    /// `None` when the base weights are absent or all zero, i.e. when the mesh
    /// has no static deformation.
    pub base_weights: Option<SampledTextureHandle>,
    /// Number of morph targets, i.e. the number of weights a clip provides.
    pub channel_count: u32,
    /// Number of weight channels [`Self::base_weights`] provides.
    ///
    /// `None` when the texture does not exist, otherwise the target count.
    pub base_channels: Option<NonZero<u32>>,
}

