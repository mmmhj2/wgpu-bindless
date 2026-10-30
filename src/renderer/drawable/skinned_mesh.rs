use std::{num::NonZero, sync::Arc};

use crate::{
    asset::asset_types::{skinned_mesh_asset::GpuSkinnedMesh, vertex::{VertexBufferOthers, VertexBufferPosition}},
    renderer::{
        drawable::{drawable_mesh_traits::{DrawableMesh, Mesh}, vertex_types::VertexType},
        pipeline::{compute_library::ComputePipelineKey, pbr_material::PBRMaterial},
        resources::{asset_cache::MeshUploadContext, bindless::SampledTextureHandle},
    },
};

/// Workgroup size declared by `shaders/gpu_skinning.wgsl`.
const SKINNING_WORKGROUP_SIZE: u32 = 64;

/// Size of one column-major `mat4x4<f32>` entry in the model-matrix buffer.
const MODEL_MATRIX_SIZE: u64 = std::mem::size_of::<[[f32; 4]; 4]>() as u64;

/// Playback information of the baked animation bound to a skinned mesh.
#[derive(Debug, Clone, Copy)]
pub struct SkinnedAnimation {
    /// Bindless index of the baked joint-matrix texture.
    pub texture: SampledTextureHandle,
    /// Number of rows in that texture.
    pub frames: u32,
    /// Rate the rows were baked at.
    pub fps: f32,
}

impl SkinnedAnimation {
    /// Resolve the frame index for a playback time, wrapping at the end.
    pub fn frame_at(&self, time_seconds: f32) -> u32 {
        if self.fps <= 0.0 {
            return 0;
        }
        let frames = self.frames.max(1);
        ((time_seconds.max(0.0) * self.fps) as u32) % frames
    }
}

/// Per-instance GPU state of an animated mesh.
///
/// Every buffer here is written by the skinning compute shader, so it cannot be
/// shared between instances of the same mesh.
#[derive(Debug)]
struct SkinningState {
    /// Deformed vertex streams, in the same order as the rest pose streams.
    temp_buffer: Vec<wgpu::Buffer>,
    /// Bind group 1 of the skinning pipeline: rest pose in, deformed out, plus
    /// the joint indices and weights. Immutable for the mesh's lifetime.
    dispatch_bg: wgpu::BindGroup,
    animation: SkinnedAnimation,
}

/// Skinned animated mesh.
/// 
/// Skinned meshes store both a rest-pose reference mesh in `geometry`, and the
/// intermediate vertex attributes that are produced by the compute shader.
/// 
/// Contrary to static meshes, skinned meshes are *never* instanced: every
/// instance owns its own deformed vertex buffers and its own model matrix.
#[derive(Debug)]
pub struct SkinnedMesh {
    geometry: Arc<GpuSkinnedMesh>,
    material: PBRMaterial,
    /// `None` when no animation is bound: the mesh then renders its rest pose.
    skinning: Option<SkinningState>,

    #[allow(dead_code)]
    model_matrix_buffer: wgpu::Buffer,
    model_matrix_bg: wgpu::BindGroup,
}

impl SkinnedMesh {

    /// Create a skinned mesh with possible animation attached.
    /// 
    /// TODO: figure out a way of re-attaching a new animation for a skinned mesh.
    pub(crate) fn new(
        ctx: &mut MeshUploadContext<'_>,
        geometry: Arc<GpuSkinnedMesh>,
        material: PBRMaterial,
        world_matrix: [[f32; 4]; 4],
        animation: Option<SkinnedAnimation>,
    ) -> Self {
        let model_matrix_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("skinned mesh model matrix"),
            size: MODEL_MATRIX_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });
        ctx.queue.write_buffer(&model_matrix_buffer, 0, bytemuck::cast_slice(&world_matrix));
        let model_matrix_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("skinned mesh model matrix bind group"),
            layout: ctx.mesh_layout.get_bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &model_matrix_buffer,
                        offset: 0,
                        size: NonZero::new(model_matrix_buffer.size()),
                    }),
                },
            ],
        });

        let skinning = animation.map(|animation| Self::create_skinning_state(ctx, &geometry, animation));

        Self {
            geometry,
            material,
            skinning,
            model_matrix_buffer,
            model_matrix_bg,
        }
    }

    /// Allocate the deformed buffers and the per-mesh compute bind group.
    fn create_skinning_state(
        ctx: &mut MeshUploadContext<'_>,
        geometry: &GpuSkinnedMesh,
        animation: SkinnedAnimation,
    ) -> SkinningState {
        let vertex_count = u64::from(geometry.vertex_count);

        let deformed_position = ctx.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("skinned mesh deformed position"),
            size: vertex_count * std::mem::size_of::<VertexBufferPosition>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false
        });
        let deformed_attribute = ctx.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("skinned mesh deformed attribute"),
            size: vertex_count * std::mem::size_of::<VertexBufferOthers>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false
        });
        let temp_buffer = vec![deformed_position, deformed_attribute];

        let skinning_layout = ctx.compute_pipeline
            .get_dispatch_bind_group_layout(ComputePipelineKey::GpuSkinning)
            .expect("the GPU skinning compute pipeline should be registered");
        let dispatch_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("skinning dispatch bind group"),
            layout: skinning_layout,
            entries: &[
                Self::storage_entry(0, &geometry.restpose_vertex_buffer[0]),
                Self::storage_entry(1, &geometry.restpose_vertex_buffer[1]),
                Self::storage_entry(2, &temp_buffer[0]),
                Self::storage_entry(3, &temp_buffer[1]),
                Self::storage_entry(4, &geometry.skinning_buffer),
            ],
        });

        SkinningState { temp_buffer, dispatch_bg, animation }
    }

    fn storage_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: NonZero::new(buffer.size()),
            }),
        }
    }

    /// Whether this mesh is driven by an animation.
    ///
    /// A skinned mesh without an animation renders its rest pose and skips the
    /// compute dispatch entirely.
    pub fn is_animated(&self) -> bool { self.skinning.is_some() }

    /// Record the skinning dispatch that produces the deformed vertex buffers.
    /// 
    /// This method assumes that the compute pass is set up correctly:
    /// - The skinning compute pipeline is bound;
    /// - Bindless bind group (group 0) is bound.
    ///
    /// It resolves the animation frame from `time_seconds`, binds the per-mesh
    /// bind group (group 1), sets the immediates and dispatches one workgroup per
    /// 64 vertices. Meshes without an animation are a no-op.
    ///
    /// TODO: revisit this API design after Skinning animation is actually running.
    pub fn skin_with(&self, cp: &mut wgpu::ComputePass, time_seconds: f32) {
        let Some(skinning) = &self.skinning else { return };

        cp.set_bind_group(1, Some(&skinning.dispatch_bg), &[]);

        // Layout must stay in sync with `Immediate` in `gpu_skinning.wgsl`.
        let mut immediates = [0u8; 8];
        immediates[..4].copy_from_slice(&(skinning.animation.texture.index() as u32).to_le_bytes());
        immediates[4..].copy_from_slice(&skinning.animation.frame_at(time_seconds).to_le_bytes());
        cp.set_immediates(0, &immediates);

        cp.dispatch_workgroups(self.geometry.vertex_count.div_ceil(SKINNING_WORKGROUP_SIZE), 1, 1);
    }

    pub fn get_material(&self) -> &PBRMaterial { &self.material }
}

impl Mesh for SkinnedMesh {
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer] {
        match &self.skinning {
            Some(skinning) => &skinning.temp_buffer,
            None => &self.geometry.restpose_vertex_buffer,
        }
    }
    fn get_vertex_draw_count(&self) -> u32 { self.geometry.vertex_draw_count }
    fn get_index_buffer(&self) -> Option<&wgpu::Buffer> { self.geometry.index_buffer.as_ref() }
    fn get_vertex_type(&self) -> VertexType { VertexType::Basic }
}

impl DrawableMesh for SkinnedMesh {
    fn get_model_matrix_bind_group(&self) -> &wgpu::BindGroup { &self.model_matrix_bg }
    fn get_instance_range(&self) -> std::ops::Range<u32> { 0..1 }
    fn get_material(&self) -> &PBRMaterial { &self.material }
}
