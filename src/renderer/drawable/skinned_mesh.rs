use std::{num::NonZero, sync::Arc};

use crate::{
    asset::asset_types::vertex::{VertexBufferOthers, VertexBufferPosition}, renderer::{
        animation::{MorphAnimationFrame, SkinnedAnimationFrame}, drawable::{drawable_mesh_traits::{DrawableMesh, Mesh}, vertex_types::VertexType}, pipeline::{
            compute_library::{ComputeDispatchBindingSource, ComputePipelineKey},
            pbr_material::PBRMaterial,
        }, resources::mesh_layout::ModelMatrixBindingSource, upload::{buffer::upload_buffer, error::UploadError, gpu_mesh::GpuSkinnedMesh},
    },
};

/// Workgroup size declared by `shaders/gpu_skinning.wgsl`.
const SKINNING_WORKGROUP_SIZE: u32 = 64;

/// Per-instance GPU state of an animated mesh.
///
/// Every buffer here is written by the skinning compute shader, so it cannot be
/// shared between instances of the same mesh.
#[derive(Debug)]
struct AnimationState {
    /// Deformed vertex streams, in the same order as the rest pose streams.
    temp_buffer: Vec<wgpu::Buffer>,
    /// Bind group 1 of the skinning pipeline: rest pose in, deformed out, plus
    /// the joint indices and weights.
    ///
    /// Materialized by [`AnimatedMesh::ensure_skinning_bound`], because the
    /// dispatch layout belongs to the compute pipeline and does not exist yet
    /// when the mesh is created.
    skin_dispatch_bg: Option<wgpu::BindGroup>,

    /// Bind group 1 and 2 of the morphing pipeline.
    /// 
    /// Materialized by [`AnimatedMesh::ensure_morphing_bound`].
    morph_dispatch_bg: Option<[wgpu::BindGroup; 2]>
}

/// Animated mesh.
/// 
/// Skinned meshes store both a rest-pose reference mesh in `geometry`, and the
/// intermediate vertex attributes that are produced by the compute shader.
/// 
/// Contrary to static meshes, skinned meshes are *never* instanced: every
/// instance owns its own deformed vertex buffers and its own model matrix.
///
/// Both bind groups this type needs are deferred: the model-matrix group is
/// created by `ensure_bound`, and the skinning dispatch group by
/// `ensure_skinning_bound`. Mesh creation therefore needs no pipeline layout.
#[derive(Debug)]
pub struct AnimatedMesh {
    geometry: Arc<GpuSkinnedMesh>,
    morph_targets: Option<Arc<()>>,
    material: PBRMaterial,
    /// `None` when the skin has no clips: the mesh then renders its rest pose.
    animation: Option<AnimationState>,

    model_matrix_buffer: wgpu::Buffer,
    model_matrix_bg: Option<wgpu::BindGroup>,
}

impl AnimatedMesh {

    /// Create an animated mesh.
    /// 
    /// The mesh can be animated through skeleton skinning or vertex morphing.
    /// In case only vertex morphing is needed, the skeleton attributes (i.e
    /// bone indices and weights) are still needed. They may be filled with
    /// defaulted values a priori.
    ///
    /// `animated` must be true when the mesh has animation clip attached to it.
    /// This flag controls allocation of the per-instance deformed vertex
    /// buffers that the skinning compute pass writes.
    /// 
    /// The actual clip is selected by the
    /// [`crate::renderer::animation::AnimationManager`] at draw time, and this
    /// constructor does not keep any playback state.
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        geometry: Arc<GpuSkinnedMesh>,
        material: PBRMaterial,
        world_matrix: [[f32; 4]; 4],
        animated: bool,
    ) -> Result<Self, UploadError> {
        let model_matrix_buffer = upload_buffer(
            device,
            queue,
            "skinned mesh model matrix",
            wgpu::BufferUsages::STORAGE,
            world_matrix.as_slice(),
        )?;

        let skinning = animated.then(|| Self::create_skinning_state(device, &geometry));

        Ok(Self {
            geometry,
            morph_targets: None,
            material,
            animation: skinning,
            model_matrix_buffer,
            model_matrix_bg: None,
        })
    }

    /// Materialize the group-3 model-matrix bind group on first use.
    ///
    /// Idempotent. Must run before the mesh is drawn.
    /// 
    /// [`crate::renderer::drawable::draw_list::DrawList::ensure_bound`] is the
    /// normal entry point.
    pub(crate) fn ensure_bound(
        &mut self,
        device: &wgpu::Device,
        bindings: &impl ModelMatrixBindingSource,
    ) {
        if self.model_matrix_bg.is_some() {
            return;
        }
        self.model_matrix_bg = Some(bindings.create_model_matrix_bind_group(
            device,
            "skinned mesh model matrix bind group",
            &self.model_matrix_buffer,
        ));
    }

    /// Materialize the skinning dispatch bind group on first use.
    ///
    /// Idempotent. No-op for meshes without an animation. 
    /// 
    /// Must run before the skinning compute pass records [`Self::skin_with`].
    pub(crate) fn ensure_skinning_bound(
        &mut self,
        device: &wgpu::Device,
        bindings: &impl ComputeDispatchBindingSource,
    ) {
        let Some(skinning) = &mut self.animation else { return };
        if skinning.skin_dispatch_bg.is_some() {
            return;
        }

        let entries = [
            Self::storage_entry(0, &self.geometry.restpose_vertex_buffer[0]),
            Self::storage_entry(1, &self.geometry.restpose_vertex_buffer[1]),
            Self::storage_entry(2, &skinning.temp_buffer[0]),
            Self::storage_entry(3, &skinning.temp_buffer[1]),
            Self::storage_entry(4, &self.geometry.skinning_buffer),
        ];
        skinning.skin_dispatch_bg = Some(bindings.create_dispatch_bind_group(
            device,
            ComputePipelineKey::GpuSkinning,
            "skinning dispatch bind group",
            0,
            &entries,
        ));
    }

    /// Materialize the vertex morphing dispatch bind group on first use.
    ///
    /// Idempotent. No-op for meshes without an animation. 
    pub(crate) fn ensure_morphing_bound(
        &mut self,
        device: &wgpu::Device,
        bindings: &impl ComputeDispatchBindingSource
    ) {
        todo!()
    }

    /// Allocate the deformed buffers for one instance.
    ///
    /// Bind group layout creation is deferred.
    fn create_skinning_state(
        device: &wgpu::Device,
        geometry: &GpuSkinnedMesh,
    ) -> AnimationState {
        let vertex_count = u64::from(geometry.vertex_count);

        let deformed_position = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("skinned mesh deformed position"),
            size: vertex_count * std::mem::size_of::<VertexBufferPosition>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false
        });
        let deformed_attribute = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("skinned mesh deformed attribute"),
            size: vertex_count * std::mem::size_of::<VertexBufferOthers>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false
        });
        let temp_buffer = vec![deformed_position, deformed_attribute];

        AnimationState { temp_buffer, skin_dispatch_bg: None, morph_dispatch_bg: None }
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
    pub fn is_animated(&self) -> bool { self.animation.is_some() }

    pub fn morph_with(&self, cp: &mut wgpu::ComputePass, frame: MorphAnimationFrame) {
        let Some(anim) = &self.animation else { return };
        let dispatch_bg = anim.morph_dispatch_bg.as_ref().expect(
            "morph dispatch bind group is not initialized: call `DrawList::ensure_morphing_bound` before the morphing pass",
        );

        cp.set_bind_group(1, Some(&dispatch_bg[0]), &[]);
        cp.set_bind_group(2, &dispatch_bg[1], &[]);

        todo!("fill up push constants");

        cp.dispatch_workgroups(self.geometry.vertex_count.div_ceil(SKINNING_WORKGROUP_SIZE), 1, 1);
    }

    /// Record the skinning dispatch that produces the deformed vertex buffers.
    /// 
    /// This method assumes that the compute pass is set up correctly:
    /// - The skinning compute pipeline is bound;
    /// - Bindless bind group (group 0) is bound;
    /// - The per-mesh dispatch bind group has been materialized by
    ///   [`crate::renderer::drawable::draw_list::DrawList::ensure_skinning_bound`].
    ///
    /// It binds the per-mesh bind group (group 1), writes the clip texture and
    /// fractional frame supplied by the animation manager, and dispatches one
    /// workgroup per 64 vertices. Meshes without animation state are a no-op.
    pub fn skin_with(&self, cp: &mut wgpu::ComputePass, frame: SkinnedAnimationFrame) {
        let Some(skinning) = &self.animation else { return };
        let dispatch_bg = skinning.skin_dispatch_bg.as_ref().expect(
            "skinning dispatch bind group is not initialized: call `DrawList::ensure_skinning_bound` before the skinning pass",
        );

        cp.set_bind_group(1, Some(dispatch_bg), &[]);

        // Layout must stay in sync with `Immediate` in `gpu_skinning.wgsl`.
        let mut immediates = [0u8; 8];
        immediates[..4].copy_from_slice(&(frame.texture().index() as u32).to_ne_bytes());   // WGSL assumes host endianness
        immediates[4..].copy_from_slice(&frame.frame().to_le_bytes());
        cp.set_immediates(0, &immediates);

        cp.dispatch_workgroups(self.geometry.vertex_count.div_ceil(SKINNING_WORKGROUP_SIZE), 1, 1);
    }

    pub fn get_material(&self) -> &PBRMaterial { &self.material }
}

impl Mesh for AnimatedMesh {
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer] {
        match &self.animation {
            Some(skinning) => &skinning.temp_buffer,
            None => &self.geometry.restpose_vertex_buffer,
        }
    }
    fn get_vertex_draw_count(&self) -> u32 { self.geometry.vertex_draw_count }
    fn get_index_buffer(&self) -> Option<&wgpu::Buffer> { self.geometry.index_buffer.as_ref() }
    fn get_vertex_type(&self) -> VertexType { VertexType::Basic }
}

impl DrawableMesh for AnimatedMesh {
    fn get_model_matrix_bind_group(&self) -> &wgpu::BindGroup {
        self.model_matrix_bg.as_ref().expect(
            "model matrix bind group is not initialized: call `DrawList::ensure_bound` before drawing",
        )
    }
    fn get_instance_range(&self) -> std::ops::Range<u32> { 0..1 }
    fn get_material(&self) -> &PBRMaterial { &self.material }
}
