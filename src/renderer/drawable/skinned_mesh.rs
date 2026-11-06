use std::{num::NonZero, sync::Arc};

use crate::{
    asset::asset_types::vertex::{VertexBufferOthers, VertexBufferPosition}, renderer::{
        animation::{AnimationTracks, MorphAnimationFrame, SkinnedAnimationFrame},
        drawable::{drawable_mesh_traits::{DrawableMesh, Mesh}, vertex_types::VertexType},
        pipeline::{
            compute_library::{ComputeDispatchBindingSource, ComputePipelineKey, ComputePipelineLibrary},
            pbr_material::PBRMaterial,
        },
        resources::{
            mesh_layout::ModelMatrixBindingSource,
            rig_plan::{FinalBuffer, RigStages},
        },
        upload::{
            buffer::upload_buffer,
            error::UploadError,
            gpu_mesh::{GpuMorphTargets, GpuSkinnedMesh},
        },
    },
};

/// Workgroup size declared by both deformation compute shaders.
const DEFORMATION_WORKGROUP_SIZE: u32 = 64;

/// Per-instance GPU state of an animated mesh.
///
/// Every buffer here is written by a deformation compute shader, so it cannot be
/// shared between instances of the same mesh.
#[derive(Debug)]
struct AnimationState {
    /// Morphing output streams, in the same order as the rest pose streams.
    morph_out: Option<[wgpu::Buffer; 2]>,
    /// Skinning output streams, in the same order as the rest pose streams.
    skin_out: Option<[wgpu::Buffer; 2]>,

    /// Group 1 and 2 of the morphing pipeline.
    ///
    /// Materialized by [`AnimatedMesh::ensure_deformation_bound`], because the
    /// dispatch layouts belong to the compute pipelines and do not exist when
    /// the mesh is created.
    morph_dispatch_bg: Option<[wgpu::BindGroup; 2]>,
    /// Group 1 of the skinning pipeline.
    skin_dispatch_bg: Option<wgpu::BindGroup>,
}

/// Animated mesh.
///
/// Deformable meshes store both a rest-pose reference mesh in `geometry` and
/// the per-instance intermediate vertex streams produced by the deformation
/// compute shaders.
///
/// Contrary to static meshes, animated meshes are *never* instanced: every
/// instance owns its own deformed vertex buffers and its own model matrix.
///
/// Bind groups are deferred.  Mesh creation therefore needs no pipeline layout.
/// - the model-matrix group is created by [`Self::ensure_bound`] 
/// - the dispatch groups by [`Self::ensure_deformation_bound`].
#[derive(Debug)]
pub struct AnimatedMesh {
    geometry: Arc<GpuSkinnedMesh>,
    /// Stage set and final vertex stream, decided during `prepare_scene`.
    stages: RigStages,
    /// Morph targets of this mesh, shared with every instance.
    morph_targets: Option<Arc<GpuMorphTargets>>,
    material: PBRMaterial,
    /// `None` when the rig has no deformation stage.
    animation: Option<AnimationState>,

    model_matrix_buffer: wgpu::Buffer,
    model_matrix_bg: Option<wgpu::BindGroup>,
}

impl AnimatedMesh {

    /// Create an animated mesh.
    ///
    /// `stages` decides which deformation stages exist and which vertex stream
    /// is drawn. `morph_targets` must be present when the stages contain the
    /// morph stage, which is shared between every instance of the mesh.
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
        stages: RigStages,
        morph_targets: Option<Arc<GpuMorphTargets>>,
    ) -> Result<Self, UploadError> {
        let model_matrix_buffer = upload_buffer(
            device,
            queue,
            "skinned mesh model matrix",
            wgpu::BufferUsages::STORAGE,
            world_matrix.as_slice(),
        )?;

        let animation = Self::create_stages(device, &geometry, stages);

        Ok(Self {
            geometry,
            stages,
            morph_targets,
            material,
            animation,
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

    /// Materialize the deformation dispatch bind groups on first use.
    ///
    /// Idempotent. No-op for rigs without a deformation stage.
    ///
    /// Must run before the compute pass records [`Self::record_deformation`].
    pub(crate) fn ensure_deformation_bound(
        &mut self,
        device: &wgpu::Device,
        bindings: &impl ComputeDispatchBindingSource,
    ) {
        let Some(animation) = &mut self.animation else { return };

        if self.stages.morphs() && animation.morph_dispatch_bg.is_none() {
            let target = self
                .morph_targets
                .as_ref()
                .expect("a morphing plan requires a morph target asset");
            let morph_out = animation
                .morph_out
                .as_ref()
                .expect("a morphing plan allocates morph output buffers");

            let group1 = bindings.create_dispatch_bind_group(
                device,
                ComputePipelineKey::GpuVertexMorphing,
                "morphing dispatch bind group 1",
                0,
                &[
                    Self::storage_entry(0, &self.geometry.restpose_vertex_buffer[0]),
                    Self::storage_entry(1, &self.geometry.restpose_vertex_buffer[1]),
                    Self::storage_entry(2, &morph_out[0]),
                    Self::storage_entry(3, &morph_out[1]),
                ],
            );
            let group2 = bindings.create_dispatch_bind_group(
                device,
                ComputePipelineKey::GpuVertexMorphing,
                "morphing dispatch bind group 2",
                1,
                &[
                    Self::storage_entry(0, &target.layout),
                    Self::storage_entry(1, &target.deltas),
                ],
            );
            animation.morph_dispatch_bg = Some([group1, group2]);
        }

        if self.stages.skins() && animation.skin_dispatch_bg.is_none() {
            let skin_out = animation
                .skin_out
                .as_ref()
                .expect("a skinning plan allocates skin output buffers");
            let skinning_buffer = self
                .geometry
                .skinning_buffer
                .as_ref()
                .expect("a skinning plan requires per-vertex skinning attributes");
            let (input_position, input_attribute) = match &animation.morph_out {
                Some(morph_out) => (&morph_out[0], &morph_out[1]),
                None => (&self.geometry.restpose_vertex_buffer[0], &self.geometry.restpose_vertex_buffer[1]),
            };

            animation.skin_dispatch_bg = Some(bindings.create_dispatch_bind_group(
                device,
                ComputePipelineKey::GpuSkinning,
                "skinning dispatch bind group",
                0,
                &[
                    Self::storage_entry(0, input_position),
                    Self::storage_entry(1, input_attribute),
                    Self::storage_entry(2, &skin_out[0]),
                    Self::storage_entry(3, &skin_out[1]),
                    Self::storage_entry(4, skinning_buffer),
                ],
            ));
        }
    }

    /// Allocate the per-instance deformed buffers of every planned stage.
    fn create_stages(
        device: &wgpu::Device,
        geometry: &GpuSkinnedMesh,
        stages: RigStages,
    ) -> Option<AnimationState> {
        if stages.is_empty() {
            return None;
        }

        Some(AnimationState {
            morph_out: stages.morphs().then(|| {
                Self::create_deformed_buffers(device, geometry.vertex_count, "morph")
            }),
            skin_out: stages.skins().then(|| {
                Self::create_deformed_buffers(device, geometry.vertex_count, "skin")
            }),
            morph_dispatch_bg: None,
            skin_dispatch_bg: None,
        })
    }

    fn create_deformed_buffers(
        device: &wgpu::Device,
        vertex_count: u32,
        stage: &str,
    ) -> [wgpu::Buffer; 2] {
        let vertex_count = u64::from(vertex_count);
        let position_label = format!("{stage} deformed position");
        let attribute_label = format!("{stage} deformed attribute");
        let position = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&position_label),
            size: vertex_count * std::mem::size_of::<VertexBufferPosition>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let attribute = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&attribute_label),
            size: vertex_count * std::mem::size_of::<VertexBufferOthers>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        [position, attribute]
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

    /// Whether this mesh runs at least one deformation stage.
    pub fn is_animated(&self) -> bool { !self.stages.is_empty() }

    /// Stage set decided during `prepare_scene`.
    pub fn stages(&self) -> RigStages { self.stages }

    /// Vertex stream the rasterizer reads.
    pub fn final_buffer(&self) -> FinalBuffer { self.stages.final_buffer() }

    /// Rest-pose vertex streams, for read-only inspection and tests.
    pub fn restpose_vertex_buffers(&self) -> &[wgpu::Buffer] {
        &self.geometry.restpose_vertex_buffer
    }

    /// Number of per-instance deformed buffers allocated by this mesh.
    ///
    /// Two per stage: one position stream and one attribute stream.
    pub fn deformed_buffer_count(&self) -> usize {
        self.animation.as_ref().map_or(0, |animation| {
            animation.morph_out.as_ref().map_or(0, |_| 2)
                + animation.skin_out.as_ref().map_or(0, |_| 2)
        })
    }

    /// Record the morphing dispatch, if the rig has a morph stage.
    ///
    /// The caller must already have bound group 0 (bindless resources). A
    /// missing clip track uses the target's base weights, or the `channels = 0`
    /// rest-pose copy when the target has no base weights.
    pub fn morph_with(&self, cp: &mut wgpu::ComputePass, frame: Option<MorphAnimationFrame>) {
        let Some(animation) = &self.animation else { return };
        if !self.stages.morphs() {
            return;
        }
        let target = self
            .morph_targets
            .as_ref()
            .expect("a morphing plan requires a morph target asset");
        let dispatch_bg = animation.morph_dispatch_bg.as_ref().expect(
            "morph dispatch bind group is not initialized: call `DrawList::ensure_deformation_bound` before the deformation pass",
        );

        let (texture, frame, channels) = match frame {
            Some(frame) => (frame.texture().index() as u32, frame.frame(), target.channel_count),
            None => match (target.base_weights, target.base_channels) {
                (Some(texture), Some(channels)) => (texture.index() as u32, 0.0, channels.get()),
                // No weight track and no static base: the shader copies the
                // rest pose without reading the texture.
                _ => (0, 0.0, 0),
            },
        };

        cp.set_bind_group(1, Some(&dispatch_bg[0]), &[]);
        cp.set_bind_group(2, &dispatch_bg[1], &[]);

        // Layout must stay in sync with `Immediate` in `gpu_vertex_morph.wgsl`.
        let mut immediates = [0u8; 12];
        immediates[..4].copy_from_slice(&texture.to_ne_bytes());
        immediates[4..8].copy_from_slice(&frame.to_le_bytes());
        immediates[8..].copy_from_slice(&channels.to_le_bytes());
        cp.set_immediates(0, &immediates);

        cp.dispatch_workgroups(self.geometry.vertex_count.div_ceil(DEFORMATION_WORKGROUP_SIZE), 1, 1);
    }

    /// Record the skinning dispatch, if the rig has a skin stage.
    ///
    /// A missing joints track is encoded as `frame = -1`, which makes the shader
    /// copy its input to the output before reading any joint data.
    pub fn skin_with(&self, cp: &mut wgpu::ComputePass, frame: Option<SkinnedAnimationFrame>) {
        let Some(animation) = &self.animation else { return };
        if !self.stages.skins() {
            return;
        }
        let dispatch_bg = animation.skin_dispatch_bg.as_ref().expect(
            "skinning dispatch bind group is not initialized: call `DrawList::ensure_deformation_bound` before the deformation pass",
        );

        cp.set_bind_group(1, Some(dispatch_bg), &[]);
        cp.set_immediates(0, &Self::skin_immediates(frame));

        cp.dispatch_workgroups(self.geometry.vertex_count.div_ceil(DEFORMATION_WORKGROUP_SIZE), 1, 1);
    }

    /// Encode the skinning immediates. `None` is the identity sentinel.
    ///
    /// Layout must stay in sync with `Immediate` in `gpu_skinning.wgsl`.
    fn skin_immediates(frame: Option<SkinnedAnimationFrame>) -> [u8; 8] {
        let (texture, frame) = match frame {
            Some(frame) => (frame.texture().index() as u32, frame.frame()),
            None => (0, -1.0),
        };

        let mut immediates = [0u8; 8];
        immediates[..4].copy_from_slice(&texture.to_ne_bytes());
        immediates[4..].copy_from_slice(&frame.to_le_bytes());
        immediates
    }

    /// Record every dispatch of this mesh in the required order.
    ///
    /// Morphing runs first and feeds skinning when both stages exist. A missing
    /// track is deformed with netural values, such that stages are never skipped.
    pub fn record_deformation(
        &self,
        cp: &mut wgpu::ComputePass,
        pipelines: &ComputePipelineLibrary,
        tracks: Option<AnimationTracks>,
    ) {
        let (joints, weights) = match tracks {
            Some(tracks) => (tracks.joints, tracks.weights),
            None => (None, None),
        };

        if self.stages.morphs() {
            let pipeline = pipelines
                .get_pipeline(ComputePipelineKey::GpuVertexMorphing)
                .expect("the GPU vertex morphing pipeline should be registered");
            cp.set_pipeline(pipeline);
            self.morph_with(cp, weights);
        }
        if self.stages.skins() {
            let pipeline = pipelines
                .get_pipeline(ComputePipelineKey::GpuSkinning)
                .expect("the GPU skinning pipeline should be registered");
            cp.set_pipeline(pipeline);
            self.skin_with(cp, joints);
        }
    }

    pub fn get_material(&self) -> &PBRMaterial { &self.material }
}

impl Mesh for AnimatedMesh {
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer] {
        match self.stages.final_buffer() {
            FinalBuffer::Restpose => self.geometry.restpose_vertex_buffer.as_slice(),
            FinalBuffer::MorphOut => self
                .animation
                .as_ref()
                .and_then(|animation| animation.morph_out.as_ref())
                .expect("a morphing plan has morph output buffers")
                .as_slice(),
            FinalBuffer::SkinOut => self
                .animation
                .as_ref()
                .and_then(|animation| animation.skin_out.as_ref())
                .expect("a skinning plan has skin output buffers")
                .as_slice(),
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::renderer::resources::bindless::SampledTextureHandle;

    #[test]
    fn missing_joints_encode_the_identity_sentinel() {
        let immediates = AnimatedMesh::skin_immediates(None);
        assert_eq!(u32::from_ne_bytes(immediates[..4].try_into().unwrap()), 0);
        assert_eq!(
            f32::from_le_bytes(immediates[4..].try_into().unwrap()),
            -1.0,
            "a missing joints track must disable joint reads"
        );
    }

    #[test]
    fn present_joints_encode_the_texture_and_frame() {
        let frame = SkinnedAnimationFrame::new(SampledTextureHandle::from_index(7), 1.5);
        let immediates = AnimatedMesh::skin_immediates(Some(frame));
        assert_eq!(u32::from_ne_bytes(immediates[..4].try_into().unwrap()), 7);
        assert_eq!(f32::from_le_bytes(immediates[4..].try_into().unwrap()), 1.5);
    }
}
