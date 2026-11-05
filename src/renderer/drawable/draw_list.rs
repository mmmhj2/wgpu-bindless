use crate::renderer::{
    animation::AnimationManager,
    drawable::{drawable_mesh_traits::DrawableMesh, skinned_mesh::SkinnedMesh, static_mesh_batch::StaticMeshBatch},
    pipeline::compute_library::ComputeDispatchBindingSource,
    resources::mesh_layout::ModelMatrixBindingSource,
};

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum DrawableType {
    OpaqueStatic,
    OpaqueSkinned
}

/// List of assorted drawables that can be sent to draw directly.
#[derive(Debug)]
pub struct DrawList {
    static_opaque_batch: Vec<StaticMeshBatch>,
    skinned_opaque: Vec<SkinnedMesh>,
    animations: AnimationManager,
}

impl DrawList {
    pub(crate) fn new(
        static_opaque_batch: Vec<StaticMeshBatch>,
        skinned_opaque: Vec<SkinnedMesh>,
        animations: AnimationManager,
    ) -> Self {
        assert_eq!(
            animations.player_count(),
            skinned_opaque.len(),
            "every skinned mesh needs exactly one animation player"
        );
        Self { static_opaque_batch, skinned_opaque, animations }
    }

    /// CPU animation state of this draw list.
    pub fn animations(&self) -> &AnimationManager { &self.animations }
    pub fn animations_mut(&mut self) -> &mut AnimationManager { &mut self.animations }

    /// Advance every animation player by `dt` seconds.
    pub fn update_animations(&mut self, dt: f32) { self.animations.update(dt); }

    /// Record one morphing dispatch per animated mesh.
    /// 
    /// Vertex morphing should be performed before any skinning.
    /// 
    /// The caller must already have bound the GPU skinning pipeline and the
    /// global bindless bind group (group 0).
    pub fn record_morphing(&self, cp: &mut wgpu::ComputePass) -> &Self {
        todo!()
    }

    /// Record one skinning dispatch per animated mesh.
    ///
    /// The caller must already have bound the GPU skinning pipeline and the
    /// global bindless bind group (group 0). Meshes without a clip (rest pose)
    /// are skipped.
    pub fn record_skinning(&self, cp: &mut wgpu::ComputePass) -> &Self {
        for (index, mesh) in self.skinned_opaque.iter().enumerate() {
            if let Some(frame) = self.animations.frame(index) {
                mesh.skin_with(cp, frame);
            }
        }

        self
    }

    /// Materialize every mesh's group-3 (model matrix) bind group.
    ///
    /// Idempotent: calling this once after
    /// [`crate::renderer::orchestrator::RenderOrchestrator::prepare_scene`]
    /// is enough, and calling it every frame is cheap.
    ///
    /// Must run before [`Self::draw`]. Drawing an unbound mesh panics.
    /// 
    /// TODO: return a token for safe drawing after bound, and make the panicking
    /// unreachable.
    pub fn ensure_bound(
        &mut self,
        device: &wgpu::Device,
        bindings: &impl ModelMatrixBindingSource,
    ) -> &mut Self {
        for batch in &mut self.static_opaque_batch {
            batch.ensure_bound(device, bindings);
        }
        for mesh in &mut self.skinned_opaque {
            mesh.ensure_bound(device, bindings);
        }
        self
    }

    /// Materialize the skinning dispatch bind groups.
    ///
    /// Must run before the skinning compute pass records
    /// [`crate::renderer::drawable::skinned_mesh::SkinnedMesh::skin_with`].
    /// A no-op for static-only lists and for meshes without an animation.
    /// 
    /// Drawing an unbound skinned mesh panics.
    pub fn ensure_skinning_bound(
        &mut self,
        device: &wgpu::Device,
        bindings: &impl ComputeDispatchBindingSource,
    ) -> &mut Self {
        for mesh in &mut self.skinned_opaque {
            mesh.ensure_skinning_bound(device, bindings);
        }
        self
    }

    /// Materialize the morphing dispatch bind groups.
    ///
    /// Must run before the morping compute pass records
    /// [`...`].
    /// A no-op for static-only lists and for meshes without an animation.
    /// 
    /// Drawing an unbound mesh with vertex morph panics.
    pub fn ensure_morphing_bound(
        &mut self,
        device: &wgpu::Device,
        bindings: &impl ComputeDispatchBindingSource
    ) -> &mut Self {
        for mesh in &mut self.skinned_opaque {
            mesh.ensure_morphing_bound(device, bindings);
        }
        self
    }

    /// Issue draw calls to draw this list.
    /// 
    /// Panic
    /// ---
    /// Panics if any draw calls panic. Drawing meshes that are not bound will
    /// panic.
    pub fn draw(&self, draw_type: DrawableType, rp: &mut wgpu::RenderPass) {
        match draw_type {
            DrawableType::OpaqueStatic => {
                for m in &self.static_opaque_batch {
                    m.draw(rp);
                }
            },
            DrawableType::OpaqueSkinned => {
                for m in &self.skinned_opaque {
                    m.draw(rp);
                }
            },
        }
    }

    /// Get the number of draw calls in this list.
    /// 
    /// Multiple instanced meshes may be grouped into a single instanced draw
    /// call. This count therefore does not reflect the number of actual
    /// instances.
    pub fn count(&self, draw_type: DrawableType) -> usize {
        match draw_type {
            DrawableType::OpaqueStatic => self.static_opaque_batch.len(),
            DrawableType::OpaqueSkinned => self.skinned_opaque.len(),
        }
    }

    /// Get the skinned meshes in this draw call.
    ///
    /// Escape hatch to get all skinned meshes. Prefer to use other methods when
    /// available.
    pub fn get_skinned_meshes(&self) -> &[SkinnedMesh] {
        &self.skinned_opaque
    }
}
