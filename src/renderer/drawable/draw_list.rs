use crate::renderer::{
    animation::AnimationManager,
    drawable::{drawable_mesh_traits::DrawableMesh, skinned_mesh::AnimatedMesh, static_mesh_batch::StaticMeshBatch},
    pipeline::compute_library::{ComputeDispatchBindingSource, ComputePipelineLibrary},
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
    skinned_opaque: Vec<AnimatedMesh>,
    animations: AnimationManager,
}

impl DrawList {
    pub(crate) fn new(
        static_opaque_batch: Vec<StaticMeshBatch>,
        skinned_opaque: Vec<AnimatedMesh>,
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

    /// Record every deformation dispatch of every animated mesh.
    ///
    /// Morphing runs before skinning for each mesh. Missing clip tracks are
    /// deformed with a netural value. No stage is skipped because of a missing
    /// track. The caller must already have bound the global bindless bind group
    /// (group 0).
    pub fn record_deformation(
        &self,
        cp: &mut wgpu::ComputePass,
        pipelines: &ComputePipelineLibrary,
    ) -> &Self {
        for (index, mesh) in self.skinned_opaque.iter().enumerate() {
            mesh.record_deformation(cp, pipelines, self.animations.frame(index));
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

    /// Materialize the deformation dispatch bind groups.
    ///
    /// Must run before the compute pass records
    /// [`Self::record_deformation`]. A no-op for static-only lists and for rigs
    /// without a deformation stage.
    pub fn ensure_deformation_bound(
        &mut self,
        device: &wgpu::Device,
        bindings: &impl ComputeDispatchBindingSource,
    ) -> &mut Self {
        for mesh in &mut self.skinned_opaque {
            mesh.ensure_deformation_bound(device, bindings);
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
    pub fn get_skinned_meshes(&self) -> &[AnimatedMesh] {
        &self.skinned_opaque
    }
}
