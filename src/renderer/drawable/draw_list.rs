use crate::renderer::{
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
    skinned_opaque: Vec<SkinnedMesh>
}

impl DrawList {
    pub(crate) fn new(
        static_opaque_batch: Vec<StaticMeshBatch>,
        skinned_opaque: Vec<SkinnedMesh>
    ) -> Self {
        Self { static_opaque_batch, skinned_opaque }
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

    /// Get a list of skinned meshes in this draw call.
    /// 
    /// Useful to produce immediate vertex buffers used for the draw proper.
    /// 
    /// TODO: this API design kind of smells. Figure out an improvement after
    /// skinning animation is actually working.
    pub fn get_skinned_meshes(&self) -> &[SkinnedMesh] {
        &self.skinned_opaque
    }
}
