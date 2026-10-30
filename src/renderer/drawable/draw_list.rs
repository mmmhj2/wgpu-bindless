use crate::renderer::drawable::{drawable_mesh_traits::DrawableMesh, skinned_mesh::SkinnedMesh, static_mesh_batch::StaticMeshBatch};

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
