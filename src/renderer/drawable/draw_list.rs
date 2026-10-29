use crate::renderer::drawable::{drawable_mesh_traits::DrawableMesh, static_mesh_batch::StaticMeshBatch};

#[derive(Debug, Clone, Copy)]
pub enum DrawableType {
    All,
    Opaque
}

/// List of assorted drawables that can be sent to draw directly.
#[derive(Debug, Clone)]
pub struct DrawList {
    static_opaque_batch: Vec<StaticMeshBatch>
}

impl DrawList {
    pub(crate) fn new(
        static_opaque_batch: Vec<StaticMeshBatch>
    ) -> Self {
        Self { static_opaque_batch }
    }

    pub fn draw(&self, draw_type: DrawableType, rp: &mut wgpu::RenderPass) {
        match draw_type {
            DrawableType::All => {
                for m in &self.static_opaque_batch {
                    m.draw(rp);
                }
            },
            DrawableType::Opaque => {
                for m in &self.static_opaque_batch {
                    m.draw(rp);
                }
            },
        }
    }

    pub fn count(&self, draw_type: DrawableType) -> usize {
        match draw_type {
            DrawableType::All => {
                self.static_opaque_batch.len()
            },
            DrawableType::Opaque => self.static_opaque_batch.len(),
        }
    }
}
