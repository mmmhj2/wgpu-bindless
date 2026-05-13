use crate::renderer::{mesh::vertex_types::VertexType, pipeline::pbr_material::{PBRMaterial}};

pub mod vertex_types;
pub mod vertex_reconditioner;
pub mod immediate_mesh;
pub mod instanced_mesh;

pub trait Mesh {
    /// Get a slice of references to all vertex attribute buffers.
    /// 
    /// The binding of these buffers into the render pass is specified
    /// by the `VertexType` of the current mesh.
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer];

    /// Query how many vertices should be drawn for the mesh.
    /// 
    /// If the mesh is indexed, it should be equal to the index count.
    /// If it is not indexed, it should be equal to the vertex count.
    fn get_vertex_draw_count(&self) -> u32;

    /// Get the index buffer of the mesh.
    /// 
    /// For non-indexed meshes, None is returned.
    /// In which case the draw call should not be indexed either.
    fn get_index_buffer(&self) -> Option<&wgpu::Buffer>;

    /// Get the vertex type of the mesh.
    /// Currently unused.
    #[allow(unused)]
    fn get_vertex_type(&self) -> VertexType;
}

pub trait DrawableMesh : Mesh{
    /// Get the model matrix of the mesh.
    /// 
    /// To save bandwidth the model matrix should have 4 rows and 4 columns.
    /// Note that both cgmath and WGSL uses *column-major* matrices.
    /// While Rust has enforces row-major order, so long as you don't manipulate
    /// the matrix directly with Rust array, it will be fine.
    fn get_model_matrix(&self) -> &[[f32; 4]; 4];

    /// Get the material description of the mesh.
    fn get_material(&self) -> &PBRMaterial;

    fn draw(&self, rp: &mut wgpu::RenderPass) -> () {

        rp.set_immediates(0, bytemuck::cast_slice(self.get_model_matrix()));
        rp.set_immediates(64, &self.get_material().as_u8_arr());

        let vbs = self.get_vertex_buffer();
        for (i, b) in vbs.iter().enumerate() {
            rp.set_vertex_buffer(i as u32, b.slice(..));
        }

        if let Some(ib) = self.get_index_buffer() {
            rp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.get_vertex_draw_count(), 0, 0..1);
        } else {
            rp.draw(0..self.get_vertex_draw_count(), 0..1);
        }
    }
} 
