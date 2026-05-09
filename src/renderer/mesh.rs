use crate::renderer::mesh::vertex_types::VertexType;

pub mod vertex_types;
pub mod immediate_mesh;

trait Mesh {
    /// Get a slice of references to all vertex attribute buffers.
    /// The binding of these buffers into the render pass is specified
    /// by the `VertexType` of the current mesh.
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer];
    fn get_index_buffer(&self) -> Option<&wgpu::Buffer>;
    fn get_vertex_type(&self) -> VertexType;
}
