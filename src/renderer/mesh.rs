use std::num::NonZero;

use crate::renderer::{device_interface::DeviceInterface, mesh::{mesh_manager::MeshManager, vertex_types::VertexType}, pipeline::pbr_material::PBRMaterial};

pub mod vertex_types;
pub mod vertex_reconditioner;
pub mod immediate_mesh;
pub mod static_mesh;
pub mod static_mesh_instance;
pub mod mesh_manager;
pub mod drawable_mesh_traits;

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
