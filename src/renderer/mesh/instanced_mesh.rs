use std::sync::Arc;

use cgmath::SquareMatrix;

use crate::renderer::{device_interface::DeviceInterface, mesh::{DrawableMesh, Mesh}, pipeline::pbr_material::PBRMaterial};


struct InstancedMesh {

}

impl InstancedMesh {
    fn get_material(&self) -> &PBRMaterial {
        todo!()
    }
}

impl InstancedMesh {
    pub fn create_from_gltf(di: &DeviceInterface, p: gltf::Primitive) -> Arc<Self> {
        todo!()
    }
}

impl Mesh for InstancedMesh {
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer] {
        todo!()
    }

    fn get_vertex_draw_count(&self) -> u32 {
        todo!()
    }

    fn get_index_buffer(&self) -> Option<&wgpu::Buffer> {
        todo!()
    }

    fn get_vertex_type(&self) -> super::vertex_types::VertexType {
        todo!()
    }
}

#[derive(Clone)]
struct InstancedMeshInstance {
    mesh            : Arc<InstancedMesh>,
    model_matrix    : cgmath::Matrix4<f32>
}

impl InstancedMeshInstance {
    /// Create a new instance from a pre-existing mesh and a new model matrix.
    pub fn new(mesh: Arc<InstancedMesh>, model_matrix: cgmath::Matrix4<f32>) -> Self {
        Self { mesh, model_matrix }
    }

    pub fn create_from_gltf(di: &DeviceInterface, m: gltf::Mesh) -> Vec<Self> {
        let mut ret = Vec::new();

        for primitive in m.primitives() {
            ret.push(
                Self::new(
                    InstancedMesh::create_from_gltf(di, primitive),
                    cgmath::Matrix4::identity().into()
                )
            )
        }

        return ret;
    }
}

impl Mesh for InstancedMeshInstance {
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer] {
        self.mesh.get_vertex_buffer()
    }

    fn get_vertex_draw_count(&self) -> u32 {
        self.mesh.get_vertex_draw_count()
    }

    fn get_index_buffer(&self) -> Option<&wgpu::Buffer> {
        self.mesh.get_index_buffer()
    }

    fn get_vertex_type(&self) -> super::vertex_types::VertexType {
        self.mesh.get_vertex_type()
    }
}

impl DrawableMesh for InstancedMeshInstance {
    fn get_model_matrix(&self) -> &[[f32; 4]; 4] {
        self.model_matrix.as_ref()
    }

    fn get_material(&self) -> &PBRMaterial {
        self.mesh.get_material()
    }
}
