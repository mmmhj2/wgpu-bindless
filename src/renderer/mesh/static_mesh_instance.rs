use std::{collections::VecDeque, num::NonZero, sync::Arc};
use crate::renderer::{device_interface::DeviceInterface, mesh::{Mesh, drawable_mesh_traits::DrawableMesh, mesh_manager::MeshManager, static_mesh::{StaticMesh, StaticMeshBuilder}}, pipeline::{bindless_resource_manager::BindlessResourceManager, pbr_material::PBRMaterial, pipeline_state::PipelineStates}};

/// Instances of instanced mesh.
/// Holds unique data for each instance such as model matrix, and a reference to the underlying mesh.
#[derive(Clone)]
pub struct StaticMeshInstance {
    mesh            : Arc<StaticMesh>,
    model_matrix    : wgpu::Buffer,
    model_matrix_bg : wgpu::BindGroup
}

impl StaticMeshInstance {
    fn generate_bind_group(di: &DeviceInterface, mmgr: &MeshManager, buffer: &wgpu::Buffer) -> wgpu::BindGroup {
        di.get_device().create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("model matrix buffer bind group"),
            layout: mmgr.get_bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(
                        wgpu::BufferBinding { buffer, offset: 0, size: NonZero::new(buffer.size()) }
                    )
                }
            ]
        })
    }

    /// Create instances from a GLTF scene.
    /// 
    /// Only nodes that contains meshes are processed.
    /// Transforms of the nodes will be preserved in the model matrices of the
    /// instances produced.
    pub fn create_from_gltf_scene(
        di: &DeviceInterface,
        pipeline_states: &mut PipelineStates,
        scene: &gltf::Scene,
        buffers: &Vec<gltf::buffer::Data>,
        images: &Vec<gltf::image::Data>
    ) -> Vec<Self> {
        let mut ret = Vec::new();

        // Do a BFS to collect all meshes.
        // DFS, recursion and trees in Rust are simply PITA.
        let mut transform_queue = VecDeque::<cgmath::Matrix4<f32>>::new();
        let mut node_queue = VecDeque::<usize>::new();

        for root_node in scene.nodes() {
            transform_queue.push_back(root_node.transform().matrix().into());
            node_queue.push_back(root_node.index());

            while !node_queue.is_empty() {
                let current_node = scene.nodes().nth(node_queue.pop_front().unwrap()).expect("valid node index");
                let current_transform = transform_queue.pop_front().unwrap();
                if let Some(m) = current_node.mesh() {
                    let instances = Self::create_from_gltf_mesh(di, pipeline_states, &m, buffers, images);
                    for inst in &instances {
                        di.get_queue().write_buffer(&inst.model_matrix, 0, bytemuck::cast_slice(AsRef::<[f32; 16]>::as_ref(&current_transform)));
                    }
                    ret.extend(instances);
                }

                for ch in current_node.children() {
                    transform_queue.push_back(current_transform * cgmath::Matrix4::from(ch.transform().matrix()));
                    node_queue.push_back(ch.index());
                }
            }
        }

        return ret;
    }

    /// Create instances from a GLTF mesh.
    /// 
    /// All primitives will be processed, each one corresponding to an new instance.
    /// Model matrix of the mesh will contain undefined contents.
    pub fn create_from_gltf_mesh(
        di: &DeviceInterface,
        pipeline_states: &mut PipelineStates,
        mesh: &gltf::Mesh,
        buffers: &Vec<gltf::buffer::Data>,
        images: &Vec<gltf::image::Data>
    ) -> Vec<Self> {
        let mut ret = Vec::new();

        for primitive in mesh.primitives() {
            let buffer = di.get_device().create_buffer(&wgpu::BufferDescriptor{
                label: Some("model matrix buffer"),
                size: (std::mem::size_of::<f32>() * 16) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = Self::generate_bind_group(di, pipeline_states.get_mesh_manager(), &buffer);
            ret.push(
                Self{
                    mesh: StaticMeshBuilder::new(di, pipeline_states.get_bindless_resource_manager_mut(), &primitive, buffers, images).recondition_and_commit(di).into(),
                    model_matrix: buffer,
                    model_matrix_bg: bg
                }
            )
        }
        return ret;
    }
}

impl Mesh for StaticMeshInstance {
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer] { self.mesh.get_vertex_buffer() }
    fn get_vertex_draw_count(&self) -> u32 { self.mesh.get_vertex_draw_count() }
    fn get_index_buffer(&self) -> Option<&wgpu::Buffer> { self.mesh.get_index_buffer() }
    fn get_vertex_type(&self) -> super::vertex_types::VertexType { self.mesh.get_vertex_type() }
}

impl DrawableMesh for StaticMeshInstance {
    fn get_material(&self) -> &PBRMaterial { self.mesh.get_material() }
    fn get_model_matrix_bind_group(&self) -> &wgpu::BindGroup { &self.model_matrix_bg }
    fn get_instance_range(&self) -> std::ops::Range<u32> { 0..1 }
}
