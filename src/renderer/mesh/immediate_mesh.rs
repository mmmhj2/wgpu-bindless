use bytemuck::Zeroable;
use wgpu::{BufferDescriptor, BufferUsages};

use crate::renderer::{device_interface::DeviceInterface, mesh::vertex_types::{VertexBufferOthers, VertexBufferPosition}};

pub struct ImmediateMeshBuilder {
    state       : VertexBufferOthers,
    position    : Vec<VertexBufferPosition>,
    attributes  : Vec<VertexBufferOthers>
}

pub struct ImmediateMesh {
    vertex_count        : u32,
    vertex_buffer_arr   : [wgpu::Buffer; 2]
}

impl ImmediateMesh {
    fn new(cnt: u32, pb: wgpu::Buffer, ab: wgpu::Buffer) -> Self {
        Self { vertex_count: cnt, vertex_buffer_arr: [pb, ab] }
    }
}

impl ImmediateMeshBuilder {
    pub fn new() -> Self {
        Self { state: VertexBufferOthers::zeroed(), position: Vec::new(), attributes: Vec::new() }
    }

    pub fn color4f(&mut self, c: [f32; 4]) -> () {
        self.state.color = c;
    }

    pub fn normal3f(&mut self, n: [f32; 3]) -> () {
        self.state.normal = n;
    }

    pub fn texcoord2f(&mut self, t: [f32; 2]) -> () {
        self.state.uv0 = t;
    }

    pub fn vertex3f(&mut self, v: [f32; 3]) -> () {
        self.position.push(VertexBufferPosition{position: v});
        self.attributes.push(self.state.clone());
    }

    /// Commit 
    pub fn commit(self, di: &DeviceInterface) -> ImmediateMesh {
        assert_eq!(self.position.len(), self.attributes.len());

        let pb = di.get_device().create_buffer(&BufferDescriptor{
            label: None,
            mapped_at_creation: false,
            size: (self.position.len() as u64) * (std::mem::size_of::<VertexBufferPosition>() as u64),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST
        });

        let ab = di.get_device().create_buffer(&BufferDescriptor{
            label: None,
            mapped_at_creation: false,
            size: (self.attributes.len() as u64) * (std::mem::size_of::<VertexBufferOthers>() as u64),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST
        });

        di.get_queue().write_buffer(&pb, 0, bytemuck::cast_slice(self.position.as_slice()));
        di.get_queue().write_buffer(&ab, 0, bytemuck::cast_slice(self.attributes.as_slice()));
        ImmediateMesh::new(self.position.len() as u32, pb, ab)
    }
}

impl super::Mesh for ImmediateMesh {
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer] {
        &self.vertex_buffer_arr
    }

    fn get_index_buffer(&self) -> Option<&wgpu::Buffer> {
        None
    }
    
    fn get_vertex_type(&self) -> super::vertex_types::VertexType {
        super::vertex_types::VertexType::Basic
    }
    
    fn get_vertex_draw_count(&self) -> u32 {
        self.vertex_count
    }
}
