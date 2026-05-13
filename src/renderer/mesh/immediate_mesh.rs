use bytemuck::Zeroable;
use wgpu::{BufferDescriptor, BufferUsages};

use crate::renderer::{device_interface::DeviceInterface, mesh::{tangent_calulation::{CanRecaluclateTangent, TangentRecalculator}, vertex_types::{VertexBufferOthers, VertexBufferPosition}}, pipeline::{bindless_resource_manager::BindlessResourceManager, pbr_material::PBRMaterial}};

pub struct ImmediateMeshBuilder {
    state       : VertexBufferOthers,
    position    : Vec<VertexBufferPosition>,
    attributes  : Vec<VertexBufferOthers>,
    material    : PBRMaterial
}

pub struct ImmediateMesh {
    vertex_count        : u32,
    vertex_buffer_arr   : [wgpu::Buffer; 2],
    material            : PBRMaterial
}

impl ImmediateMesh {
    fn new(cnt: u32, pb: wgpu::Buffer, ab: wgpu::Buffer, material: PBRMaterial) -> Self {
        Self { vertex_count: cnt, vertex_buffer_arr: [pb, ab], material}
    }
}

impl ImmediateMeshBuilder {
    pub fn new(di: &DeviceInterface, mgr: &mut BindlessResourceManager) -> Self {
        let material = PBRMaterial::create_from_views(
            di.get_device(),
            mgr,
            None,
            None,
            None
        ).expect("Failed to create material for immediate mesh");

        Self {
            state: VertexBufferOthers::zeroed(),
            position: Vec::new(),
            attributes: Vec::new(),
            material
        }
    }

    pub fn color3f(&mut self, c: [f32; 3]) -> () {
        self.color4f([c[0], c[1], c[2], 1.0]);
    }

    pub fn color4f(&mut self, c: [f32; 4]) -> () {
        self.state.color = c;
    }

    pub fn tangent3f(&mut self, t: [f32; 3]) -> () {
        self.tangent4f([t[0], t[1], t[2], 1.0]);
    }

    pub fn tangent4f(&mut self, t: [f32; 4]) -> () {
        self.state.tangent = t;
    }

    pub fn normal3f(&mut self, n: [f32; 3]) -> () {
        self.state.normal = n;
    }

    pub fn texcoord2f(&mut self, t: [f32; 2]) -> () {
        self.state.uv0 = t;
    }

    pub fn vertex2f(&mut self, v: [f32; 2]) -> () {
        self.vertex3f([v[0], v[1], 0.0])
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
        ImmediateMesh::new(self.position.len() as u32, pb, ab, self.material)
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

impl super::DrawableMesh for ImmediateMesh {

    fn get_model_matrix(&self) -> &[[f32; 4]; 4] {
        // Wanted to use cgmath::Matrix4::identity() but failed.
        &[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    }

    fn get_material(&self) -> &PBRMaterial {
        &self.material
    }
}

impl CanRecaluclateTangent for ImmediateMeshBuilder {
    fn get_position_buffer_tgt(&self) -> &Vec<VertexBufferPosition> {
        &self.position
    }

    fn get_attribute_buffer_tgt(&self) -> &Vec<VertexBufferOthers> {
        &self.attributes
    }

    fn get_attribute_buffer_tgt_mut(&mut self) -> &mut Vec<VertexBufferOthers> {
        &mut self.attributes
    }

    fn get_index_buffer_tgt(&self) -> Option<&Vec<u32>> {
        None
    }
}

impl TangentRecalculator for ImmediateMeshBuilder {}
