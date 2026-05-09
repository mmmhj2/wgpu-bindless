
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexBufferPosition {
    pub position: [f32; 3]
}

impl VertexBufferPosition {
    const BINDINGS : [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute{
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0
    }];
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexBufferOthers {
    /// Vertex color in RGBA
    pub color: [f32; 4],
    /// Object space normal in 3D
    pub normal: [f32; 3],
    /// Barycentric 
    pub uv0: [f32; 2]
}

impl VertexBufferOthers {
    const BINDINGS : [wgpu::VertexAttribute; 3] = [
        wgpu::VertexAttribute{
            format: wgpu::VertexFormat::Float32x4,
            offset: 0,
            shader_location: 1
        },
        wgpu::VertexAttribute{
            format: wgpu::VertexFormat::Float32x3,
            offset: (size_of::<f32>() * 4) as u64,
            shader_location: 2
        },
        wgpu::VertexAttribute{
            format: wgpu::VertexFormat::Float32x2,
            offset: (size_of::<f32>() * (4 + 3)) as u64,
            shader_location: 3
        }
    ];
}

pub enum VertexType {
    PositionOnly,
    Basic
}

impl VertexType {

    const VBL_POSITION_ONLY : [wgpu::VertexBufferLayout<'static>; 1] = [
        wgpu::VertexBufferLayout{
            array_stride: std::mem::size_of::<VertexBufferPosition>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &VertexBufferPosition::BINDINGS
        }
    ];

    const VBL_BASIC : [wgpu::VertexBufferLayout<'static>; 2] = [
        wgpu::VertexBufferLayout{
            array_stride: std::mem::size_of::<VertexBufferPosition>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &VertexBufferPosition::BINDINGS
        },
        wgpu::VertexBufferLayout{
            array_stride: std::mem::size_of::<VertexBufferOthers>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &VertexBufferOthers::BINDINGS
        }
    ];

    pub fn get_vertex_buffer_layout(vt: VertexType) -> &'static[wgpu::VertexBufferLayout<'static>] {
        match vt {
            Self::PositionOnly => &Self::VBL_POSITION_ONLY,
            Self::Basic => &Self::VBL_BASIC
        }
    }
}
