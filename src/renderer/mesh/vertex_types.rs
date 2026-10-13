use crate::asset::asset_types::vertex::{VertexBufferOthers, VertexBufferPosition};

#[derive(Clone, Copy, Debug)]
pub enum VertexType {
    PositionOnly,
    Basic
}

impl VertexType {

    const POSITION_ATTRIBUTES : [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute{
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0
    }];

    /// TODO: try `wgpu::vertex_attr_array!()` macro
    const OTHERS_ATTRIBUTES : [wgpu::VertexAttribute; 4] = [
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
            format: wgpu::VertexFormat::Float32x4,
            offset: (size_of::<f32>() * (4 + 3)) as u64,
            shader_location: 3
        },
        wgpu::VertexAttribute{
            format: wgpu::VertexFormat::Float32x2,
            offset: (size_of::<f32>() * (4 + 3 + 4)) as u64,
            shader_location: 4
        }
    ];

    const VBL_POSITION_ONLY : [wgpu::VertexBufferLayout<'static>; 1] = [
        wgpu::VertexBufferLayout{
            array_stride: std::mem::size_of::<VertexBufferPosition>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::POSITION_ATTRIBUTES
        }
    ];

    const VBL_BASIC : [wgpu::VertexBufferLayout<'static>; 2] = [
        wgpu::VertexBufferLayout{
            array_stride: std::mem::size_of::<VertexBufferPosition>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::POSITION_ATTRIBUTES
        },
        wgpu::VertexBufferLayout{
            array_stride: std::mem::size_of::<VertexBufferOthers>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::OTHERS_ATTRIBUTES
        }
    ];

    pub fn get_vertex_buffer_layout(vt: VertexType) -> &'static[wgpu::VertexBufferLayout<'static>] {
        match vt {
            Self::PositionOnly => &Self::VBL_POSITION_ONLY,
            Self::Basic => &Self::VBL_BASIC
        }
    }
}
