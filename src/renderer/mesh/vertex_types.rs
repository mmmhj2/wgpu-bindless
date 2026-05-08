
pub struct VertexBufferPosition {
    position: [f32; 3]
}

impl VertexBufferPosition {
    const BINDINGS : [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute{
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0
    }];
}

pub struct VertexBufferOthers {
    /// Vertex color in RGBA
    color: [f32; 4],
    /// Object space normal in 3D
    normal: [f32; 3],
    /// Barycentric 
    uv0: [f32; 2]
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
