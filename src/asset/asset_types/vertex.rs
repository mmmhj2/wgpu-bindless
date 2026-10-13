/// Object-space vertex position.
///
/// This is pure vertex data shared by the asset layer and the renderer; the
/// corresponding GPU vertex-attribute bindings live in the renderer
/// [`VertexType`].
#[repr(C)]
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexBufferPosition {
    pub position: [f32; 3]
}

/// Vertex attributes other than position.
///
/// This is pure vertex data shared by the asset layer and the renderer; the
/// corresponding GPU vertex-attribute bindings live in the renderer
/// [`VertexType`].
#[repr(C)]
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexBufferOthers {
    /// Vertex color in RGBA
    pub color: [f32; 4],
    /// Object space normal in 3D
    pub normal: [f32; 3],
    /// Object space tangent vector, with the w component indicating handiness.
    pub tangent: [f32; 4],
    /// Texture coordinate channel 0
    pub uv0: [f32; 2]
}
