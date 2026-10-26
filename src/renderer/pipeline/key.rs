/// Identifies a render-pipeline program (shader entry points and fixed state).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PipelineKey {
    Default,
    CookTorrancePbr,
    Skybox,
}

/// Identifies a registered vertex-buffer layout.
///
/// New layouts (for example, ones derived from Naga reflection) can be added at
/// runtime without touching the pipeline library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VertexLayoutKey(pub u32);

impl VertexLayoutKey {
    pub const BASIC: Self = Self(0);
    pub const POSITION_ONLY: Self = Self(1);
    pub const EMPTY: Self = Self(2);
}

/// Identifies a `wgpu::PipelineLayout` (bind-group layout combination).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PipelineLayoutKey {
    Default,
    Skybox,
}

/// Complete cache key for a rasterizer pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RenderPipelineKey {
    pub program: PipelineKey,
    pub vertex_layout: VertexLayoutKey,
    pub layout: PipelineLayoutKey,
}
