use crate::renderer::pipeline::{
    key::{PipelineKey, PipelineLayoutKey},
    rasterizer_pipeline::RasterizerPipeline,
};

/// Creates render pipelines for one program.
///
/// Each pipeline recipe has a unique pipeline key indicating its associated
/// shader program. A layout key is also present to determine its bind group
/// layout.
/// 
/// The recipe currently acts as the extension point for runtime/Naga-derived
/// vertex state. It can also provide potential extensions for other states (wireframe, etc.).
pub trait PipelineRecipe {
    fn key(&self) -> PipelineKey;
    fn layout_key(&self) -> PipelineLayoutKey;

    fn create(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        vertex_layout: &[wgpu::VertexBufferLayout<'static>],
    ) -> RasterizerPipeline;
}

/// Create a shader module used for bindless rendering.
/// 
/// The shader module will be prefixed automatically by the bindless header.
pub(crate) fn bindless_shader_module(
    device: &wgpu::Device,
    label: &'static str,
    body: &'static str,
) -> wgpu::ShaderModule {
    let header = include_str!("shaders/bindless_header.wgsl");
    let complete = format!("{header}{body}");
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(complete.into()),
    })
}
