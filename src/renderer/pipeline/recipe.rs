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

const BINDLESS_SCENE_HEADER: &str = concat!(
    include_str!("shaders/bindless_resources.wgsl"),
    include_str!("shaders/bindless_header.wgsl"),
);

const BINDLESS_RESOURCES_HEADER: &str = include_str!("shaders/bindless_resources.wgsl");

/// Create a shader module used for bindless rendering.
/// 
/// The shader module will be prefixed automatically by the bindless resource
/// declarations and the scene header (camera, lighting, model matrices and
/// material immediates).
pub(crate) fn bindless_shader_module(
    device: &wgpu::Device,
    label: &'static str,
    body: &'static str,
) -> wgpu::ShaderModule {
    shader_module_with_header(device, label, BINDLESS_SCENE_HEADER, body)
}

/// Create a shader module prefixed only with the bindless resource arrays.
///
/// Passes that own their bind group layout but sample through the global
/// bindless sampler table use this. The shader body is responsible for
/// declaring its own `var<immediate>` block; the scene header would conflict
/// with it.
pub(crate) fn bindless_resources_shader_module(
    device: &wgpu::Device,
    label: &'static str,
    body: &'static str,
) -> wgpu::ShaderModule {
    shader_module_with_header(device, label, BINDLESS_RESOURCES_HEADER, body)
}

fn shader_module_with_header(
    device: &wgpu::Device,
    label: &'static str,
    header: &'static str,
    body: &'static str,
) -> wgpu::ShaderModule {
    let complete = format!("{header}{body}");
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(complete.into()),
    })
}
