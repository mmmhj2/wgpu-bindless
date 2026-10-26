use crate::renderer::pipeline::{
    key::{PipelineKey, PipelineLayoutKey},
    rasterizer_pipeline::{RasterizerPipeline, RasterizerPipelineDescriptorBuilder},
    recipe::{PipelineRecipe, bindless_shader_module},
};

const DEFAULT_SHADER: &str = include_str!("shaders/default_shaders.wgsl");
const COOK_TORRANCE_SHADER: &str = include_str!("shaders/cook_torrance.wgsl");

fn create_forward_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    vertex_layout: &[wgpu::VertexBufferLayout<'static>],
    label: &'static str,
    shader_body: &'static str,
) -> RasterizerPipeline {
    let shader_module = bindless_shader_module(device, label, shader_body);
    let targets = [
        Some(wgpu::ColorTargetState {
            format: color_format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        }),
    ];

    let descriptor = RasterizerPipelineDescriptorBuilder::new(wgpu::VertexState {
        module: &shader_module,
        entry_point: Some("vs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        buffers: vertex_layout,
    })
    .set_fragment_state(wgpu::FragmentState {
        module: &shader_module,
        entry_point: Some("fs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        targets: &targets,
    })
    .set_depth_stencil_state(wgpu::DepthStencilState {
        format: depth_format,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::LessEqual),
        stencil: Default::default(),
        bias: Default::default(),
    })
    .build();

    RasterizerPipeline::new(device, descriptor, Some(layout))
}

pub(crate) struct DefaultRecipe;

impl PipelineRecipe for DefaultRecipe {
    fn key(&self) -> PipelineKey { PipelineKey::Default }
    fn layout_key(&self) -> PipelineLayoutKey { PipelineLayoutKey::Default }

    fn create(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        vertex_layout: &[wgpu::VertexBufferLayout<'static>],
    ) -> RasterizerPipeline {
        create_forward_pipeline(
            device,
            layout,
            color_format,
            depth_format,
            vertex_layout,
            "default pipeline",
            DEFAULT_SHADER,
        )
    }
}

pub(crate) struct CookTorranceRecipe;

impl PipelineRecipe for CookTorranceRecipe {
    fn key(&self) -> PipelineKey { PipelineKey::CookTorrancePbr }
    fn layout_key(&self) -> PipelineLayoutKey { PipelineLayoutKey::Default }

    fn create(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        vertex_layout: &[wgpu::VertexBufferLayout<'static>],
    ) -> RasterizerPipeline {
        create_forward_pipeline(
            device,
            layout,
            color_format,
            depth_format,
            vertex_layout,
            "cook torrance pipeline",
            COOK_TORRANCE_SHADER,
        )
    }
}
