use std::collections::HashMap;

use crate::renderer::{device_interface::DeviceInterface, mesh::vertex_types::VertexType, pipeline::rasterizer_pipeline::{RasterizerPipeline, RasterizerPipelineDescriptorBuilder}};

macro_rules! include_bindless_wgsl {
    ($($token:tt)*) => {
        {
            let header = include_str!("./shaders/bindless_header.wgsl").to_owned();
            let body = include_str!($($token)*);
            let complete: String = header + body;
            wgpu::ShaderModuleDescriptor {
                label: Some($($token)*),
                source: wgpu::ShaderSource::Wgsl(
                    std::borrow::Cow::from(complete)
                ),
            }
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PipelineKey {
    Default,
    CookTorrancePbr
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PipelineLayoutKey {
    Default
}

/// Cache for main pipelines.
/// 
/// This cache is responsible for main passes (i.e. forward lighting, shadow maps).
/// Pipelines for post-processing passes are handled by the passes themselves.
pub(crate) struct PipelineCache {
    // TODO: Use proper interfaces instead of `pub(super)`
    pub(super) pipelines: HashMap<PipelineKey, RasterizerPipeline>,
    pub(super) layouts: HashMap<PipelineLayoutKey, wgpu::PipelineLayout>
}

impl PipelineCache {
    /// Build a new pipeline cache, containing only default pipeline and layout.
    /// 
    /// Parameters
    /// ---
    /// - `color_format` : Color attachment format.
    /// - `depth_format` : Depth stencil attachment format.
    /// - `default_bgl` : Bind group layout for default layout. Generally assembled
    /// from bindless resources, camera, lighting and mesh resources.
    pub fn new(
        di: &DeviceInterface,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        default_bgl: &[Option<&wgpu::BindGroupLayout>]
    ) -> Self {
        let default_shader_module = di.get_device().create_shader_module(
            include_bindless_wgsl!("./shaders/default_shaders.wgsl")
        );
        let default_rasterizer_pipeline_layout = di.get_device().create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor{
                label: None,
                bind_group_layouts: default_bgl,
                immediate_size: (std::mem::size_of::<u32>() * 3) as u32
            }
        );
        let targets = [
            Some(wgpu::ColorTargetState{
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL
            })
        ];
        let default_rasterizer_pipeline_descriptor =
            RasterizerPipelineDescriptorBuilder::new(wgpu::VertexState{
                module: &default_shader_module,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: VertexType::get_vertex_buffer_layout(VertexType::Basic)
            }).set_fragment_state(wgpu::FragmentState{
                module: &default_shader_module,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &targets
            }).set_depth_stencil_state(wgpu::DepthStencilState{
                format: depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default()
            }).build();

        

        let mut pipelines = HashMap::new();
        let mut layouts = HashMap::new();
        
        pipelines.insert(PipelineKey::Default, RasterizerPipeline::new(di, default_rasterizer_pipeline_descriptor, Some(&default_rasterizer_pipeline_layout)));
        layouts.insert(PipelineLayoutKey::Default, default_rasterizer_pipeline_layout);

        PipelineCache { pipelines, layouts }
    }

    /// Prepare the Cook-Torrance PBR pipeline.
    /// 
    /// This pipeline shares the same layout as the default pipeline.
    pub fn prepare_pbr_pipeline(&mut self,
        di: &DeviceInterface,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) {
        let layout = self.layouts.get(&PipelineLayoutKey::Default).expect("default layout should be ready");
        let targets = [
            Some(wgpu::ColorTargetState{
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL
            })
        ];
        let pbr_shader_module = di.get_device().create_shader_module(
            include_bindless_wgsl!("./shaders/cook_torrance.wgsl")
        );
        let pbr_rasterizer_pipeline_descriptor = 
            RasterizerPipelineDescriptorBuilder::new(wgpu::VertexState{
                module: &pbr_shader_module,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: VertexType::get_vertex_buffer_layout(VertexType::Basic)
            }).set_fragment_state(wgpu::FragmentState{
                module: &pbr_shader_module,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &targets
            }).set_depth_stencil_state(wgpu::DepthStencilState{
                format: depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default()
            }).build();
        self.pipelines.insert(PipelineKey::CookTorrancePbr, RasterizerPipeline::new(di, pbr_rasterizer_pipeline_descriptor, Some(&layout)));
    }
}
