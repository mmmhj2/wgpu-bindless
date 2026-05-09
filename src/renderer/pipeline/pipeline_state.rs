use std::collections::HashMap;
use wgpu::{ColorTargetState, ColorWrites, PipelineLayoutDescriptor};

use crate::renderer::{device_interface::DeviceInterface, mesh::vertex_types::VertexType, pipeline::rasterizer_pipeline::{RasterizerPipeline, RasterizerPipelineDescriptor}};

pub struct PipelineStates {
    rasterizer_pipeline_layouts: HashMap<String, wgpu::PipelineLayout>,
    rasterizer_pipelines: HashMap<String, RasterizerPipeline>
}

impl PipelineStates {
    pub fn prepare_default_pipelines (device: &DeviceInterface) -> Self {

        let mut rasterizer_pipeline_layouts: HashMap<String, wgpu::PipelineLayout> = HashMap::new();
        let mut rasterizer_pipelines: HashMap<String, RasterizerPipeline> = HashMap::new();

        let default_shader_module = device.get_device().create_shader_module(
            wgpu::include_wgsl!("default_shaders.wgsl")
        );
        let default_rasterizer_pipeline_layout = device.get_device().create_pipeline_layout(
            &PipelineLayoutDescriptor{
                label: None,
                bind_group_layouts: &[],
                immediate_size: 0
            }
        );

        let targets = [
            Some(ColorTargetState{
                    format: device.get_default_texture_format(),
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: ColorWrites::ALL
            })
        ];

        let default_rasterizer_pipeline_descriptor = RasterizerPipelineDescriptor::from((
            wgpu::VertexState{
                module: &default_shader_module,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: VertexType::get_vertex_buffer_layout(VertexType::Basic)
            },
            wgpu::FragmentState{
                module: &default_shader_module,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &targets
            }
        ));

        rasterizer_pipelines.insert(String::from("default"), RasterizerPipeline::new(device, default_rasterizer_pipeline_descriptor, Some(&default_rasterizer_pipeline_layout)));
        rasterizer_pipeline_layouts.insert(String::from("default"), default_rasterizer_pipeline_layout);

        return Self{
            rasterizer_pipeline_layouts,
            rasterizer_pipelines
        }
    }

    /// Query the default pipeline.
    /// It is guaranteed to be valid. 
    /// 
    /// Panic
    /// ---
    /// If the default pipeline cannot be found, the program is considered to be in
    /// invalid state and will panic and terminate.
    pub fn get_default_pipeline(&self) -> &RasterizerPipeline {
        self.query_pipeline(&String::from("default")).expect("Default pipeline not found for some reason.")
    }

    pub fn query_pipeline(&self, name: &String) -> Option<&RasterizerPipeline> {
        self.rasterizer_pipelines.get(name)
    }

    /// Query the default pipeline layout.
    /// It is guaranteed to be valid. 
    /// 
    /// Panic
    /// ---
    /// If the default pipeline layout cannot be found, the program is considered to be in
    /// invalid state and will panic and terminate.
    pub fn get_default_pipeline_layout(&self) -> &wgpu::PipelineLayout {
        self.query_pipeline_layout(&String::from("default")).expect("Default pipeline layout not found for some reason.")
    }

    pub fn query_pipeline_layout(&self, name: &String) -> Option<&wgpu::PipelineLayout> {
        self.rasterizer_pipeline_layouts.get(name)
    }
}
