use std::{collections::HashMap, num::NonZero};
use wgpu::{ColorTargetState, ColorWrites, PipelineLayoutDescriptor};

use crate::renderer::{device_interface::DeviceInterface, mesh::vertex_types::VertexType, pipeline::{bindless_resource_manager::BindlessResourceManager, camera::{CameraManager, HasViewProjectionMatrix}, rasterizer_pipeline::{RasterizerPipeline, RasterizerPipelineDescriptor}}};

pub struct PipelineStates {
    bindless_resources: BindlessResourceManager,
    cameras: CameraManager,
    rasterizer_pipeline_layouts: HashMap<String, wgpu::PipelineLayout>,
    rasterizer_pipelines: HashMap<String, RasterizerPipeline>
}

impl PipelineStates {
    pub fn prepare_default_pipelines (di: & DeviceInterface) -> Self {

        let bindless_resources = BindlessResourceManager::new(di);
        let cameras = CameraManager::new(di);
        let mut rasterizer_pipeline_layouts: HashMap<String, wgpu::PipelineLayout> = HashMap::new();
        let mut rasterizer_pipelines: HashMap<String, RasterizerPipeline> = HashMap::new();

        let default_shader_module = di.get_device().create_shader_module(
            wgpu::include_wgsl!("default_shaders.wgsl")
        );
        let default_rasterizer_pipeline_layout = di.get_device().create_pipeline_layout(
            &PipelineLayoutDescriptor{
                label: None,
                bind_group_layouts: &[Some(bindless_resources.get_bind_group_layout()), Some(cameras.get_bind_group_layout())],
                immediate_size: 64
            }
        );

        let targets = [
            Some(ColorTargetState{
                    format: di.get_default_texture_format(),
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

        rasterizer_pipelines.insert(String::from("default"), RasterizerPipeline::new(di, default_rasterizer_pipeline_descriptor, Some(&default_rasterizer_pipeline_layout)));
        rasterizer_pipeline_layouts.insert(String::from("default"), default_rasterizer_pipeline_layout);

        return Self{
            bindless_resources,
            cameras,
            rasterizer_pipeline_layouts,
            rasterizer_pipelines
        }
    }

    /// Prepare to use stored render pipelines.
    /// 
    /// It first sets the bindless bind groups to the render pass.
    /// Then it pushes the camera matrix of the current active camera to GPU and bind its uniform buffer.
    pub fn prepare_render_pass(&self, di: &DeviceInterface, rp: &mut wgpu::RenderPass) -> () {
        let bindless_bind_group = self.bindless_resources.get_bind_group(di.get_device());

        let camera_buffer_size = std::mem::size_of::<[[f32; 4]; 4]>() as u64;
        let camera_buffer = di.get_device().create_buffer(&wgpu::wgt::BufferDescriptor {
            label: None,
            size: camera_buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });
        let camera_bind_group = di.get_device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: self.cameras.get_bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(
                        wgpu::BufferBinding {
                            buffer: &camera_buffer,
                            offset: 0,
                            size: None
                        }
                    ) 
                } 
            ]
        });

        rp.set_bind_group(0, &bindless_bind_group, &[]);
        di.get_queue().write_buffer(&camera_buffer, 0, &self.cameras.get_active_camera().get_vp_matrix_as_u8_arr());
        rp.set_bind_group(1, &camera_bind_group, &[]);
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

    pub fn get_bindless_resource_manager(&self) -> &BindlessResourceManager { &self.bindless_resources }
    pub fn get_bindless_resource_manager_mut(&mut self) -> &mut BindlessResourceManager { &mut self.bindless_resources }
}
