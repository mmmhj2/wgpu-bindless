use std::{collections::HashMap, num::NonZero, borrow::Cow};
use wgpu::{ColorTargetState, ColorWrites, DepthBiasState, PipelineLayoutDescriptor};

use crate::renderer::{asset_registry::GpuAssetRegistry, device_interface::DeviceInterface, mesh::{mesh_manager::MeshManager, vertex_types::VertexType}, pipeline::{bindless_resource_manager::BindlessResourceManager, camera::{CameraManager, CameraPerspective, HasViewProjectionMatrix}, rasterizer_pipeline::{RasterizerPipeline, RasterizerPipelineDescriptor}}};

macro_rules! include_bindless_wgsl {
    ($($token:tt)*) => {
        {
            let header = include_str!("./shaders/bindless_header.wgsl").to_owned();
            let body = include_str!($($token)*);
            let complete: String = header + body;
            wgpu::ShaderModuleDescriptor {
                label: Some($($token)*),
                source: wgpu::ShaderSource::Wgsl(
                    Cow::from(complete)
                ),
            }
        }
    };
}

pub struct PipelineStates {
    asset_registry: GpuAssetRegistry,
    cameras: CameraManager,
    mesh_manager: MeshManager,
    rasterizer_pipeline_layouts: HashMap<String, wgpu::PipelineLayout>,
    rasterizer_pipelines: HashMap<String, RasterizerPipeline>
}

impl PipelineStates {
    pub fn prepare_default_pipelines (di: & DeviceInterface, format: wgpu::TextureFormat) -> Self {

        let asset_registry = GpuAssetRegistry::new(di);
        let cameras = CameraManager::new(di);
        let mesh_manager = MeshManager::new(di);
        let mut rasterizer_pipeline_layouts: HashMap<String, wgpu::PipelineLayout> = HashMap::new();
        let mut rasterizer_pipelines: HashMap<String, RasterizerPipeline> = HashMap::new();

        let default_shader_module = di.get_device().create_shader_module(
            include_bindless_wgsl!("./shaders/default_shaders.wgsl")
        );
        let default_rasterizer_pipeline_layout = di.get_device().create_pipeline_layout(
            &PipelineLayoutDescriptor{
                label: None,
                bind_group_layouts: &[
                        Some(asset_registry.get_bindless_resource_manager().get_bind_group_layout()),
                        Some(cameras.get_bind_group_layout()),
                        Some(mesh_manager.get_bind_group_layout())
                    ],
                immediate_size: (std::mem::size_of::<u32>() * 3) as u32
            }
        );
        let targets = [
            Some(ColorTargetState{
                    format: format,
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
            },
            wgpu::DepthStencilState{
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default()
            }
        ));

        let pbr_shader_module = di.get_device().create_shader_module(include_bindless_wgsl!("./shaders/cook_torrance.wgsl"));
        let pbr_rasterizer_pipeline_descriptor = RasterizerPipelineDescriptor::from((
            wgpu::VertexState{
                module: &pbr_shader_module,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: VertexType::get_vertex_buffer_layout(VertexType::Basic)
            },
            wgpu::FragmentState{
                module: &pbr_shader_module,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &targets
            },
            wgpu::DepthStencilState{
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default()
            }
        ));

        rasterizer_pipelines.insert(String::from("default"), RasterizerPipeline::new(di, default_rasterizer_pipeline_descriptor, Some(&default_rasterizer_pipeline_layout)));
        rasterizer_pipelines.insert(String::from("cook_torrance"), RasterizerPipeline::new(di, pbr_rasterizer_pipeline_descriptor, Some(&default_rasterizer_pipeline_layout)));
        rasterizer_pipeline_layouts.insert(String::from("default"), default_rasterizer_pipeline_layout);

        return Self{
            asset_registry,
            cameras,
            mesh_manager,
            rasterizer_pipeline_layouts,
            rasterizer_pipelines
        }
    }

    /// Prepare to use stored render pipelines.
    /// 
    /// It first sets the bindless bind groups to the render pass.
    /// Then it pushes the camera matrix of the current active camera to GPU and bind its uniform buffer.
    pub fn prepare_render_pass(&self, di: &DeviceInterface, rp: &mut wgpu::RenderPass) -> () {
        let bindless_bind_group = self.asset_registry.get_bindless_resource_manager().get_bind_group(di.get_device());
        rp.set_bind_group(0, &bindless_bind_group, &[]);
        rp.set_bind_group(1, self.cameras.prepare(di), &[]);
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

    pub fn get_camera_manager(&self) -> &CameraManager { &self.cameras }
    pub fn set_active_camera(&mut self, camera: CameraPerspective) { self.cameras.set_active_camera(camera); }

    pub fn get_mesh_manager(&self) -> &MeshManager { &self.mesh_manager }

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

    pub fn get_bindless_resource_manager(&self) -> &BindlessResourceManager { self.asset_registry.get_bindless_resource_manager() }
    pub fn get_bindless_resource_manager_mut(&mut self) -> &mut BindlessResourceManager { self.asset_registry.get_bindless_resource_manager_mut() }
    pub fn get_asset_registry_mut(&mut self) -> &mut GpuAssetRegistry { &mut self.asset_registry }
}
