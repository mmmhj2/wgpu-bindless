use std::{collections::HashMap, num::NonZero, borrow::Cow};
use wgpu::{ColorTargetState, ColorWrites, DepthBiasState, PipelineLayoutDescriptor};

use crate::renderer::{asset_registry::GpuAssetRegistry, device_interface::DeviceInterface, mesh::{mesh_manager::MeshManager, vertex_types::VertexType}, pipeline::{bindless_resource_manager::BindlessResourceManager, camera::{CameraManager, CameraPerspective, HasViewProjectionMatrix}, pipeline_cache::{PipelineCache, PipelineKey, PipelineLayoutKey}, rasterizer_pipeline::{RasterizerPipeline, RasterizerPipelineDescriptor, RasterizerPipelineDescriptorBuilder}}};

pub struct PipelineStates {
    asset_registry: GpuAssetRegistry,
    cameras: CameraManager,
    mesh_manager: MeshManager,
    cache: PipelineCache
}

impl PipelineStates {
    pub fn prepare_default_pipelines (di: & DeviceInterface, format: wgpu::TextureFormat) -> Self {

        let asset_registry = GpuAssetRegistry::new(di);
        let cameras = CameraManager::new(di);
        let mesh_manager = MeshManager::new(di);
        let bind_group_layouts = [
            Some(asset_registry.get_bindless_resource_manager().get_bind_group_layout()),
            Some(cameras.get_bind_group_layout()),
            Some(mesh_manager.get_bind_group_layout())
        ];
        let mut cache = PipelineCache::new(di, format, wgpu::TextureFormat::Depth32Float, &bind_group_layouts);
        cache.prepare_pbr_pipeline(di, format, wgpu::TextureFormat::Depth32Float);

        return Self{
            asset_registry,
            cameras,
            mesh_manager,
            cache
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

    /// Query the default pipeline, which is guaranteed to be valid.
    /// 
    /// Panic
    /// ---
    /// If the default pipeline cannot be found, the program is considered to be in
    /// invalid state and will panic and terminate.
    pub fn get_default_pipeline(&self) -> &RasterizerPipeline {
        self.query_pipeline(PipelineKey::Default).expect("default pipeline should always be ready.")
    }

    pub fn query_pipeline(&self, key: PipelineKey) -> Option<&RasterizerPipeline> {
        self.cache.pipelines.get(&key)
    }

    pub fn get_camera_manager(&self) -> &CameraManager { &self.cameras }
    pub fn set_active_camera(&mut self, camera: CameraPerspective) { self.cameras.set_active_camera(camera); }

    pub fn get_mesh_manager(&self) -> &MeshManager { &self.mesh_manager }

    /// Query the default pipeline layout which is guaranteed to be valid. 
    /// 
    /// Panic
    /// ---
    /// If the default pipeline layout cannot be found, the program is considered to be in
    /// invalid state and will panic and terminate.
    pub fn get_default_pipeline_layout(&self) -> &wgpu::PipelineLayout {
        self.query_pipeline_layout(PipelineLayoutKey::Default).expect("Default pipeline layout should always be ready.")
    }

    pub fn query_pipeline_layout(&self, key: PipelineLayoutKey) -> Option<&wgpu::PipelineLayout> {
        self.cache.layouts.get(&key)
    }

    pub fn get_bindless_resource_manager(&self) -> &BindlessResourceManager { self.asset_registry.get_bindless_resource_manager() }
    pub fn get_bindless_resource_manager_mut(&mut self) -> &mut BindlessResourceManager { self.asset_registry.get_bindless_resource_manager_mut() }
    pub fn get_asset_registry_mut(&mut self) -> &mut GpuAssetRegistry { &mut self.asset_registry }
}
