use crate::renderer::{
    pipeline::library::PipelineLibrary,
    resources::{
        asset_cache::GpuAssetCache,
        bindless::BindlessResources,
        camera::CameraManager,
        lighting::LightingManager,
        mesh_layout::MeshLayout,
    },
    scene::bindings::SceneBindings,
};

/// Per-frame split borrow of the scene renderer state.
///
/// This type does not own anything; the orchestrator owns the individual
/// resources and hands out this transient view while recording a frame.
pub struct SceneContext<'a> {
    pub bindless: &'a mut BindlessResources,
    pub camera: &'a mut CameraManager,
    pub lighting: &'a mut LightingManager,
    pub mesh_layout: &'a MeshLayout,
    pub assets: &'a mut GpuAssetCache,
    pub pipelines: &'a mut PipelineLibrary,
}

impl SceneContext<'_> {
    /// Bind the global bind groups used by the scene pipelines.
    pub fn bind_globals(&self, device: &wgpu::Device, queue: &wgpu::Queue, rp: &mut wgpu::RenderPass<'_>) {
        SceneBindings {
            bindless: self.bindless,
            camera: self.camera,
            lighting: self.lighting,
        }
        .bind(device, queue, rp);
    }
}
