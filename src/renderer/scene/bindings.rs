use crate::renderer::resources::{
    bindless::BindlessResources,
    camera::CameraManager,
    lighting::LightingManager,
};

/// Bind group 0/1/2 for the scene pass.
pub struct SceneBindings<'a> {
    pub bindless: &'a BindlessResources,
    pub camera: &'a CameraManager,
    pub lighting: &'a LightingManager,
}

impl SceneBindings<'_> {
    pub fn bind(&self, device: &wgpu::Device, queue: &wgpu::Queue, rp: &mut wgpu::RenderPass<'_>) {
        let bindless_bind_group = self.bindless.get_bind_group(device);
        rp.set_bind_group(0, &bindless_bind_group, &[]);
        rp.set_bind_group(1, self.camera.prepare(queue), &[]);
        rp.set_bind_group(2, self.lighting.get_bind_group(), &[]);
    }
}
