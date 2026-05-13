
use crate::renderer::{device_interface::DeviceInterface, mesh::{DrawableMesh, immediate_mesh::ImmediateMeshBuilder}, pipeline::{pipeline_state::PipelineStates, rasterizer_pipeline::UseRasterizerPipeline}};

use std::sync::Arc;

use winit::{
    window::Window
};

pub trait RendererState {
    fn get_device_interface(&self) -> &DeviceInterface;
    fn get_device_interface_mut(&mut self) -> &mut DeviceInterface;

    fn resize(&mut self, width: u32, height: u32) {
        self.get_device_interface_mut().create_surface(width, height);
    }
    
    fn new(window: Arc<Window>) -> impl std::future::Future<Output = Self> + Send;
    fn render(&mut self) -> Result<(), ()>;
}

pub struct DefaultRendererState {
    window: Arc<Window>,
    device: DeviceInterface,
    pipelines: PipelineStates
}

impl RendererState for DefaultRendererState {
    async fn new(window: Arc<Window>) -> Self {
        let device = DeviceInterface::new(window.clone()).await.unwrap();
        let pipelines = PipelineStates::prepare_default_pipelines(&device);
        Self {
            window: window.clone(),
            device,
            pipelines
        }
    }

    fn render(&mut self) -> Result<(), ()>{
        self.window.request_redraw();

        // We can't render unless the surface is configured
        if !self.device.is_presentation_ready() {
            return Ok(());
        }
            
        let output = match self.device.get_current_surface_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                self.device.configure_surface();
                surface_texture
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                // Skip this frame
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.device.configure_surface();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                // You could recreate the devices and all resources
                // created with it here, but we'll just bail
                panic!("Lost device");
            }
        };

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.get_device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // Do some draw call here
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            let mut mesh_builder = ImmediateMeshBuilder::new(&self.device, self.pipelines.get_bindless_resource_manager_mut());
            mesh_builder.color3f([1.0, 0.0, 0.0]);
            mesh_builder.vertex3f([0.0,  0.0, 0.5]);
            mesh_builder.color3f([0.0, 1.0, 0.0]);
            mesh_builder.vertex3f([0.0, -0.5, -0.5]);
            mesh_builder.color3f([0.0, 0.0, 1.0]);
            mesh_builder.vertex3f([0.0,  0.5, -0.5]);
            let mesh = mesh_builder.commit(&self.device);

            self.pipelines.prepare_render_pass(&self.device, &mut render_pass);
            render_pass.set_rasterizer_pipeline(self.pipelines.get_default_pipeline());
            mesh.draw(&mut render_pass);
        }

        // submit will accept anything that implements IntoIter
        self.device.get_queue().submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
    
    fn get_device_interface(&self) -> &DeviceInterface {
        &self.device
    }
    
    fn get_device_interface_mut(&mut self) -> &mut DeviceInterface {
        &mut self.device
    }
}
