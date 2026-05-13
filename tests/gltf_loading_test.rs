mod test_fixture;

#[cfg(test)]
#[cfg(target_os = "windows")]
mod test {
    use std::sync::Arc;
    use rust_renderer::renderer::{device_interface::DeviceInterface, mesh::{DrawableMesh, instanced_mesh::InstancedMeshInstance}, pipeline::{pipeline_state::PipelineStates, rasterizer_pipeline::UseRasterizerPipeline}, window::RendererState};
    use winit::{event_loop::EventLoop, window::Window};

    use crate::test_fixture;

    pub struct State {
        window: Arc<Window>,
        device: DeviceInterface,
        pipeline: PipelineStates,
        gltf_models: Vec<InstancedMeshInstance>
    }

    impl RendererState for State {
        async fn new(window: Arc<Window>) -> Self {
            let device = DeviceInterface::new(window.clone()).await.unwrap();
            let mut pipeline = PipelineStates::prepare_default_pipelines(&device);

            let (document, buffers, images) = gltf::import("resource/test_two_cubes.glb").expect("Cannot open glb file.");
            
            let mut gltf_models = Vec::new();
            for mesh in document.meshes() {
                let instances = InstancedMeshInstance::create_from_gltf(
                    &device,
                    pipeline.get_bindless_resource_manager_mut(),
                    &mesh,
                    &buffers,
                    &images);
                gltf_models.extend(instances);
            }

            Self {
                window: window.clone(),
                device,
                pipeline,
                gltf_models
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
                    panic!("Lost device");
                }
            };

            let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = self.device.get_device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

            {
                let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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

                self.pipeline.prepare_render_pass(self.get_device_interface(), &mut rp);
                rp.set_rasterizer_pipeline(self.pipeline.get_default_pipeline());
                for m in &self.gltf_models {
                    m.draw(&mut rp);
                }
            }

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

    #[test]
    fn gltf_loading_test() {
        env_logger::init();

        // We need to create the eventloop from the test thread.
        #[cfg(target_os = "windows")]
	    use winit::platform::windows::EventLoopBuilderExtWindows;

        let event_loop = EventLoop::<State>::with_user_event().with_any_thread(true).build().expect("Failed to build event loop.");
        let mut app = test_fixture::App::new();
        event_loop.run_app(&mut app).unwrap();
    }
}
