#[cfg(test)]
#[cfg(target_os = "windows")]
mod test {
    use std::sync::Arc;

    use rust_renderer::{
        app::DefaultAppHandler,
        renderer::{
            context::{
                frame_context::Backbuffers, gpu_context::GpuContext,
                surface_context::SurfaceContext,
            },
            window::RendererState,
        },
    };
    use winit::{event_loop::EventLoop, window::Window};

    pub struct State {
        window: Arc<Window>,
        gpu: GpuContext,
        surface: SurfaceContext<'static>,
        backbuffers: Backbuffers,
    }

    impl RendererState<'static> for State {
        async fn new(window: Arc<Window>) -> Self {
            let mut gpu = GpuContext::new();
            let mut surface = SurfaceContext::new(gpu.instance(), window.clone()).unwrap();
            gpu.create_device(Some(surface.surface())).await.unwrap();
            surface.select_color_format(gpu.try_adapter().expect("adapter should be ready"), true);
            surface.configure(gpu.device());

            Self {
                window,
                gpu,
                surface,
                backbuffers: Backbuffers::new(wgpu::TextureFormat::Depth32Float),
            }
        }

        fn render(&mut self) -> Result<(), ()> {
            self.window.request_redraw();

            let Some(frame) = self.backbuffers.new_frame(self.gpu.device(), &mut self.surface) else {
                return Ok(());
            };

            let mut encoder = self.gpu.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

            {
                let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: frame.get_color_view(),
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
            }

            self.gpu.queue().submit(std::iter::once(encoder.finish()));
            frame.present();

            Ok(())
        }

        fn gpu_context(&self) -> &GpuContext {
            &self.gpu
        }

        fn surface_context(&self) -> &SurfaceContext<'static> {
            &self.surface
        }

        fn surface_context_mut(&mut self) -> &mut SurfaceContext<'static> {
            &mut self.surface
        }

        fn backbuffers_mut(&mut self) -> &mut Backbuffers {
            &mut self.backbuffers
        }
    }

    #[test]
    fn basic_functionality_test() {
        env_logger::init();

        // We need to create the eventloop from the test thread.
        #[cfg(target_os = "windows")]
        use winit::platform::windows::EventLoopBuilderExtWindows;

        let event_loop = EventLoop::<State>::with_user_event().with_any_thread(true).build().expect("Failed to build event loop.");
        let mut app = DefaultAppHandler::new();
        event_loop.run_app(&mut app).unwrap();
    }
}
