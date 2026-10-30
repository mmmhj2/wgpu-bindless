#[cfg(test)]
#[cfg(target_os = "windows")]
mod test {
    use std::sync::Arc;

    use rust_renderer::{
        app::DefaultAppHandler,
        renderer::{
            orchestrator::{RenderOrchestrator, RendererConfig},
            window::RendererState,
        },
    };
    use winit::{event_loop::EventLoop, window::Window};

    pub struct State {
        orchestrator: RenderOrchestrator,
    }

    impl RendererState for State {
        async fn new(window: Arc<Window>) -> Self {
            Self {
                orchestrator: RenderOrchestrator::new(window, RendererConfig::default()).await,
            }
        }

        fn resize(&mut self, width: u32, height: u32) {
            self.orchestrator.resize(width, height);
        }

        fn render(&mut self) -> Result<(), ()> {
            self.orchestrator.with_frame(|frame| {
                frame.render_scene(
                    wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    },
                    |_, _, _| {},
                );
            });

            Ok(())
        }
    }

    /// NOTE: this test loops forever! This is not good for CI or regression.
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
