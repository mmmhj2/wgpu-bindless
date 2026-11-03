#[cfg(any(target_os = "windows", target_os = "linux"))]
mod common;

#[cfg(test)]
#[cfg(any(target_os = "windows", target_os = "linux"))]
mod test {
    use std::sync::Arc;
    use rust_renderer::renderer::{
        orchestrator::{RenderOrchestrator, RendererConfig},
        window::RendererState,
    };
    use winit::window::Window;
    use crate::common;

    const FRAMES_TO_RENDER: u32 = 3;

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

    #[test]
    fn basic_functionality_test() {
        env_logger::init();

        let frames = common::run_windowed::<State>(FRAMES_TO_RENDER);
        assert_eq!(
            frames,
            Some(FRAMES_TO_RENDER),
            "the event loop should stop after the configured frame limit"
        );
    }
}
