use std::sync::Arc;
use rust_renderer::renderer::window::RendererState;
use winit::{
    application::ApplicationHandler, event::*, event_loop::{ActiveEventLoop, EventLoop}, window::Window
};

pub struct App<RState : RendererState + 'static> {
    state: Option<RState>,
}

impl<RState : RendererState + 'static> App<RState> {
    pub fn new() -> Self {
        Self {
            state: None,
        }
    }
}

impl<RState : RendererState + 'static> ApplicationHandler<RState> for App<RState> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes();
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        self.state = Some(pollster::block_on(RState::new(window)));
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: RState) {
        self.state = Some(event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                // state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(_) => {
                        event_loop.exit();
                    }
                }
            }
            _ => {}
        }
    }
}

pub fn run_test<RState: rust_renderer::renderer::window::RendererState + 'static>() {
    env_logger::init();

    let event_loop = EventLoop::<RState>::with_user_event().build().expect("Failed to build event loop.");
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
