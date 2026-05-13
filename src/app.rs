use std::sync::Arc;
use crate::renderer::window::{RendererState, DefaultRendererState};
use winit::{
    application::ApplicationHandler, event::*, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::Window
};

pub struct DefaultAppHandler<RState : RendererState + 'static> {
    state: Option<RState>,
}

impl<RState : RendererState + 'static> DefaultAppHandler<RState> {
    pub fn new() -> Self {
        Self {
            state: None,
        }
    }
}

impl<RState : RendererState + 'static> ApplicationHandler<RState> for DefaultAppHandler<RState> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let mut window_attributes = Window::default_attributes();
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        self.state = Some(pollster::block_on(RState::new(window)));
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: RState) {
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
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => match (code, key_state.is_pressed()) {
                (KeyCode::Escape, true) => event_loop.exit(),
                _ => {}
            },
            _ => {}
        }
    }
}


pub fn run() -> Result<(), winit::error::EventLoopError> {

    env_logger::init();

    let event_loop = EventLoop::<DefaultRendererState>::with_user_event().build()?;
    let mut app = DefaultAppHandler::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}
