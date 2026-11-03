use std::sync::Arc;
use winit::{application::ApplicationHandler, error::EventLoopError, event::{KeyEvent, WindowEvent}, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::Window};
use crate::renderer::window::RendererState;

pub struct DefaultAppHandler<RState : RendererState + 'static> {
    pub state: Option<RState>,
    /// Stop the event loop after this many successful render callbacks.
    ///
    /// `None` runs until the window is closed or a render fails, which is what
    /// the examples want. Tests use `Some(n)` so that `cargo test` cannot block
    /// on the event loop forever.
    max_frames: Option<u32>,
    /// Number of [`RendererState::render`] calls that returned `Ok`.
    rendered_frames: u32,
}

impl<RState : RendererState + 'static> DefaultAppHandler<RState> {
    pub fn new() -> Self {
        Self {
            state: None,
            max_frames: None,
            rendered_frames: 0,
        }
    }

    /// Create a handler that stops the event loop after `max_frames` successful
    /// [`RendererState::render`] calls.
    ///
    /// This is meant for tests and other automation: examples keep using
    /// [`Self::new`] and run until the user closes the window.
    ///
    /// Panics
    /// ---
    /// Panics when `max_frames` is zero.
    pub fn with_max_frames(max_frames: u32) -> Self {
        assert!(max_frames > 0, "the frame limit must be at least 1");

        Self {
            state: None,
            max_frames: Some(max_frames),
            rendered_frames: 0,
        }
    }

    /// Number of successful render callbacks since the handler was created.
    pub fn frames_rendered(&self) -> u32 {
        self.rendered_frames
    }
}

/// Build a winit event loop for a windowed application or test.
///
/// winit restricts event-loop creation to the main thread by default. Windows
/// and Linux expose an `any_thread` builder extension, which is what allows
/// `cargo test` to create the loop on the worker thread it runs every test on.
///
/// On Linux the extension is declared for X11, but winit stores the flag in the
/// platform-specific attributes shared by the X11 and Wayland backends, so it
/// also applies when Wayland is selected. The relevant winit features (`x11`
/// and `wayland`) are part of the crate's default features.
///
/// Other platforms keep winit's default policy (macOS requires the main
/// thread). The renderer is Vulkan-only and the windowed tests do not target
/// them.
pub fn build_event_loop<U: 'static>() -> Result<EventLoop<U>, EventLoopError> {
    // `mut` is only needed when one of the platform extensions below is
    // compiled in.
    #[allow(unused_mut)]
    let mut builder = EventLoop::<U>::with_user_event();

    #[cfg(target_os = "windows")]
    {
        use winit::platform::windows::EventLoopBuilderExtWindows;
        builder.with_any_thread(true);
    }

    #[cfg(target_os = "linux")]
    {
        use winit::platform::x11::EventLoopBuilderExtX11;
        builder.with_any_thread(true);
    }

    builder.build()
}

impl<RState : RendererState + 'static> ApplicationHandler<RState> for DefaultAppHandler<RState> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes();
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        self.state = Some(pollster::block_on(RState::new(window.clone())));
        // Do not rely on the platform synthesizing an initial expose/redraw
        // event: windowed tests need at least one frame to reach the limit.
        window.request_redraw();
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
                    Ok(()) => {
                        self.rendered_frames += 1;
                        if self.max_frames.is_some_and(|max| self.rendered_frames >= max) {
                            event_loop.exit();
                        }
                    }
                    Err(()) => {
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

