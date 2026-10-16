 use rust_renderer::{app::DefaultAppHandler, renderer::window::DefaultRendererState};
use winit::
    event_loop::EventLoop 
;

pub fn run() -> Result<(), winit::error::EventLoopError> {
    // To enable the logger when running from Cargo, set `$env:RUST_LOG` to debug or something else.
    env_logger::init();

    let event_loop = EventLoop::<DefaultRendererState>::with_user_event().build()?;
    let mut app = DefaultAppHandler::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}

fn main() {
    run().unwrap();
}
