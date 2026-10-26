use std::{error::Error, num::NonZeroU32, sync::Arc};

use winit::window::Window;

enum SurfaceState {
    Created,
    Configured {
        color_format: wgpu::TextureFormat
    },
    ReadyForPresentation {
        color_format: wgpu::TextureFormat,
        size: (NonZeroU32, NonZeroU32)
    }
}

/// Wrapper of WGPU surfaces, acting as the backbuffer of the renderer
pub struct SurfaceContext<'a> {
    surface: wgpu::Surface<'a>,
    window: Arc<Window>,
    state: SurfaceState
}

impl<'a> SurfaceContext<'a> {
    /// Create the surface context by initializing the surface object.
    /// 
    /// As per WGPU specification, contents of the surface (i.e. swapchain
    /// images) are not created at this point.
    /// 
    /// After creating the [`SurfaceContext`], you should then set the color
    /// format, possibly via [`Self::select_color_format`], and finally configure
    /// the surface for presentation via [`Self::configure`].
    pub fn new(instance: &wgpu::Instance, window: Arc<Window>) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            surface: instance.create_surface(window.clone())?,
            window,
            state: SurfaceState::Created,
        })
    }

    /// Select the format of the surface according to the adapter.
    /// 
    /// Contents of the surface are still not created at this point.
    /// 
    /// Invalidates the current surface, regardless of whether the format
    /// is identical, meaning that you must manually call [`configure`] again.
    pub fn select_color_format(&mut self, adapter: &wgpu::Adapter, request_srgb_surface: bool) {

        let surface_caps = self.surface.get_capabilities(&adapter);
        let surface_format = if request_srgb_surface {
            surface_caps.formats.iter()
                .find(|f| f.is_srgb()) 
        } else {
            surface_caps.formats.iter()
                .find(|f| !f.is_srgb())
        }.copied()
        .unwrap_or(surface_caps.formats[0]);
        
        self.state = SurfaceState::Configured { color_format: surface_format };
    }

    /// Configure (or reconfigure) the surface to prepare it for the presentation.
    /// 
    /// Creates the current color surface and depth backbuffer (if applicable).
    /// 
    /// If the depth backbuffer format is not set before the call, the depth
    /// buffer will not be created.
    pub fn configure_with_size(&mut self, device: &wgpu::Device, size: (NonZeroU32, NonZeroU32)) {
        let color_format = match self.state {
            SurfaceState::Created => panic!("surface has not been configured"),
            SurfaceState::Configured { color_format } => color_format,
            SurfaceState::ReadyForPresentation { color_format, size: _ } => color_format,
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: color_format,
            width: size.0.into(),
            height: size.1.into(),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![color_format.add_srgb_suffix(), color_format.remove_srgb_suffix()],
        };
        self.surface.configure(device, &config);

        self.state = SurfaceState::ReadyForPresentation { color_format, size };
    }

    pub fn configure(&mut self, device: &wgpu::Device) {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            log::info!("Zero sized window found. The window may be minimized.");
        } else {
            self.configure_with_size(
                device,
                (
                    size.width.try_into().expect("width of the window should be greater than zero"),
                    size.height.try_into().expect("height of the window should be greater than zero")
                ));
        }
        
    }

    pub fn surface(&'_ self) -> &'_ wgpu::Surface<'_> { &self.surface }

    /// Tries to obtain a [`wgpu::SurfaceTexture`] for presenting.
    /// 
    /// `None` might be returned, indicating one of the following:
    /// - The surface is incorrectly configured;
    /// - Current frame is skipped due to missed vsync or occlusion;
    /// - The swap chain is suboptimal or outdated and is being recreated.
    /// 
    /// Panics
    /// ---
    /// If a device lost is encountered, this function panicks.
    /// I've never encountered any app that handles device lost gracefully, so panicking is sufficent.
    pub fn try_present(&mut self, device: &wgpu::Device) -> Option<wgpu::SurfaceTexture> {
        let surf = match &self.state {
            SurfaceState::ReadyForPresentation { color_format:_, size: _ } => self.surface.get_current_texture(),
            _ => return None
        };

        match surf {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => Some(surface_texture),
            wgpu::CurrentSurfaceTexture::Suboptimal(_surface_texture) => 
                { self.configure(device); None },
            wgpu::CurrentSurfaceTexture::Outdated => { self.configure(device); None },
            wgpu::CurrentSurfaceTexture::Timeout |
            wgpu::CurrentSurfaceTexture::Occluded => None,
            wgpu::CurrentSurfaceTexture::Validation => 
            {
                #[cfg(debug_assertions)]
                panic!("WGPU validation error");
                #[cfg(not(debug_assertions))]
                None
            },
            wgpu::CurrentSurfaceTexture::Lost => panic!("device lost"),
        }
    }

    pub fn is_ready(&self) -> bool { 
        match &self.state {
            SurfaceState::ReadyForPresentation { color_format: _, size: _ } => true,
            _ => false
        }
     }

    pub fn get_current_size(&self) -> Option<(NonZeroU32, NonZeroU32)> {
        match &self.state {
            SurfaceState::ReadyForPresentation { color_format:_, size } => Some(*size),
            _ => None
        }
    }

    /// Get the format of the color surface.
    /// 
    /// Note: the color format may be returned before the actual swap chain creation happens.
    pub fn get_color_format(&self) -> Option<wgpu::TextureFormat> {
        match &self.state {
            SurfaceState::Configured { color_format } => Some(*color_format),
            SurfaceState::ReadyForPresentation { color_format, size: _ } => Some(*color_format),
            _ => None
        }
    }
}
