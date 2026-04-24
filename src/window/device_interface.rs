use std::{error, sync::Arc};
use winit::window::Window;

/// Interface to WGPU hardware device
pub struct DeviceInterface {
    pub device              : wgpu::Device,
    pub surface             : wgpu::Surface<'static>,
    pub graphics_queue      : wgpu::Queue,
    pub config              : wgpu::SurfaceConfiguration,
    pub presentation_ready  : bool
}

impl DeviceInterface {
    /// Create a device interface fitting the window.
    /// 
    /// Swapchain creation (i.e. surface configuration) is delayed until render
    /// time. Check `DeviceInterface::presentation_ready` to query its state.
    /// 
    /// TODO: replace `Box<dyn error::Error>` by anyhow.
    pub async fn new(window : Arc<Window>) -> Result<DeviceInterface, Box<dyn error::Error>> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        // Physical device creation
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;
        
        // Logical device creation
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        // Set up swapchain
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())          // Enforce sRGB texture format
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };


        Ok(DeviceInterface {
            device,
            surface,
            config,
            graphics_queue: queue,
            presentation_ready: false
        })
    }

    /// Set up the surface with specified configuration.
    pub fn create_surface(&mut self, width : u32, height : u32) -> () {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.presentation_ready = true;
        }
    }
}
