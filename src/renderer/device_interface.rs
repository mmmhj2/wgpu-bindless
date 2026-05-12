use std::{error, sync::Arc};
use winit::window::Window;

use crate::renderer::pipeline::bindless_resource_manager::{MAX_SAMPLER_SLOTS, MAX_TEXTURE_SLOTS};

struct SurfaceDetailInfo {
    surface     : wgpu::Surface<'static>,
    config      : wgpu::SurfaceConfiguration
}

/// Interface to WGPU hardware device
pub struct DeviceInterface {
    device              : wgpu::Device,
    surface             : SurfaceDetailInfo,
    graphics_queue      : wgpu::Queue,
    presentation_ready  : bool
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
                required_features:
                    // For material data in push constants (i.e. immediates)
                    wgpu::Features::IMMEDIATES |
                    // For bindless rendering
                    wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY |
                    wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING |
                    wgpu::Features::TEXTURE_BINDING_ARRAY,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: wgpu::Limits{
                    max_binding_array_sampler_elements_per_shader_stage: MAX_SAMPLER_SLOTS as u32,
                    max_binding_array_elements_per_shader_stage: (MAX_TEXTURE_SLOTS + MAX_SAMPLER_SLOTS) as u32,
                    max_immediate_size: 64,
                    ..wgpu::Limits::default()
                },
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
            surface: SurfaceDetailInfo {
                surface, config
            },
            graphics_queue: queue,
            presentation_ready: false
        })
    }

    pub fn configure_surface(&mut self) {
        self.surface.surface.configure(&self.device, &self.surface.config);
        self.presentation_ready = true;
    }

    /// Set up the surface with specified configuration.
    pub fn create_surface(&mut self, width : u32, height : u32) {
        if width > 0 && height > 0 {
            self.surface.config.width = width;
            self.surface.config.height = height;
            self.configure_surface();
        }
    }

    pub fn get_current_surface_texture(&self) -> wgpu::CurrentSurfaceTexture {
        self.surface.surface.get_current_texture()
    }

    pub fn is_presentation_ready(&self) -> bool { self.presentation_ready }

    /// Request the default color texture format for this device.
    /// This format is selected by the surface creation routine.
    pub fn get_default_texture_format(&self) -> wgpu::TextureFormat {
        self.surface.config.format
    }

    /// Acquire the graphics queue used to submit command encoders
    pub fn get_queue(&self) -> &wgpu::Queue { &self.graphics_queue }

    /// Acquire the logical device
    pub fn get_device(&self) -> &wgpu::Device { &self.device }
}
