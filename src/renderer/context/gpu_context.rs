use std::error::Error;

use crate::renderer::pipeline::{bindless_resource_manager::MAX_TEXTURE_SLOTS, sampler_manager::MAX_SAMPLER_SLOTS};

/// Abstraction over WGPU hardware related objects: Device and queue.
pub struct GpuContext {
    instance: wgpu::Instance,
    adapter: Option<wgpu::Adapter>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>
}

impl GpuContext {
    /// Create a new context with only instance initialized.
    /// 
    /// Surfaces must be created after the instance. Then the surface must
    /// be used via [`Self::create_device()`] to initialize the adapter, device
    /// and queue.
    pub fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        Self { instance, adapter: None, device: None, queue: None }
    }

    /// Create the logical device that WGPU works on.
    /// 
    /// Forwards any error happened during WGPU initialization.
    /// 
    /// Parameters
    /// ---
    /// - `surface`: surface that the device works on.
    pub async fn create_device(&mut self, surface: Option<&wgpu::Surface<'_>>) -> Result<(), Box<dyn Error>> {
        let adapter = self.instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface,
                force_fallback_adapter: false,
            })
            .await?;
        
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
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
                    max_immediate_size: 80,
                    ..wgpu::Limits::default()
                },
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            }
        ).await?;

        self.adapter = Some(adapter);
        self.device = Some(device);
        self.queue = Some(queue);

        Ok(())
    }

    pub fn instance(&self) -> &wgpu::Instance { &self.instance }
    
    /// Get the WGPU device.
    /// 
    /// Panicks if the device has not been initialized via [`Self::create_device()`]
    pub fn device(&self) -> &wgpu::Device { self.device.as_ref().unwrap() }

    /// Get the WGPU queue.
    /// 
    /// Panicks if the device has not been initialized via [`Self::create_device()`]
    pub fn queue(&self) -> &wgpu::Queue { self.queue.as_ref().unwrap() }

    pub fn try_device(&self) -> Option<&wgpu::Device> { self.device.as_ref() }
    pub fn try_queue(&self) -> Option<&wgpu::Queue> { self.queue.as_ref() }

    // panicking variant is not provided as querying the adapter is very rare.
    pub fn try_adapter(&self) -> Option<&wgpu::Adapter> { self.adapter.as_ref() }
}
