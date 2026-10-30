use std::num::NonZeroU32;

use crate::renderer::context::surface_context::SurfaceContext;


/// Unified manager for color and depth render targets.
pub struct Backbuffers {
    /// Color back buffer for headless mode. Currently unused.
    #[allow(dead_code)]
    color: Option<wgpu::Texture>,
    depth_format: wgpu::TextureFormat,
    depth: Option<wgpu::Texture>
}

pub struct FrameContext {
    surface_texture: wgpu::SurfaceTexture,
    color: wgpu::TextureView,
    depth: wgpu::TextureView,
    size: (NonZeroU32, NonZeroU32)
}

impl Backbuffers{

    pub fn new(depth_format: wgpu::TextureFormat) -> Self {
        Self { color: None, depth_format, depth: None }
    }

    pub fn new_frame(&mut self, device: &wgpu::Device, surface_context: &mut SurfaceContext) -> Option<FrameContext> {
        let color_surface = surface_context.try_present(device)?;
        let color_view = color_surface.texture.create_view(&wgpu::TextureViewDescriptor{..Default::default()});

        // If color surface returns successfully, size is always correct.
        let size = surface_context.get_current_size().expect("size should be ready for surface");
        
        let needs_recreation = if let Some(t) = &self.depth {
            let (width, height) = (t.size().width, t.size().height);
            !(width == size.0.get() && height == size.1.get())
        } else {
            true
        };

        if needs_recreation {
            self.depth = Some(device.create_texture(&wgpu::TextureDescriptor{
                label: Some("depth backbuffer"),
                size: wgpu::Extent3d { width: size.0.into(), height: size.1.into(), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.depth_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[self.depth_format],
            }))
        };

        Some(FrameContext {
            surface_texture: color_surface,
            color: color_view,
            depth: self.depth.as_ref().expect("depth texture should be ready").create_view(&wgpu::TextureViewDescriptor{..Default::default()}),
            size
        })
    }
}

impl FrameContext {
    pub fn present(self) {
        self.surface_texture.present();
    }

    pub fn get_color_view(&self) -> &wgpu::TextureView { &self.color }
    pub fn get_depth_view(&self) -> &wgpu::TextureView { &self.depth }
    pub fn get_size(&self) -> (NonZeroU32, NonZeroU32) { self.size }
}
