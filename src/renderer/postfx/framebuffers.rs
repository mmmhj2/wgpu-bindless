use std::{cmp::max, num::NonZero};

/// A single-resolution render target usable as a color attachment and a
/// sampled texture.
pub struct Framebuffer {
    format: wgpu::TextureFormat,
    label: &'static str,
    texture: Option<wgpu::TextureView>
}

impl Framebuffer {
    /// Create a framebuffer with the specified format.
    ///
    /// The texture is created lazily by [`Framebuffer::create_texture`].
    pub fn new(format: wgpu::TextureFormat, label: &'static str) -> Self {
        Self { format, label, texture: None }
    }

    pub fn get_format(&self) -> wgpu::TextureFormat { self.format }
    pub fn get_texture(&self) -> Option<&wgpu::TextureView> { self.texture.as_ref() }

    /// The texture view, panicking if it has not been created yet.
    pub fn view(&self) -> &wgpu::TextureView {
        self.texture.as_ref().expect("framebuffer texture should be created")
    }

    pub fn create_texture(&mut self, device: &wgpu::Device, width: NonZero<u32>, height: NonZero<u32>) {
        let texture = device.create_texture(&wgpu::TextureDescriptor{
            label: Some(self.label),
            size: wgpu::Extent3d { width: width.get(), height: height.get(), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[]
        }).create_view(&Default::default());
        self.texture = Some(texture.into());
    }
}

/// A half-resolution mip chain used for screen space effects.
pub struct FramebufferMipChain {
    format: wgpu::TextureFormat,
    texture: Vec<wgpu::TextureView>
}

impl FramebufferMipChain {
    /// Create a new mipchain with the specified format.
    /// 
    /// The mipchain will be empty until [`create_textures()`] is called.
    pub fn new(format: wgpu::TextureFormat) -> Self {
        Self { format, texture: Vec::new() }
    }

    pub fn get_format(&self) -> wgpu::TextureFormat { self.format }
    pub fn get_textures(&self) -> &Vec<wgpu::TextureView> { self.texture.as_ref() }
    pub fn get_mipchain_length(&self) -> Option<NonZero<usize>> { NonZero::new(self.texture.len()) }
    
    /// Create a mipchain from the framebuffer size.
    /// 
    /// The first miplevel of the mipchain will have a size of (full_res_width / 2, full_res_height / 2).
    pub fn create_textures(&mut self, device: &wgpu::Device, full_res_width: NonZero<u32>, full_res_height: NonZero<u32>) {
        self.texture.clear();
        let width = full_res_width.get() / 2;
        let height = full_res_height.get() / 2;
        let miplevels = (max(width, height) as f32).log2().floor() as u32 + 1;

        let texture = device.create_texture(
            &wgpu::TextureDescriptor {
                label: Some("Framebuffer mipchain"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: miplevels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            }
        );

        for i in 0..miplevels {
            let view = texture.create_view(
                &wgpu::TextureViewDescriptor{ base_mip_level: i, mip_level_count: Some(1), ..Default::default() }
            );
            self.texture.push(view.into())
        }
    }
}
