use std::num::NonZero;

use crate::renderer::{
    device_interface::DeviceInterface,
    pipeline::sampler_manager::SamplerManager,
    postfx::framebuffers::Framebuffer,
};

/// A single post-processing pass.
///
/// Passes operate in place on the stack's shared HDR target: for every pass but
/// the final one, `input` and `output` are the same HDR view. The final pass
/// writes to the swapchain image instead.
pub trait PostFxPass {
    /// Human-readable name, used in diagnostics.
    fn name(&self) -> &'static str;
    /// Format this pass expects for its input texture.
    fn input_format(&self) -> wgpu::TextureFormat;
    /// Format this pass writes to its output texture.
    fn output_format(&self) -> wgpu::TextureFormat;
    /// Called whenever the render target size changes.
    ///
    /// Passes allocate size-dependent resources here instead of polling the
    /// surface configuration every frame.
    fn resize(&mut self, _di: &DeviceInterface, _width: NonZero<u32>, _height: NonZero<u32>) {}
    /// Record this pass into `ce`, reading `input` and writing `output`.
    fn apply(
        &mut self,
        di: &DeviceInterface,
        samplers: &mut SamplerManager,
        ce: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView
    );
}

/// Owns the HDR working target and an ordered chain of post-processing passes.
///
/// The scene is rendered into [`PostFxStack::get_scene_framebuffer`]; the passes
/// are then applied in insertion order by [`PostFxStack::run`].
pub struct PostFxStack {
    hdr_format      : wgpu::TextureFormat,
    output_format   : wgpu::TextureFormat,
    size            : Option<(NonZero<u32>, NonZero<u32>)>,
    hdr_target      : Framebuffer,
    passes          : Vec<Box<dyn PostFxPass>>
}

impl PostFxStack {
    /// Create a stack rendering through an HDR target of `hdr_format` and
    /// presenting in the surface's default format.
    pub fn new(di: &DeviceInterface, hdr_format: wgpu::TextureFormat) -> Self {
        Self {
            hdr_format,
            output_format: di.get_default_texture_format(),
            size: None,
            hdr_target: Framebuffer::new(hdr_format, "PostFx HDR target"),
            passes: Vec::new(),
        }
    }

    /// Append a pass to the end of the chain.
    pub fn push_pass(&mut self, pass: Box<dyn PostFxPass>) {
        self.passes.push(pass);
    }

    pub fn get_hdr_format(&self) -> wgpu::TextureFormat { self.hdr_format }

    /// The HDR texture the scene must be rendered into before [`PostFxStack::run`].
    pub fn get_scene_framebuffer(&self) -> Option<&wgpu::TextureView> {
        self.hdr_target.get_texture()
    }

    /// Rebuild the HDR target and notify the passes when the surface size
    /// changes.
    pub fn configure_with_surface(&mut self, di: &DeviceInterface) {
        let width = NonZero::<u32>::new(di.get_current_surface_configuration().width)
            .expect("framebuffer width should be larger than zero.");
        let height = NonZero::<u32>::new(di.get_current_surface_configuration().height)
            .expect("framebuffer height should be larger than zero.");

        let need_rebuild = self.size.map_or(true, |s| s != (width, height));
        if need_rebuild {
            log::info!("Rebuilding post-processing stack with size {}x{}", width, height);
            self.hdr_target.create_texture(di, width, height);
            for pass in &mut self.passes {
                pass.resize(di, width, height);
            }
            self.size = Some((width, height));
        }
    }

    /// Apply every pass in order, presenting the result into `final_view`.
    ///
    /// Panic
    /// ---
    /// Panics if the stack has no passes, if `configure_with_surface` has not
    /// been called, or if a pass declares a format incompatible with the chain.
    pub fn run(
        &mut self,
        di: &DeviceInterface,
        samplers: &mut SamplerManager,
        ce: &mut wgpu::CommandEncoder,
        final_view: &wgpu::TextureView
    ) {
        let count = self.passes.len();
        assert!(count > 0, "PostFxStack has no passes to run.");

        for i in 0..count {
            let is_last = i + 1 == count;
            let expected_out = if is_last { self.output_format } else { self.hdr_format };
            let pass = &self.passes[i];
            assert_eq!(
                pass.input_format(), self.hdr_format,
                "PostFx pass '{}' expects input {:?}, but the stack HDR format is {:?}.",
                pass.name(), pass.input_format(), self.hdr_format
            );
            assert_eq!(
                pass.output_format(), expected_out,
                "PostFx pass '{}' outputs {:?}, but {:?} is expected in this position.",
                pass.name(), pass.output_format(), expected_out
            );
        }

        let hdr = self.hdr_target.get_texture().expect("call configure_with_surface before run.");
        for i in 0..count {
            let is_last = i + 1 == count;
            let output = if is_last { final_view } else { hdr };
            self.passes[i].apply(di, samplers, ce, hdr, output);
        }
    }
}
