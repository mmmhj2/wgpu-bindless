use std::{cell::RefCell, collections::HashMap, num::NonZero, rc::Rc};

use crate::renderer::{
    device_interface::DeviceInterface,
    pipeline::sampler_manager::SamplerManager,
    postfx::framebuffers::Framebuffer,
};

/// A named resource produced by the  rendering passes and
/// consumed by post-processing passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostFxInput {
    /// The HDR colour the scene was rendered into. Owned and injected by the
    /// stack itself.
    SceneColor,
    /// Scene depth.
    Depth,
}

/// Extra optional inputs the renderer supplies to the post-processing stack for
/// current frame. `SceneColor` is provided by the stack itself.
#[derive(Default)]
pub struct PostFxExtraInputs<'a> {
    depth: Option<&'a wgpu::TextureView>,
}

impl<'a> PostFxExtraInputs<'a> {
    pub fn new() -> Self { Self::default() }

    /// Register one external input. Panics if `SceneColor` is passed, since it
    /// is owned by the stack.
    pub fn with(mut self, input: PostFxInput, view: &'a wgpu::TextureView) -> Self {
        match input {
            PostFxInput::SceneColor => log::warn!("SceneColor is provided by the stack, not the renderer. This request is ignored."),
            PostFxInput::Depth => self.depth = Some(view),
        }
        self
    }

    pub fn get(&self, input: PostFxInput) -> Option<&'a wgpu::TextureView> {
        match input {
            PostFxInput::SceneColor => {
                log::warn!("SceneColor is provided by the stack, not the renderer. This request is ignored.");
                None
            },
            PostFxInput::Depth => self.depth,
        }
    }
}

/// Execution context handed to a single post-processing pass.
pub struct PostFxContext<'a> {
    pub device: &'a DeviceInterface,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub samplers: &'a mut SamplerManager,
    pub size: (u32, u32),

    /// The chain's current input: the scene color for the first pass, and the
    /// previous pass's output afterwards.
    pub input: &'a wgpu::TextureView,
    /// The target this pass must write to: an internal scratch buffer, or the
    /// swapchain view for the final pass.
    pub output: &'a wgpu::TextureView,

    scene_color: &'a wgpu::TextureView,
    depth: Option<&'a wgpu::TextureView>,
}

impl PostFxContext<'_> {
    /// Fetch a required input, panicking with the offending name if absent.
    pub fn get(&self, input: PostFxInput) -> &wgpu::TextureView {
        self.try_get(input).unwrap_or_else(|| panic!("post-fx input {:?} was not provided", input))
    }

    pub fn try_get(&self, input: PostFxInput) -> Option<&wgpu::TextureView> {
        match input {
            PostFxInput::SceneColor => Some(self.scene_color),
            PostFxInput::Depth => self.depth
        }
    }
}

/// Temporary resources for the PostFx stack.
/// 
/// Used to pass around resources shared by different passes. Transient ones of
/// resources cannot persist across frames, and the references are cleaned up
/// once the PostFx stack finishes execution. Persistent ones can, but are
/// currently unimplemented.
/// 
/// *TODO:* Covers texture only resource for now. Maybe we need more compatibility
/// to cover storage buffers, etc.
#[derive(Debug)]
pub struct PostFxResourceCache {
    transient: HashMap<String, Rc<wgpu::TextureView>>,
    /// Persisent resource cache is currently stub code only.
    persistent: HashMap<String, Rc<wgpu::TextureView>>
}

impl PostFxResourceCache {
    pub fn new() -> Self { Self { transient: HashMap::new(), persistent: HashMap::new() } }

    pub fn invalidate_transient(&mut self) {
        self.transient.clear();
    }

    pub fn invalidate_persistent(&mut self) {
        self.persistent.clear();
    }

    pub fn publish(&mut self, name: &str, resource: Rc<wgpu::TextureView>) {
        self.transient.insert(name.into(), resource);
    }

    pub fn get(&self, name: &str) -> &Rc<wgpu::TextureView> {
        self.try_get(name).unwrap()
    }

    pub fn try_get(&self, name: &str) -> Option<&Rc<wgpu::TextureView>> {
        self.transient.get(name)
    }
}

/// A single post-processing pass.
///
/// Passes are executed in registration order. A pass reads [`PostFxContext::input`]
/// and writes [`PostFxContext::output`]; by default these are the same HDR target
/// (in place), unless [`PostFxPass::separate_output`] requests a distinct buffer.
pub trait PostFxPass {
    /// Human-readable name, used in diagnostics.
    fn name(&self) -> &'static str;

    /// Format this pass expects for its input texture.
    /// 
    /// As of now, input format of every pass should be identical to the color 
    /// attachment format of render pass (`hdr_format` argument in
    /// [`PostFxStack`] constructor).
    fn input_format(&self) -> wgpu::TextureFormat;

    /// Format this pass writes to its output texture.
    /// 
    /// As of now, output format of every pass (except the last pass) should be
    /// identical to the color attachment format, while the last one must be
    /// identical to the swap chain image format acquired from
    /// [`get_default_texture_format`].
    /// 
    /// [`get_default_texture_format`]: DeviceInterface::get_default_texture_format()
    fn output_format(&self) -> wgpu::TextureFormat;

    /// External inputs this pass reads, validated by the stack before execution.
    fn requires(&self) -> &'static [PostFxInput] { &[] }

    /// Whether the pass needs an output buffer distinct from its input.
    ///
    /// `false` (default) means the pass operates in place on the shared HDR
    /// target, so `input` and `output` alias.
    fn separate_output(&self) -> bool { false }

    /// Called whenever the render target size changes.
    ///
    /// Passes allocate size-dependent resources here instead of polling the
    /// surface configuration every frame.
    fn resize(&mut self, _di: &DeviceInterface, _width: NonZero<u32>, _height: NonZero<u32>) {}

    /// Record this pass into the encoder using `ctx`.
    fn apply(&mut self, ctx: &mut PostFxContext<'_>, resource: &mut PostFxResourceCache);
}

/// Owns the HDR working target, two scratch targets for out-of-place passes, and
/// an ordered chain of post-processing passes.
///
/// The scene is expected to be rendered into [`PostFxStack::scene_target`].
/// The passes are then applied in insertion order by [`PostFxStack::run`].
pub struct PostFxStack {
    hdr_format      : wgpu::TextureFormat,
    output_format   : wgpu::TextureFormat,
    cache           : PostFxResourceCache,
    size            : Option<(NonZero<u32>, NonZero<u32>)>,
    hdr_target      : Framebuffer,
    // Currently the stack only has two identical scratch targets, which are used in a ping-pong fashion for out-of-place passes.
    // We need to think about how to handle more complex cases, e.g. a pass that needs two inputs and one output, or halfres passes, etc.
    scratch         : [Framebuffer; 2],
    passes          : Vec<Box<dyn PostFxPass>>
}

impl PostFxStack {
    /// Create a stack rendering through an HDR target of `hdr_format` and
    /// presenting in the surface's default format.
    /// 
    /// A non-HDR enabled format can be specified, but some passes such as
    /// tonemapping may yield weird results.
    /// 
    /// **Note:** since the HDR target should be rendered to, it must be
    /// identical to the color render target format of the main render pass.
    pub fn new(di: &DeviceInterface, hdr_format: wgpu::TextureFormat) -> Self {
        Self {
            hdr_format,
            output_format: di.get_default_texture_format(),
            cache: PostFxResourceCache::new(),
            size: None,
            hdr_target: Framebuffer::new(hdr_format, "PostFx HDR target"),
            scratch: [
                Framebuffer::new(hdr_format, "PostFx scratch A"),
                Framebuffer::new(hdr_format, "PostFx scratch B"),
            ],
            passes: Vec::new(),
        }
    }

    /// Append a pass to the end of the chain.
    pub fn push_pass(&mut self, pass: Box<dyn PostFxPass>) {
        self.passes.push(pass);
    }

    pub fn get_hdr_format(&self) -> wgpu::TextureFormat { self.hdr_format }

    /// The HDR texture the scene must be rendered into before [`PostFxStack::run`].
    pub fn scene_target(&self) -> &Rc<wgpu::TextureView> {
        self.hdr_target.get_texture().expect("call configure_with_surface first")
    }

    /// Rebuild the HDR and scratch targets and notify the passes when the
    /// surface size changes.
    pub fn configure_with_surface(&mut self, di: &DeviceInterface) {
        let width = NonZero::<u32>::new(di.get_current_surface_configuration().width)
            .expect("framebuffer width should be larger than zero.");
        let height = NonZero::<u32>::new(di.get_current_surface_configuration().height)
            .expect("framebuffer height should be larger than zero.");

        let need_rebuild = self.size.map_or(true, |s| s != (width, height));
        if need_rebuild {
            log::info!("Rebuilding post-processing stack with size {}x{}", width, height);
            self.hdr_target.create_texture(di, width, height);
            for fb in &mut self.scratch {
                fb.create_texture(di, width, height);
            }
            for pass in &mut self.passes {
                pass.resize(di, width, height);
            }
            self.size = Some((width, height));

            self.cache.invalidate_persistent();
        }
    }

    /// Apply every pass in order, presenting the result into `final_view`.
    ///
    /// Panic
    /// ---
    /// Panics if the stack has no passes, if `configure_with_surface` has not
    /// been called, if a pass requires an input that was not provided, or if a
    /// pass declares a format incompatible with the chain.
    pub fn run(
        &mut self,
        di: &DeviceInterface,
        samplers: &mut SamplerManager,
        ce: &mut wgpu::CommandEncoder,
        final_view: &wgpu::TextureView,
        inputs: &PostFxExtraInputs<'_>
    ) {
        let count = self.passes.len();
        assert!(count > 0, "PostFxStack has no passes to run.");

        let hdr_format = self.hdr_format;
        let output_format = self.output_format;
        let (width, height) = self.size.expect("call configure_with_surface before run.");
        let size = (width.get(), height.get());

        let scene = self.hdr_target.get_texture().expect("call configure_with_surface before run.");
        let mut current: &wgpu::TextureView = scene.as_ref();
        let mut scratch_idx = 0usize;

        for i in 0..count {
            let is_last = i + 1 == count;
            let pass = &mut self.passes[i];

            assert_eq!(
                pass.input_format(), hdr_format,
                "PostFx pass '{}' expects input {:?}, but the stack HDR format is {:?}.",
                pass.name(), pass.input_format(), hdr_format
            );
            let expected_out = if is_last { output_format } else { hdr_format };
            assert_eq!(
                pass.output_format(), expected_out,
                "PostFx pass '{}' outputs {:?}, but {:?} is expected in this position.",
                pass.name(), pass.output_format(), expected_out
            );
            for required in pass.requires() {
                assert!(
                    *required == PostFxInput::SceneColor || inputs.get(*required).is_some(),
                    "PostFx pass '{}' requires input {:?}, but it was not provided.",
                    pass.name(), required
                );
            }

            let separate = pass.separate_output() && !is_last;
            let scratch_rc = self.scratch[scratch_idx].get_texture().expect("scratch target should be ready");

            let output: &wgpu::TextureView = if is_last {
                final_view
            } else if separate {
                scratch_rc.as_ref()
            } else {
                current
            };

            let mut ctx = PostFxContext {
                device: di,
                encoder: ce,
                samplers: samplers,
                size,
                input: current,
                output,
                scene_color: scene.as_ref(),
                depth: inputs.get(PostFxInput::Depth),
            };
            pass.apply(&mut ctx, &mut self.cache);

            if separate {
                current = scratch_rc.as_ref();
                scratch_idx = (scratch_idx + 1) % self.scratch.len();
            }
        }

        self.cache.invalidate_transient();
    }
}
