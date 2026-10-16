use std::{collections::HashMap, num::NonZero, rc::Rc};

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

/// How a post-processing pass writes its result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// The pass writes to a buffer distinct from its input (the default).
    Separate,
    /// The pass reads and writes the same working target, so `input` and
    /// `output` alias.
    ///
    /// # Safety
    ///
    /// The implementation must guarantee that no single render pass both
    /// samples and renders the same subresource. Otherwise this is a undefined
    /// behaviour. Construct it with [`OutputMode::in_place`].
    InPlace,
}

impl OutputMode {
    /// Request an out-of-place output buffer (the safe default).
    pub const fn separate() -> Self { OutputMode::Separate }

    /// Request in-place output.
    ///
    /// # Safety
    ///
    /// See [`OutputMode::InPlace`].
    pub unsafe fn in_place() -> Self { OutputMode::InPlace }
}

/// A single post-processing pass.
///
/// Passes are executed in registration order. A pass reads [`PostFxContext::input`]
/// and writes [`PostFxContext::output`].
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

    /// How this pass writes its result.
    ///
    /// Defaults to [`OutputMode::separate`].
    /// [`OutputMode::in_place`] is unsafe and must be opted into explicitly.
    fn output_mode(&self) -> OutputMode { OutputMode::separate() }

    /// Called whenever the render target size changes.
    ///
    /// Passes allocate size-dependent resources here instead of polling the
    /// surface configuration every frame.
    fn resize(&mut self, _di: &DeviceInterface, _width: NonZero<u32>, _height: NonZero<u32>) {}

    /// Record this pass into the encoder using `ctx`.
    fn apply(&mut self, ctx: &mut PostFxContext<'_>, resource: &mut PostFxResourceCache);
}

/// Owns the HDR working target, a lazily grown pool of scratch targets for
/// out-of-place passes, and an ordered chain of post-processing passes.
///
/// At most one scratch buffer is live at any time (the chain's current input),
/// so the pool never grows beyond two entries. The scene is expected to be
/// rendered into [`PostFxStack::scene_target`]; the passes are then applied in
/// insertion order by [`PostFxStack::run`].
pub struct PostFxStack {
    hdr_format      : wgpu::TextureFormat,
    output_format   : wgpu::TextureFormat,
    cache           : PostFxResourceCache,
    size            : Option<(NonZero<u32>, NonZero<u32>)>,
    hdr_target      : Framebuffer,
    scratch         : Vec<Framebuffer>,
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
            scratch: Vec::new(),
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

    /// Return a scratch buffer index that is not `exclude`, growing the pool if
    /// every existing buffer is excluded.
    ///
    /// Because only the chain's current input is ever live, this returns the
    /// first free buffer and the pool stabilises at two entries.
    /// 
    /// TODO: Support more varieties of scratch textures (halfres, etc.)
    fn acquire_scratch(&mut self, exclude: Option<usize>) -> usize {
        for i in 0..self.scratch.len() {
            if Some(i) != exclude {
                return i;
            }
        }
        self.scratch.push(Framebuffer::new(self.hdr_format, "PostFx scratch"));
        self.scratch.len() - 1
    }

    /// Apply every pass in order, presenting the result into `final_view`.
    ///
    /// Panic
    /// ---
    /// Panics if the stack has no passes, if `configure_with_surface` has not
    /// been called, if a pass requires an input that was not provided, if a
    /// pass declares a format incompatible with the chain, or if the final pass
    /// requests in-place output.
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

        // Validate the whole chain before recording anything.
        #[cfg(debug_assertions)]
        for i in 0..count {
            let is_last = i + 1 == count;
            let pass = &self.passes[i];

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
        }

        let scene = self.hdr_target.get_texture().expect("call configure_with_surface before run.").clone();

        // Index of the buffer holding the chain's current input; `None` = scene.
        let mut current: Option<usize> = None;

        for i in 0..count {
            let is_last = i + 1 == count;
            let mode = self.passes[i].output_mode();
            assert!(
                !(is_last && mode == OutputMode::InPlace),
                "The final post-fx pass '{}' writes to the swapchain and cannot be in place.",
                self.passes[i].name()
            );

            // Decide the output buffer first (this may grow the pool), so that
            // no view borrow is held across the mutable borrow of the pool.
            let out: Option<usize> = if is_last {
                None
            } else {
                match mode {
                    OutputMode::Separate => Some(self.acquire_scratch(current)),
                    OutputMode::InPlace => current,
                }
            };

            let input: &wgpu::TextureView = match current {
                None => scene.as_ref(),
                Some(k) => self.scratch[k].view().as_ref(),
            };
            let output: &wgpu::TextureView = if is_last {
                final_view
            } else {
                match out {
                    None => scene.as_ref(),
                    Some(k) => self.scratch[k].view().as_ref(),
                }
            };

            let mut ctx = PostFxContext {
                device: di,
                encoder: &mut *ce,
                samplers: &mut *samplers,
                size,
                input,
                output,
                scene_color: scene.as_ref(),
                depth: inputs.get(PostFxInput::Depth),
            };
            self.passes[i].apply(&mut ctx, &mut self.cache);

            if !is_last {
                current = out;
            }
        }

        self.cache.invalidate_transient();
    }
}
