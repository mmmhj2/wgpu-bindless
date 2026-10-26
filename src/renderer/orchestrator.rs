use std::{num::NonZeroU32, sync::Arc};

use winit::window::Window;

use crate::renderer::{
    context::{
        frame_context::{Backbuffers, FrameContext},
        gpu_context::GpuContext,
        surface_context::SurfaceContext,
    },
    pipeline::pipeline_state::PipelineStates,
    postfx::postfx_stack::{PostFxExtraInputs, PostFxInput, PostFxStack},
};

/// Configuration used to bootstrap a [`RenderOrchestrator`].
#[derive(Debug, Clone)]
pub struct RendererConfig {
    /// Request an sRGB surface format from the adapter.
    pub request_srgb_surface: bool,
    /// Depth format used by the backbuffer and the default scene pass.
    pub depth_format: wgpu::TextureFormat,
    /// Format the scene is rendered into. Defaults to the surface format,
    /// or to `Rgba16Float` when PostFX is enabled.
    pub scene_format: Option<wgpu::TextureFormat>,
    /// Allocate an HDR target and enable the post-processing stack.
    pub enable_postfx: bool,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            request_srgb_surface: true,
            depth_format: wgpu::TextureFormat::Depth32Float,
            scene_format: None,
            enable_postfx: false,
        }
    }
}

/// Single entry point owning the GPU, surface, backbuffers, pipelines and
/// post-processing stack.
///
/// The orchestrator centralises the lifecycle of the renderer. Per-frame work
/// is exposed through [`Frame`], which borrows the mutable resources needed to
/// record one frame and guarantees a single submit/present in [`Frame::finish`].
pub struct RenderOrchestrator {
    window: Arc<Window>,
    surface: SurfaceContext<'static>,
    backbuffers: Backbuffers,
    pipelines: PipelineStates,
    postfx: Option<PostFxStack>,
    gpu: GpuContext,
    surface_format: wgpu::TextureFormat,
    scene_format: wgpu::TextureFormat,
}

impl RenderOrchestrator {
    /// Create a fully initialised renderer.
    ///
    /// Panics if the surface, adapter or device cannot be created.
    pub async fn new(window: Arc<Window>, config: RendererConfig) -> Self {
        let mut gpu = GpuContext::new();
        let mut surface = SurfaceContext::new(gpu.instance(), window.clone())
            .expect("failed to create surface");
        gpu.create_device(Some(surface.surface()))
            .await
            .expect("failed to create device");
        surface.select_color_format(
            gpu.try_adapter().expect("adapter should be ready"),
            config.request_srgb_surface,
        );
        surface.configure(gpu.device());

        let surface_format = surface
            .get_color_format()
            .expect("surface color format should be selected");
        let scene_format = config.scene_format.unwrap_or_else(|| {
            if config.enable_postfx {
                wgpu::TextureFormat::Rgba16Float
            } else {
                surface_format
            }
        });

        let pipelines = PipelineStates::prepare_default_pipelines(
            gpu.device(),
            gpu.queue(),
            scene_format,
        );
        let postfx = config
            .enable_postfx
            .then(|| PostFxStack::new(scene_format, surface_format));
        let backbuffers = Backbuffers::new(config.depth_format);

        Self {
            window,
            surface,
            backbuffers,
            pipelines,
            postfx,
            gpu,
            surface_format,
            scene_format,
        }
    }

    pub fn window(&self) -> &Window { &self.window }
    pub fn device(&self) -> &wgpu::Device { self.gpu.device() }
    pub fn queue(&self) -> &wgpu::Queue { self.gpu.queue() }
    pub fn instance(&self) -> &wgpu::Instance { self.gpu.instance() }
    pub fn gpu(&self) -> &GpuContext { &self.gpu }

    /// Split the GPU handles from the mutable pipeline state.
    ///
    /// Useful during setup, where resources are uploaded while pipelines are
    /// still being mutated.
    pub fn gpu_and_pipelines_mut(&mut self) -> (&GpuContext, &mut PipelineStates) {
        (&self.gpu, &mut self.pipelines)
    }

    /// Split the device from the mutable post-processing stack.
    pub fn device_and_postfx_mut(&mut self) -> (&wgpu::Device, Option<&mut PostFxStack>) {
        (self.gpu.device(), self.postfx.as_mut())
    }

    pub fn pipelines(&self) -> &PipelineStates { &self.pipelines }
    pub fn pipelines_mut(&mut self) -> &mut PipelineStates { &mut self.pipelines }
    pub fn postfx(&self) -> Option<&PostFxStack> { self.postfx.as_ref() }
    pub fn postfx_mut(&mut self) -> Option<&mut PostFxStack> { self.postfx.as_mut() }
    pub fn surface_format(&self) -> wgpu::TextureFormat { self.surface_format }
    pub fn scene_format(&self) -> wgpu::TextureFormat { self.scene_format }

    pub fn size(&self) -> Option<(NonZeroU32, NonZeroU32)> { self.surface.get_current_size() }

    /// Reconfigure the swapchain for the new window size.
    ///
    /// Zero-sized windows are ignored; the next non-empty frame will rebuild
    /// the size-dependent resources lazily.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (Some(width), Some(height)) = (NonZeroU32::new(width), NonZeroU32::new(height)) else {
            return;
        };
        self.surface
            .configure_with_size(self.gpu.device(), (width, height));
    }

    /// Acquire the next frame.
    ///
    /// Returns `None` when the frame is skipped (vsync, occlusion, outdated
    /// swapchain), which is not an error.
    pub fn begin_frame(&mut self) -> Option<Frame<'_>> {
        self.window.request_redraw();

        let frame = self
            .backbuffers
            .new_frame(self.gpu.device(), &mut self.surface)?;

        if let Some(postfx) = &mut self.postfx {
            postfx.configure_with_size(self.gpu.device(), frame.get_size());
        }

        let encoder = self
            .gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        Some(Frame {
            gpu: &self.gpu,
            pipelines: &mut self.pipelines,
            postfx: self.postfx.as_mut(),
            frame,
            encoder: Some(encoder),
        })
    }

    /// Run `f` for one frame, then submit and present.
    ///
    /// Skips the callback when the frame is dropped by the driver.
    pub fn with_frame<F>(&mut self, f: F)
    where
        F: FnOnce(&mut Frame<'_>),
    {
        let Some(mut frame) = self.begin_frame() else {
            return;
        };
        f(&mut frame);
        frame.finish();
    }
}

/// One in-flight frame.
///
/// The frame owns the acquired surface image and the command encoder used to
/// record it. It is consumed by [`Frame::finish`] so submit/present cannot be
/// performed twice.
#[must_use]
pub struct Frame<'a> {
    gpu: &'a GpuContext,
    pipelines: &'a mut PipelineStates,
    postfx: Option<&'a mut PostFxStack>,
    frame: FrameContext,
    encoder: Option<wgpu::CommandEncoder>,
}

impl Frame<'_> {
    pub fn size(&self) -> (u32, u32) {
        let (width, height) = self.frame.get_size();
        (width.get(), height.get())
    }

    pub fn device(&self) -> &wgpu::Device { self.gpu.device() }
    pub fn queue(&self) -> &wgpu::Queue { self.gpu.queue() }
    pub fn pipelines(&self) -> &PipelineStates { self.pipelines }
    pub fn pipelines_mut(&mut self) -> &mut PipelineStates { self.pipelines }

    pub fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        self.encoder
            .as_mut()
            .expect("frame has already been finished")
    }

    /// Color target for the scene pass.
    ///
    /// This is the PostFX HDR target when the post-processing stack is enabled
    /// and has at least one pass, otherwise it is the swapchain image.
    pub fn scene_target(&self) -> &wgpu::TextureView {
        match self.postfx.as_deref() {
            Some(postfx) if postfx.has_passes() => postfx.scene_target(),
            _ => self.frame.get_color_view(),
        }
    }

    pub fn depth_view(&self) -> &wgpu::TextureView {
        self.frame.get_depth_view()
    }

    /// Record the main scene render pass.
    ///
    /// The callback receives the render pass, the GPU context and the mutable
    /// pipeline state. This split-borrow API lets the callback prepare bind
    /// groups and draw without exposing the orchestrator again.
    pub fn render_scene<F>(&mut self, clear: wgpu::Color, f: F)
    where
        F: FnOnce(&mut wgpu::RenderPass<'_>, &GpuContext, &mut PipelineStates),
    {
        let encoder = self
            .encoder
            .as_mut()
            .expect("frame has already been finished");
        let color_view = match self.postfx.as_deref() {
            Some(postfx) if postfx.has_passes() => postfx.scene_target(),
            _ => self.frame.get_color_view(),
        };
        let depth_view = self.frame.get_depth_view();

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Scene Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });

        f(&mut render_pass, self.gpu, self.pipelines);
    }

    /// Run post-processing, submit the recorded commands and present.
    pub fn finish(mut self) {
        let mut encoder = self
            .encoder
            .take()
            .expect("frame has already been finished");

        if let Some(postfx) = self.postfx.as_deref_mut() {
            if postfx.has_passes() {
                let inputs = PostFxExtraInputs::new()
                    .with(PostFxInput::Depth, self.frame.get_depth_view());
                let samplers = self
                    .pipelines
                    .get_bindless_resource_manager_mut()
                    .get_samplers_mut();
                postfx.run(
                    self.gpu.device(),
                    samplers,
                    &mut encoder,
                    self.frame.get_color_view(),
                    &inputs,
                );
            }
        }

        self.gpu.queue().submit(std::iter::once(encoder.finish()));
        self.frame.present();
    }
}
