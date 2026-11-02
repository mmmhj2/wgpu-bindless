use std::{num::NonZeroU32, sync::Arc};

use winit::window::Window;

use crate::{
    asset::{
        asset_manager::AssetManager,
        asset_types::scene_asset::SceneAsset,
    }, renderer::{
        context::{
            frame_context::{Backbuffers, FrameContext},
            gpu_context::GpuContext,
            surface_context::SurfaceContext,
        }, drawable::draw_list::DrawList, pipeline::{
            compute_library::{ComputePipelineKey, ComputePipelineLibrary},
            library::{PipelineLibrary, SceneBindGroupLayouts},
        }, postfx::{
            build_context::PostFxBuildContext,
            postfx_stack::{BindlessFrame, PostFxExtraInputs, PostFxInput, PostFxStack},
        }, resources::{
            asset_cache::{GpuAssetStore, SceneUploadContext},
            bindless::BindlessResources,
            camera::CameraManager,
            lighting::LightingManager,
            mesh_layout::MeshLayout,
        }, scene::context::SceneContext, upload::{context::UploadContext, error::UploadError},
    },
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
            enable_postfx: false
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
    bindless: BindlessResources,
    camera: CameraManager,
    lighting: LightingManager,
    mesh_layout: MeshLayout,
    /// Per-mesh bind group layout of the GPU skinning compute pipeline.
    ///
    /// Resolved once at startup so the upload phase does not have to borrow the
    /// whole compute pipeline library.
    /// 
    /// TODO: This is still a somewhat bad design. Need to figure out some other ways.
    skinning_layout: wgpu::BindGroupLayout,
    gpu_assets: GpuAssetStore,
    pipelines: PipelineLibrary,
    compute_pipelines: ComputePipelineLibrary,
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

        let bindless = BindlessResources::new(gpu.device(), gpu.queue());
        let camera = CameraManager::new(gpu.device());
        let lighting = LightingManager::new(gpu.device());
        let mesh_layout = MeshLayout::new(gpu.device());
        let pipelines = PipelineLibrary::build_default(
            gpu.device(),
            &SceneBindGroupLayouts {
                bindless: bindless.get_bind_group_layout(),
                camera: camera.get_bind_group_layout(),
                lighting: lighting.get_bind_group_layout(),
                mesh: mesh_layout.get_bind_group_layout(),
            },
            scene_format,
            config.depth_format,
        );
        let compute_pipelines = ComputePipelineLibrary::new(gpu.device(), bindless.get_bind_group_layout());
        let skinning_layout = compute_pipelines
            .get_dispatch_bind_group_layout(ComputePipelineKey::GpuSkinning)
            .expect("the GPU skinning compute pipeline should be registered")
            .clone();
        let postfx = config
            .enable_postfx
            .then(|| PostFxStack::new(scene_format, surface_format));
        let backbuffers = Backbuffers::new(config.depth_format);

        Self {
            window,
            surface,
            backbuffers,
            bindless,
            camera,
            lighting,
            mesh_layout,
            skinning_layout,
            gpu_assets: GpuAssetStore::new(),
            pipelines,
            compute_pipelines,
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

    pub fn bindless(&self) -> &BindlessResources { &self.bindless }
    pub fn camera(&self) -> &CameraManager { &self.camera }
    pub fn camera_mut(&mut self) -> &mut CameraManager { &mut self.camera }
    pub fn lighting(&self) -> &LightingManager { &self.lighting }
    pub fn lighting_mut(&mut self) -> &mut LightingManager { &mut self.lighting }
    pub fn mesh_layout(&self) -> &MeshLayout { &self.mesh_layout }

    /// Borrow the upload entry point.
    ///
    /// Setup code (texture uploads, sampler registration) uses this instead of
    /// reaching into the bindless table itself. The borrow ends with the returned
    /// context, so the orchestrator is usable again right after.
    pub fn upload_context(&mut self) -> UploadContext<'_> {
        UploadContext::new(self.gpu.device(), self.gpu.queue(), &mut self.bindless)
    }

    /// Upload every asset a scene references and build its draw list.
    ///
    /// Uploaded meshes are stored within the returned [`DrawList`], which 
    /// holds the per-batch and per-instance GPU state (buffers and bind groups).
    /// On the other hand, textures stay owned by the store and the bindless
    /// table, so keeping the store alive may let a second scene reuse them
    /// instead of re-uploading.
    ///
    /// Errors
    /// ---
    ///
    /// Any [`UploadError`] the individual assets produce are forwarded.
    /// 
    /// Panic
    /// ---
    /// Scene hierarchy validation failure panics directly, as documented in
    /// [`crate::renderer::scene::graph::SceneGraph::from_asset`].
    pub fn prepare_scene(
        &mut self,
        assets: &AssetManager,
        scene: &SceneAsset,
    ) -> Result<DrawList, UploadError> {
        let mut ctx = SceneUploadContext::new(
            UploadContext::new(self.gpu.device(), self.gpu.queue(), &mut self.bindless),
            &self.mesh_layout,
            &self.skinning_layout,
            assets,
        );
        self.gpu_assets.prepare_scene(&mut ctx, scene)
    }

    /// Split the device from the mutable pipeline library.
    pub fn device_and_pipelines_mut(&mut self) -> (&wgpu::Device, &mut PipelineLibrary) {
        (self.gpu.device(), &mut self.pipelines)
    }

    /// Build the borrowed context required to construct and register
    /// post-processing passes.
    ///
    /// Returns `None` when PostFX was not enabled. The context bundles the
    /// device, queue, global bindless table and the [`PostFxStack`], so pass
    /// constructors can register their samplers and read the stack formats
    /// without ad-hoc split-borrow helpers.
    pub fn postfx_build_context(&mut self) -> Option<PostFxBuildContext<'_>> {
        let stack = self.postfx.as_mut()?;
        Some(PostFxBuildContext::new(
            self.gpu.device(),
            self.gpu.queue(),
            &mut self.bindless,
            stack,
        ))
    }

    pub fn pipelines(&self) -> &PipelineLibrary { &self.pipelines }
    pub fn pipelines_mut(&mut self) -> &mut PipelineLibrary { &mut self.pipelines }
    pub fn compute_pipelines(&self) -> &ComputePipelineLibrary { &self.compute_pipelines }
    pub fn compute_pipelines_mut(&mut self) -> &mut ComputePipelineLibrary { &mut self.compute_pipelines }
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
            bindless: &mut self.bindless,
            camera: &mut self.camera,
            lighting: &mut self.lighting,
            mesh_layout: &self.mesh_layout,
            pipelines: &mut self.pipelines,
            compute_pipelines: &self.compute_pipelines,
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
    bindless: &'a mut BindlessResources,
    camera: &'a mut CameraManager,
    lighting: &'a mut LightingManager,
    mesh_layout: &'a MeshLayout,
    pipelines: &'a mut PipelineLibrary,
    compute_pipelines: &'a ComputePipelineLibrary,
    postfx: Option<&'a mut PostFxStack>,
    frame: FrameContext,
    encoder: Option<wgpu::CommandEncoder>,
}

impl<'a> Frame<'a> {
    pub fn size(&self) -> (u32, u32) {
        let (width, height) = self.frame.get_size();
        (width.get(), height.get())
    }

    pub fn device(&self) -> &wgpu::Device { self.gpu.device() }
    pub fn queue(&self) -> &wgpu::Queue { self.gpu.queue() }
    pub fn pipelines(&self) -> &PipelineLibrary { self.pipelines }
    pub fn pipelines_mut(&mut self) -> &mut PipelineLibrary { self.pipelines }

    /// Compute pipeline library, valid for the whole frame.
    ///
    /// The returned reference is not borrowed from `self`, so a pipeline can be
    /// kept alive across a later [`Self::encoder`] call.
    /// 
    /// TODO: Getting around this lifetime problem is awful. 
    /// Maybe we should redesign the interface around `Frame` and pipelines.
    pub fn compute_pipelines(&self) -> &'a ComputePipelineLibrary { self.compute_pipelines }

    /// Read-only bindless resources, for passes that need the bind group.
    pub fn bindless(&self) -> &BindlessResources { self.bindless }

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
    /// The callback receives the render pass, the GPU context and a transient
    /// [`SceneContext`] containing split borrows of the scene resources.
    pub fn render_scene<F>(&mut self, clear: wgpu::Color, f: F)
    where
        F: FnOnce(&mut wgpu::RenderPass<'_>, &GpuContext, &mut SceneContext<'_>),
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
                    store: if self.postfx.is_some() { wgpu::StoreOp::Store } else { wgpu::StoreOp::Discard },
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });

        let mut scene = SceneContext {
            bindless: self.bindless,
            camera: self.camera,
            lighting: self.lighting,
            mesh_layout: self.mesh_layout,
            pipelines: self.pipelines,
        };

        f(&mut render_pass, self.gpu, &mut scene);
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
                let bindless_bind_group = self.bindless.get_bind_group(self.gpu.device());
                let bindless = BindlessFrame {
                    resources: &self.bindless,
                    bind_group: &bindless_bind_group,
                };
                postfx.run(
                    self.gpu.device(),
                    &mut encoder,
                    self.frame.get_color_view(),
                    &inputs,
                    &bindless,
                );
            }
        }

        self.gpu.queue().submit(std::iter::once(encoder.finish()));
        self.frame.present();
    }
}
