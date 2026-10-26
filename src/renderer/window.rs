use crate::renderer::{
    context::{frame_context::Backbuffers, gpu_context::GpuContext, surface_context::SurfaceContext},
    mesh::{drawable_mesh_traits::DrawableMesh, immediate_mesh::ImmediateMeshBuilder},
    pipeline::{pipeline_state::PipelineStates, rasterizer_pipeline::UseRasterizerPipeline},
};

use std::{num::NonZeroU32, sync::Arc};

use winit::window::Window;

pub trait RendererState<'a> {
    fn gpu_context(&self) -> &GpuContext;
    fn surface_context(&self) -> &SurfaceContext<'a>;
    fn surface_context_mut(&mut self) -> &mut SurfaceContext<'a>;
    fn backbuffers_mut(&mut self) -> &mut Backbuffers;

    fn resize(&mut self, width: u32, height: u32) {
        let (Some(width), Some(height)) = (NonZeroU32::new(width), NonZeroU32::new(height)) else {
            return;
        };

        let device = self.gpu_context().device().clone();
        self.surface_context_mut().configure_with_size(&device, (width, height));
    }

    fn new(window: Arc<Window>) -> impl std::future::Future<Output = Self> + Send;
    fn render(&mut self) -> Result<(), ()>;
}

pub struct DefaultRendererState {
    window: Arc<Window>,
    gpu: GpuContext,
    surface: SurfaceContext<'static>,
    backbuffers: Backbuffers,
    pipelines: PipelineStates,
}

impl RendererState<'static> for DefaultRendererState {
    async fn new(window: Arc<Window>) -> Self {
        let mut gpu = GpuContext::new();
        let mut surface = SurfaceContext::new(gpu.instance(), window.clone()).unwrap();
        gpu.create_device(Some(surface.surface())).await.unwrap();
        surface.select_color_format(gpu.try_adapter().expect("adapter should be ready"), true);
        surface.configure(gpu.device());

        let format = surface.get_color_format().expect("surface color format should be selected");
        let pipelines = PipelineStates::prepare_default_pipelines(gpu.device(), gpu.queue(), format);
        let backbuffers = Backbuffers::new(wgpu::TextureFormat::Depth32Float);

        Self {
            window,
            gpu,
            surface,
            backbuffers,
            pipelines,
        }
    }

    fn render(&mut self) -> Result<(), ()> {
        self.window.request_redraw();

        let Some(frame) = self.backbuffers.new_frame(self.gpu.device(), &mut self.surface) else {
            return Ok(());
        };

        let mut encoder = self.gpu.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // Do some draw call here
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame.get_color_view(),
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: frame.get_depth_view(),
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

            let mut mesh_builder = ImmediateMeshBuilder::new(
                self.gpu.device(),
                self.pipelines.get_bindless_resource_manager_mut(),
                None,
            );
            mesh_builder.color3f([1.0, 0.0, 0.0]);
            mesh_builder.vertex3f([0.0, 0.5, 1.0]);
            mesh_builder.color3f([0.0, 1.0, 0.0]);
            mesh_builder.vertex3f([-0.5, -0.5, 1.0]);
            mesh_builder.color3f([0.0, 0.0, 1.0]);
            mesh_builder.vertex3f([0.5, -0.5, 1.0]);
            let mesh = mesh_builder.commit(
                self.gpu.device(),
                self.gpu.queue(),
                self.pipelines.get_mesh_manager(),
            );

            self.pipelines
                .prepare_render_pass(self.gpu.device(), self.gpu.queue(), &mut render_pass);
            render_pass.set_rasterizer_pipeline(self.pipelines.get_default_pipeline());
            mesh.draw(&mut render_pass);
        }

        // submit will accept anything that implements IntoIter
        self.gpu.queue().submit(std::iter::once(encoder.finish()));
        frame.present();

        Ok(())
    }

    fn gpu_context(&self) -> &GpuContext {
        &self.gpu
    }

    fn surface_context(&self) -> &SurfaceContext<'static> {
        &self.surface
    }

    fn surface_context_mut(&mut self) -> &mut SurfaceContext<'static> {
        &mut self.surface
    }

    fn backbuffers_mut(&mut self) -> &mut Backbuffers {
        &mut self.backbuffers
    }
}
