use std::sync::Arc;

use winit::window::Window;

use crate::renderer::{
    mesh::{drawable_mesh_traits::DrawableMesh, immediate_mesh::ImmediateMeshBuilder},
    orchestrator::{RenderOrchestrator, RendererConfig},
    pipeline::{
        key::{PipelineKey, VertexLayoutKey},
        rasterizer_pipeline::UseRasterizerPipeline,
    },
};

pub trait RendererState: Sized + 'static {
    fn new(window: Arc<Window>) -> impl std::future::Future<Output = Self> + Send;
    fn resize(&mut self, width: u32, height: u32);
    fn render(&mut self) -> Result<(), ()>;
}

pub struct DefaultRendererState {
    orchestrator: RenderOrchestrator,
}

impl RendererState for DefaultRendererState {
    async fn new(window: Arc<Window>) -> Self {
        Self {
            orchestrator: RenderOrchestrator::new(window, RendererConfig::default()).await,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.orchestrator.resize(width, height);
    }

    fn render(&mut self) -> Result<(), ()> {
        self.orchestrator.with_frame(|frame| {
            frame.render_scene(
                wgpu::Color {
                    r: 0.1,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                },
                |render_pass, gpu, scene| {
                    let mut mesh_builder = ImmediateMeshBuilder::new(
                        gpu.device(),
                        scene.bindless,
                        None,
                    );
                    mesh_builder.color3f([1.0, 0.0, 0.0]);
                    mesh_builder.vertex3f([0.0, 0.5, 1.0]);
                    mesh_builder.color3f([0.0, 1.0, 0.0]);
                    mesh_builder.vertex3f([-0.5, -0.5, 1.0]);
                    mesh_builder.color3f([0.0, 0.0, 1.0]);
                    mesh_builder.vertex3f([0.5, -0.5, 1.0]);
                    let mesh = mesh_builder.commit(
                        gpu.device(),
                        gpu.queue(),
                        scene.mesh_layout,
                    );

                    scene.bind_globals(gpu.device(), gpu.queue(), render_pass);
                    render_pass.set_rasterizer_pipeline(scene.pipelines.get_or_create(
                        gpu.device(),
                        PipelineKey::Default,
                        VertexLayoutKey::BASIC,
                    ));
                    mesh.draw(render_pass);
                },
            );
        });

        Ok(())
    }
}
