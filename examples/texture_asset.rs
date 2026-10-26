use std::sync::Arc;

use rust_renderer::{
    app::DefaultAppHandler,
    asset::{
        asset_manager::AssetManager,
        asset_store::{ImportJob, ImportManifest, ManifestEntry},
        asset_types::{AssetData::TextureAssetType, GpuAssetType},
        importer::image_file_importer::ColorSpace,
    },
    renderer::{
        context::{
            frame_context::Backbuffers, gpu_context::GpuContext, surface_context::SurfaceContext,
        },
        mesh::{drawable_mesh_traits::DrawableMesh, immediate_mesh::ImmediateMeshBuilder},
        pipeline::{
            pbr_material::PBRMaterial, pipeline_state::PipelineStates,
            rasterizer_pipeline::UseRasterizerPipeline,
        },
        window::RendererState,
    },
};
use winit::{event_loop::EventLoop, window::Window};

pub struct State {
    window: Arc<Window>,
    gpu: GpuContext,
    surface: SurfaceContext<'static>,
    backbuffers: Backbuffers,
    pipelines: PipelineStates,
    asset_manager: AssetManager,
    material: PBRMaterial,
}

impl RendererState<'static> for State {
    async fn new(window: Arc<Window>) -> Self {
        let mut gpu = GpuContext::new();
        let mut surface = SurfaceContext::new(gpu.instance(), window.clone()).unwrap();
        gpu.create_device(Some(surface.surface())).await.unwrap();
        surface.select_color_format(gpu.try_adapter().expect("adapter should be ready"), true);
        surface.configure(gpu.device());

        let format = surface.get_color_format().expect("surface color format should be selected");
        let mut pipelines = PipelineStates::prepare_default_pipelines(gpu.device(), gpu.queue(), format);

        // Load the texture through an import manifest, so it is imported only
        // once and reused from disk afterwards.
        let asset_manager = AssetManager::new();
        let mut manifest: ImportManifest = Default::default();
        manifest.entries = vec![
            ManifestEntry { job: ImportJob::image("resource/skybox/back.jpg", ColorSpace::Srgb), outputs: Vec::new() },
        ];
        asset_manager.from_manifest(manifest).expect("failed to import manifest");

        let texture_view;
        let sampler;
        {
            let asset_db = asset_manager.get_database();
            let texture_asset = asset_db.iter().next().expect("at least one asset should be imported successfully");
            if let TextureAssetType(asset) = texture_asset.1.read().unwrap().get_data() {
                let texture = asset.upload(texture_asset.0, gpu.device(), gpu.queue());
                texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                sampler = asset.sampler.clone().into();
            } else {
                panic!("unexpected asset type");
            }
        }

        let material = PBRMaterial::create_from_views(
            gpu.device(),
            pipelines.get_bindless_resource_manager_mut(),
            Some((texture_view, Some(sampler))),
            None,
            None
        ).expect("failed to allocate material.");

        Self {
            window,
            gpu,
            surface,
            backbuffers: Backbuffers::new(wgpu::TextureFormat::Depth32Float),
            pipelines,
            asset_manager,
            material,
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
                    depth_ops: Some(wgpu::Operations{
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard
                    }),
                    stencil_ops: None
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            let mut mesh_builder = ImmediateMeshBuilder::new(
                self.gpu.device(),
                self.pipelines.get_bindless_resource_manager_mut(),
                Some(self.material.clone()));
            mesh_builder.color3f([1.0, 1.0, 1.0]);

            mesh_builder.texcoord2f([0.0, 0.0]);
            mesh_builder.vertex3f([-0.5, 0.5, 1.0]);
            mesh_builder.texcoord2f([1.0, 0.0]);
            mesh_builder.vertex3f([-0.5,-0.5, 1.0]);
            mesh_builder.texcoord2f([1.0, 1.0]);
            mesh_builder.vertex3f([ 0.5,-0.5, 1.0]);

            mesh_builder.texcoord2f([0.0, 0.0]);
            mesh_builder.vertex3f([-0.5, 0.5, 1.0]);
            mesh_builder.texcoord2f([1.0, 1.0]);
            mesh_builder.vertex3f([ 0.5,-0.5, 1.0]);
            mesh_builder.texcoord2f([0.0, 1.0]);
            mesh_builder.vertex3f([ 0.5, 0.5, 1.0]);
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

fn main() {
    env_logger::init();

    let event_loop = EventLoop::<State>::with_user_event().build().expect("Failed to build event loop.");
    let mut app = DefaultAppHandler::<State>::new();
    event_loop.run_app(&mut app).unwrap();
}
