use std::sync::Arc;

use rust_renderer::{
    app::DefaultAppHandler,
    asset::{
        asset_manager::AssetManager,
        asset_store::{ImportJob, ImportManifest, ManifestEntry},
        asset_types::AssetData::TextureAssetType,
        importer::image_file_importer::ColorSpace,
    },
    renderer::{
        drawable::{drawable_mesh_traits::DrawableMesh, immediate_mesh::ImmediateMeshBuilder},
        orchestrator::{RenderOrchestrator, RendererConfig},
        pipeline::{
            key::{PipelineKey, VertexLayoutKey},
            pbr_material::PBRMaterial,
            rasterizer_pipeline::UseRasterizerPipeline,
        },
        window::RendererState,
    },
};
use winit::{event_loop::EventLoop, window::Window};

pub struct State {
    orchestrator: RenderOrchestrator,
    _asset_manager: AssetManager,
    material: PBRMaterial,
}

impl RendererState for State {
    async fn new(window: Arc<Window>) -> Self {
        let mut orchestrator = RenderOrchestrator::new(window, RendererConfig::default()).await;

        // Load the texture through an import manifest, so it is imported only
        // once and reused from disk afterwards.
        let asset_manager = AssetManager::new();
        let mut manifest: ImportManifest = Default::default();
        manifest.entries = vec![
            ManifestEntry { job: ImportJob::image("resource/skybox/back.jpg", ColorSpace::Srgb), outputs: Vec::new() },
        ];
        asset_manager.from_manifest(manifest).expect("failed to import manifest");

        let material = {
            let asset_db = asset_manager.get_database();
            let texture_asset = asset_db.iter().next().expect("at least one asset should be imported successfully");
            if let TextureAssetType(asset) = texture_asset.1.read().unwrap().get_data() {
                let mut upload = orchestrator.upload_context();
                let default_sampler = upload.bindless().get_default_sampler();
                let default_normal = upload.bindless().get_default_bump_texture();
                let default_mrao = upload.bindless().get_default_mrao_texture();

                let uploaded = upload
                    .upload_texture(texture_asset.0, asset)
                    .expect("failed to upload the texture asset");

                PBRMaterial::new(
                    uploaded.handle,
                    default_normal,
                    default_mrao,
                    uploaded.sampler,
                    default_sampler,
                    default_sampler,
                )
            } else {
                panic!("unexpected asset type");
            }
        };

        Self {
            orchestrator,
            _asset_manager: asset_manager,
            material,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.orchestrator.resize(width, height);
    }

    fn render(&mut self) -> Result<(), ()> {
        let material = self.material.clone();
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
                        Some(material),
                    );
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

fn main() {
    env_logger::init();

    let event_loop = EventLoop::<State>::with_user_event().build().expect("Failed to build event loop.");
    let mut app = DefaultAppHandler::<State>::new();
    event_loop.run_app(&mut app).unwrap();
}
