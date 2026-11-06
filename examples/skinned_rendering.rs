//! Skeletal animation demo: `resource/Fox.glb` played back through the GPU
//! skinning compute pass.
//!
//! The scene is imported once (assets are cached on disk), the draw list binds
//! the baked animation texture to the skinned mesh, and every frame records a
//! compute pass that deforms the rest-pose vertex buffers before the scene pass
//! draws them with the regular PBR pipeline.

use std::{path::Path, sync::Arc, time::Instant};

use rust_renderer::{
    app::DefaultAppHandler,
    asset::{
        asset_manager::AssetManager,
        asset_store::{ImportJob, ImportManifest, ManifestEntry},
        asset_types::AssetData::SceneAssetType,
        importer::gltf_importer::GltfImporter,
    },
    renderer::{
        drawable::draw_list::{DrawList, DrawableType},
        orchestrator::{RenderOrchestrator, RendererConfig},
        pipeline::{
            key::{PipelineKey, VertexLayoutKey},
            rasterizer_pipeline::UseRasterizerPipeline,
        },
        resources::{
            camera::{CameraPerspective, CameraPerspectiveBuilder},
            lighting::Light,
        },
        window::RendererState,
    },
};
use winit::{event_loop::EventLoop, window::Window};

const GLTF_PATH: &str = "resource/Fox.glb";
const MANIFEST_PATH: &str = "resource/import_fox_manifest.ff";

/// How long the demo plays one clip before switching to the next one.
const CLIP_SWITCH_SECONDS: f32 = 3.0;

struct State {
    orchestrator: RenderOrchestrator,
    draw_list: DrawList,
    camera: CameraPerspective,
    light: Light,
    /// Start of the current frame's animation delta.
    last_tick: Instant,
    /// Time spent on the currently selected clip.
    clip_timer: f32,
    _asset_manager: AssetManager,
}

/// Select the next clip of the first player, demonstrating runtime switching.
fn play_next_clip(draw_list: &mut DrawList) {
    let animations = draw_list.animations_mut();
    if animations.player_count() == 0 {
        return;
    }

    let rig = animations.player(0).rig().clone();
    let clips = animations.clips_for_rig(&rig);
    if clips.len() < 2 {
        return;
    }

    let current = animations.player(0).clip();
    let next_index = current
        .and_then(|current| clips.iter().position(|&clip| clip == current))
        .map_or(0, |index| (index + 1) % clips.len());
    let next = clips[next_index];

    animations.play(0, next);
    log::info!("Playing animation clip '{}'.", animations.clip(next).name());
}

impl RendererState for State {
    async fn new(window: Arc<Window>) -> Self {
        let mut orchestrator = RenderOrchestrator::new(window, RendererConfig::default()).await;

        // Import through the manifest: the first run bakes the skinned mesh, the
        // skin and the animations; later runs load the cached containers.
        let asset_manager = AssetManager::new();
        let manifest_path = Path::new(MANIFEST_PATH);
        let mut manifest = ImportManifest::load(manifest_path).expect("failed to load import manifest");
        if manifest.entries.is_empty() {
            manifest.entries = vec![ManifestEntry {
                job: ImportJob::gltf(GLTF_PATH),
                outputs: Vec::new(),
            }];
            manifest.save(manifest_path).expect("failed to save import manifest");
        }
        asset_manager.import_manifest(manifest_path).expect("failed to import manifest");

        let draw_list = {
            let scene_name = GltfImporter::scene_subasset_name(Path::new(GLTF_PATH), 0)
                .expect("failed to resolve scene subasset name");
            let asset = asset_manager.get_asset(&scene_name).expect("gltf scene asset should be present");
            let asset = asset.read().expect("failed to read gltf scene asset");
            let SceneAssetType(scene) = asset.get_data() else {
                panic!("asset '{scene_name}' is not a scene asset");
            };
            orchestrator
                .prepare_scene(&asset_manager, scene)
                .expect("failed to prepare the scene")
        };

        log::info!(
            "Loaded {} skinned meshes, {} static batches and {} animation clips.",
            draw_list.count(DrawableType::OpaqueSkinned),
            draw_list.count(DrawableType::OpaqueStatic),
            draw_list.animations().clips().len()
        );

        orchestrator.lighting_mut().set_ambient([0.05, 0.05, 0.05]);
        let light = Light::directional(cgmath::vec3(1.0, -1.0, 1.0), [1.0, 1.0, 1.0], 3.0);

        // The Fox model is ~150 units long, so the camera has to sit far away;
        // its bounding box is roughly y in [0, 79], z in [-88, 67].
        let camera = CameraPerspectiveBuilder::new()
            .set_origin(cgmath::point3(0.0, 40.0, 200.0))
            .set_target(cgmath::point3(0.0, 40.0, -10.0))
            .build();

        Self {
            orchestrator,
            draw_list,
            camera,
            light,
            last_tick: Instant::now(),
            clip_timer: 0.0,
            _asset_manager: asset_manager,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.orchestrator.resize(width, height);
    }

    fn render(&mut self) -> Result<(), ()> {
        let camera = self.camera.clone();
        let light = self.light.clone();

        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f32();
        self.last_tick = now;

        self.clip_timer += dt;
        if self.clip_timer >= CLIP_SWITCH_SECONDS {
            self.clip_timer = 0.0;
            play_next_clip(&mut self.draw_list);
        }
        self.draw_list.update_animations(dt);

        self.draw_list
            .ensure_bound(self.orchestrator.device(), self.orchestrator.mesh_layout())
            .ensure_deformation_bound(self.orchestrator.device(), self.orchestrator.compute_pipelines());

        let draw_list = &self.draw_list;

        self.orchestrator.with_frame(|frame| {
            {
                let pipelines = frame.compute_pipelines();
                let bindless_bg = frame.bindless().get_bind_group(frame.device());
                let mut cp = frame.encoder().begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Deformation Pass"),
                    timestamp_writes: None,
                });
                cp.set_bind_group(0, Some(&bindless_bg), &[]);
                draw_list.record_deformation(&mut cp, pipelines);
            }

            frame.render_scene(
                wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0 },
                |render_pass, gpu, scene| {
                    scene.camera.set_active_camera(camera);
                    scene.lighting.upload_lights(gpu.queue(), std::slice::from_ref(&light));
                    scene.bind_globals(gpu.device(), gpu.queue(), render_pass);
                    render_pass.set_rasterizer_pipeline(scene.pipelines.get_or_create(
                        gpu.device(),
                        PipelineKey::CookTorrancePbr,
                        VertexLayoutKey::BASIC,
                    ));

                    draw_list.draw(DrawableType::OpaqueStatic, render_pass);
                    draw_list.draw(DrawableType::OpaqueSkinned, render_pass);
                },
            );
        });

        Ok(())
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::<State>::with_user_event()
        .build()
        .expect("Failed to build event loop.");
    let mut app = DefaultAppHandler::new();
    event_loop.run_app(&mut app).unwrap();
}
