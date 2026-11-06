//! End-to-end GPU smoke test for the skinned mesh path.
//!
//! It imports the skinned test model into a temp workspace, builds the draw
//! list, records the skinning compute pass and renders a few frames. Any wgpu
//! validation error along the way aborts the test.

#[cfg(any(target_os = "windows", target_os = "linux"))]
mod common;

#[cfg(test)]
#[cfg(any(target_os = "windows", target_os = "linux"))]
mod test {
    use std::{path::PathBuf, sync::Arc};
    use rust_renderer::{
        asset::{
            asset_manager::AssetManager,
            asset_types::AssetData,
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
    use winit::window::Window;
    use crate::common;

    const FRAMES_TO_RENDER: u32 = 3;

    /// Playback time fed to the skinning pass on each frame.
    const PLAYBACK_SECONDS_PER_FRAME: f32 = 0.1;

    /// Copy the skinned test model into an isolated temp workspace.
    ///
    /// Importers persist assets next to their source and `resource/` is not
    /// writable from tests, so the model is imported from a temp copy.
    fn temp_workspace() -> PathBuf {
        let dir = std::env::temp_dir().join("rust_renderer_skinned_rendering");
        if dir.exists() {
            std::fs::remove_dir_all(&dir).ok();
        }
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy("resource/Fox.glb", dir.join("Fox.glb")).unwrap();
        dir
    }

    pub struct State {
        orchestrator: RenderOrchestrator,
        draw_list: DrawList,
        camera: CameraPerspective,
        light: Light,
        rendered_frames: u32,
        _asset_manager: AssetManager,
    }

    impl RendererState for State {
        async fn new(window: Arc<Window>) -> Self {
            let mut orchestrator = RenderOrchestrator::new(window, RendererConfig::default()).await;

            let dir = temp_workspace();
            let glb = dir.join("Fox.glb");
            let asset_manager = AssetManager::new();
            asset_manager.import(GltfImporter::new(&glb)).expect("import should succeed");

            let mut draw_list = {
                let scene_name = GltfImporter::scene_subasset_name(&glb, 0).expect("scene name should resolve");
                let asset = asset_manager.get_asset(&scene_name).expect("gltf scene asset should be present");
                let asset = asset.read().expect("failed to read gltf scene asset");
                let AssetData::SceneAssetType(scene) = asset.get_data() else {
                    panic!("asset '{scene_name}' is not a scene asset");
                };
                orchestrator
                    .prepare_scene(&asset_manager, scene)
                    .expect("failed to prepare the scene")
            };

            assert_eq!(
                draw_list.count(DrawableType::OpaqueSkinned),
                1,
                "Fox has exactly one skinned primitive"
            );
            assert_eq!(
                draw_list.count(DrawableType::OpaqueStatic),
                0,
                "the Fox mesh must not produce a static batch"
            );
            assert!(
                draw_list.get_skinned_meshes()[0].is_animated(),
                "the imported Fox animation should be bound to the skinned mesh"
            );

            // All baked clips of the skin must be registered, and the default
            // clip must be the first one in deterministic order.
            {
                let player = draw_list.get_skinned_meshes()[0].player();
                let animations = draw_list.animations_mut();
                assert_eq!(animations.clips().len(), 3, "Fox has three baked animations");
                assert_eq!(animations.player_count(), 1, "one player per skinned instance");

                let rig = animations.player(player).rig().clone();
                let clips = animations.clips_for_rig(&rig).to_vec();
                assert_eq!(clips.len(), 3);
                assert_eq!(
                    animations.player(player).clip(),
                    Some(clips[0]),
                    "the default clip should be the first sorted clip"
                );

                // Runtime switching must reset the playhead.
                animations.play(player, clips[1]);
                assert_eq!(animations.player(player).clip(), Some(clips[1]));
                assert_eq!(animations.player(player).time(), 0.0);
            }

            orchestrator.lighting_mut().set_ambient([0.05, 0.05, 0.05]);
            let light = Light::directional(cgmath::vec3(1.0, -1.0, 1.0), [1.0, 1.0, 1.0], 3.0);

            // The Fox model is ~150 units long, so the camera has to sit far
            // away; its bounding box is roughly y in [0, 79], z in [-88, 67].
            let camera = CameraPerspectiveBuilder::new()
                .set_origin(cgmath::point3(0.0, 40.0, 200.0))
                .set_target(cgmath::point3(0.0, 40.0, -10.0))
                .build();

            Self {
                orchestrator,
                draw_list,
                camera,
                light,
                rendered_frames: 0,
                _asset_manager: asset_manager,
            }
        }

        fn resize(&mut self, width: u32, height: u32) {
            self.orchestrator.resize(width, height);
        }

        fn render(&mut self) -> Result<(), ()> {
            self.rendered_frames += 1;

            let camera = self.camera.clone();
            let light = self.light.clone();

            self.draw_list.animations_mut().update(PLAYBACK_SECONDS_PER_FRAME);
            if self.rendered_frames == 1 {
                let player = self.draw_list.get_skinned_meshes()[0].player();
                let frame = self.draw_list.animations().frame(player)
                    .expect("the Fox player should have a clip")
                    .joints
                    .expect("the Fox clip drives joints")
                    .frame();
                assert!(frame > 0.0, "the manager should advance the playback frame");
            }

            self.draw_list
                .ensure_bound(self.orchestrator.device(), self.orchestrator.mesh_layout())
                .ensure_deformation_bound(self.orchestrator.device(), self.orchestrator.compute_pipelines());

            let draw_list = &self.draw_list;

            self.orchestrator.with_frame(|frame| {
                // The deformation compute pass must be recorded before the
                // scene pass that reads the deformed vertex buffers.
                {
                    let pipelines = frame.compute_pipelines();
                    let bindless_bg = frame.bindless().get_bind_group(frame.device());
                    let mut cp = frame.encoder().begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Deformation Pass"),
                        timestamp_writes: None,
                    });
                    cp.set_bind_group(0, Some(&bindless_bg), &[]);
                    // Each frame advances the manager, so the dispatched frame
                    // changes and the mesh is re-posed instead of staying in the
                    // bind pose.
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

    #[test]
    fn skinned_rendering_test() {
        env_logger::init();

        let frames = common::run_windowed::<State>(FRAMES_TO_RENDER);
        assert_eq!(
            frames,
            Some(FRAMES_TO_RENDER),
            "the event loop should stop after the configured frame limit"
        );
    }
}
