//! End-to-end GPU smoke test for the skinned mesh path.
//!
//! It imports the skinned test model into a temp workspace, builds the draw
//! list, records the skinning compute pass and renders a few frames. Any wgpu
//! validation error along the way aborts the test.

#[cfg(target_os = "windows")]
mod common;

#[cfg(test)]
#[cfg(target_os = "windows")]
mod test {
    use std::{path::PathBuf, sync::Arc};

    use rust_renderer::{
        app::DefaultAppHandler,
        asset::{
            asset_manager::AssetManager,
            asset_types::{AssetData, GpuAssetType},
            importer::gltf_importer::GltfImporter,
        },
        renderer::{
            drawable::draw_list::{DrawList, DrawableType},
            orchestrator::{RenderOrchestrator, RendererConfig},
            pipeline::{
                compute_library::ComputePipelineKey,
                key::{PipelineKey, VertexLayoutKey},
                rasterizer_pipeline::UseRasterizerPipeline,
            },
            resources::{
                camera::{CameraPerspective, CameraPerspectiveBuilder},
                lighting::Light,
            },
            scene::graph::SceneGraph,
            window::RendererState,
        },
    };
    use winit::{event_loop::EventLoop, window::Window};

    /// Render a few frames so the compute pass and the scene pass both run more
    /// than once (the second frame reuses the already-written deformed buffers).
    const FRAMES_TO_RENDER: u32 = 3;

    /// `resource/Fox.glb` has a single skin with 24 joints.
    const JOINT_COUNT: usize = 24;

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
        /// Bindless index of the bind-pose animation texture.
        animation_texture: u32,
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

            let draw_list = {
                let scene_name = GltfImporter::scene_subasset_name(&glb, 0).expect("scene name should resolve");
                let asset = asset_manager.get_asset(&scene_name).expect("gltf scene asset should be present");
                let asset = asset.read().expect("failed to read gltf scene asset");
                let AssetData::SceneAssetType(scene) = asset.get_data() else {
                    panic!("asset '{scene_name}' is not a scene asset");
                };
                let mut ctx = orchestrator.mesh_upload_context(&asset_manager);
                SceneGraph::from_asset(scene).into_draw_list(&mut ctx)
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

            orchestrator.lighting_mut().set_ambient([0.05, 0.05, 0.05]);
            let light = Light::directional(cgmath::vec3(1.0, -1.0, 1.0), [1.0, 1.0, 1.0], 3.0);

            // Phase 3 has no animation baking yet, so drive the shader with a
            // one-frame texture holding identity matrices: the mesh must come
            // out in its bind pose.
            let animation_texture = {
                let texture = crate::common::rest_pose_texture(JOINT_COUNT)
                    .upload("rest pose animation", orchestrator.device(), orchestrator.queue());
                let view = texture.create_view(&Default::default());
                orchestrator
                    .bindless_mut()
                    .create_sampled_texture(view)
                    .expect("rest pose texture should fit in the bindless table")
                    .index() as u32
            };

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
                animation_texture,
                rendered_frames: 0,
                _asset_manager: asset_manager,
            }
        }

        fn resize(&mut self, width: u32, height: u32) {
            self.orchestrator.resize(width, height);
        }

        fn render(&mut self) -> Result<(), ()> {
            if self.rendered_frames >= FRAMES_TO_RENDER {
                return Err(());
            }
            self.rendered_frames += 1;

            let camera = self.camera.clone();
            let light = self.light.clone();
            let animation_texture = self.animation_texture;
            let draw_list = &self.draw_list;

            self.orchestrator.with_frame(|frame| {
                // The skinning compute pass must be recorded before the scene
                // pass that reads the deformed vertex buffers.
                {
                    let pipeline = frame.compute_pipelines()
                        .get_pipeline(ComputePipelineKey::GpuSkinning)
                        .expect("the GPU skinning pipeline should be registered");
                    let bindless_bg = frame.bindless().get_bind_group(frame.device());
                    let mut cp = frame.encoder().begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Skinning Pass"),
                        timestamp_writes: None,
                    });
                    cp.set_pipeline(pipeline);
                    cp.set_bind_group(0, Some(&bindless_bg), &[]);
                    for mesh in draw_list.get_skinned_meshes() {
                        // Bind pose: frame 0 of the identity animation texture.
                        mesh.skin_with(&mut cp, animation_texture, 0);
                    }
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

        // The event loop must be created from the test thread.
        use winit::platform::windows::EventLoopBuilderExtWindows;

        let event_loop = EventLoop::<State>::with_user_event()
            .with_any_thread(true)
            .build()
            .expect("Failed to build event loop.");
        let mut app = DefaultAppHandler::new();

        event_loop.run_app(&mut app).unwrap();
    }
}
