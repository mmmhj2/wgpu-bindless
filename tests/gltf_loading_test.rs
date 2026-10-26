#[cfg(test)]
#[cfg(target_os = "windows")]
mod test {
    use std::sync::Arc;

    use rust_renderer::{
        app::DefaultAppHandler,
        asset::{
            asset_manager::AssetManager, asset_types::AssetData,
            importer::gltf_importer::GltfImporter,
        },
        renderer::{
            mesh::static_mesh_instance::StaticMeshInstance,
            orchestrator::{RenderOrchestrator, RendererConfig},
            window::RendererState,
        },
    };
    use winit::{event_loop::EventLoop, window::Window};

    pub struct State {
        orchestrator: RenderOrchestrator,
        _gltf_models: Vec<StaticMeshInstance>,
        _asset_manager: AssetManager,
    }

    impl RendererState for State {
        async fn new(window: Arc<Window>) -> Self {
            let mut orchestrator = RenderOrchestrator::new(window, RendererConfig::default()).await;

            let asset_manager = AssetManager::new();
            asset_manager.import(GltfImporter::new("resource/test_two_cubes.glb")).expect("import should succeed");

            // The scene graph is baked into a `SceneAsset` at import time, so
            // the glTF file is parsed exactly once.
            let gltf_models = {
                let scene_name = GltfImporter::scene_subasset_name(std::path::Path::new("resource/test_two_cubes.glb"), 0).expect("scene name should resolve");
                let asset = asset_manager.get_asset(&scene_name).expect("gltf scene asset should be present");
                let asset = asset.read().expect("failed to read gltf scene asset");
                if let AssetData::SceneAssetType(scene) = asset.get_data() {
                    let (gpu, pipelines) = orchestrator.gpu_and_pipelines_mut();
                    StaticMeshInstance::create_from_scene_asset(gpu.device(), gpu.queue(), pipelines, scene, &asset_manager)
                } else {
                    panic!("asset '{scene_name}' is not a scene asset");
                }
            };

            println!("Read {} models in total.", gltf_models.len());

            Self {
                orchestrator,
                _gltf_models: gltf_models,
                _asset_manager: asset_manager,
            }
        }

        fn resize(&mut self, width: u32, height: u32) {
            self.orchestrator.resize(width, height);
        }

        fn render(&mut self) -> Result<(), ()> {
            Err(())
        }
    }

    #[test]
    fn gltf_loading_test() {
        env_logger::init();

        // We need to create the eventloop from the test thread.
        #[cfg(target_os = "windows")]
        use winit::platform::windows::EventLoopBuilderExtWindows;

        let event_loop = EventLoop::<State>::with_user_event().with_any_thread(true).build().expect("Failed to build event loop.");
        let mut app = DefaultAppHandler::new();

        event_loop.run_app(&mut app).unwrap();
    }
}
