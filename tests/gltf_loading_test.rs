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
            context::{
                frame_context::Backbuffers, gpu_context::GpuContext,
                surface_context::SurfaceContext,
            },
            mesh::static_mesh_instance::StaticMeshInstance,
            pipeline::pipeline_state::PipelineStates,
            window::RendererState,
        },
    };
    use winit::{event_loop::EventLoop, window::Window};

    pub struct State {
        window: Arc<Window>,
        gpu: GpuContext,
        surface: SurfaceContext<'static>,
        backbuffers: Backbuffers,
        pipeline: PipelineStates,
        gltf_models: Vec<StaticMeshInstance>,
    }

    impl RendererState<'static> for State {
        async fn new(window: Arc<Window>) -> Self {
            let mut gpu = GpuContext::new();
            let mut surface = SurfaceContext::new(gpu.instance(), window.clone()).unwrap();
            gpu.create_device(Some(surface.surface())).await.unwrap();
            surface.select_color_format(gpu.try_adapter().expect("adapter should be ready"), true);

            let format = surface.get_color_format().expect("surface color format should be selected");
            let mut pipeline = PipelineStates::prepare_default_pipelines(gpu.device(), gpu.queue(), format);

            let asset_manager = AssetManager::new();
            asset_manager.import(GltfImporter::new("resource/test_two_cubes.glb")).expect("import should succeed");

            // The scene graph is baked into a `SceneAsset` at import time, so
            // the glTF file is parsed exactly once.
            let gltf_models = {
                let scene_name = GltfImporter::scene_subasset_name(std::path::Path::new("resource/test_two_cubes.glb"), 0).expect("scene name should resolve");
                let asset = asset_manager.get_asset(&scene_name).expect("gltf scene asset should be present");
                let asset = asset.read().expect("failed to read gltf scene asset");
                if let AssetData::SceneAssetType(scene) = asset.get_data() {
                    StaticMeshInstance::create_from_scene_asset(gpu.device(), gpu.queue(), &mut pipeline, scene, &asset_manager)
                } else {
                    panic!("asset '{scene_name}' is not a scene asset");
                }
            };

            println!("Read {} models in total.", gltf_models.len());

            Self {
                window,
                gpu,
                surface,
                backbuffers: Backbuffers::new(wgpu::TextureFormat::Depth32Float),
                pipeline,
                gltf_models,
            }
        }

        fn render(&mut self) -> Result<(), ()> {
            Err(())
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
