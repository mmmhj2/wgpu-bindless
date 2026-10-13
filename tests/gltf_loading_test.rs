#[cfg(test)]
#[cfg(target_os = "windows")]
mod test {
    use std::sync::Arc;
    use rust_renderer::{app::DefaultAppHandler, asset::{asset_manager::AssetManager, importer::gltf_importer::GltfImporter}, renderer::{device_interface::DeviceInterface, mesh::static_mesh_instance::StaticMeshInstance, pipeline::pipeline_state::PipelineStates, window::RendererState}};
    use winit::{event_loop::EventLoop, window::Window};

    pub struct State {
        window: Arc<Window>,
        device: DeviceInterface,
        pipeline: PipelineStates,
        gltf_models: Vec<StaticMeshInstance>
    }

    impl RendererState for State {
        async fn new(window: Arc<Window>) -> Self {
            let device = DeviceInterface::new(window.clone(), true).await.unwrap();
            let mut pipeline = PipelineStates::prepare_default_pipelines(&device, device.get_default_texture_format());

            let asset_manager = AssetManager::new();
            asset_manager.import(GltfImporter::new("resource/test_two_cubes.glb"));

            // TODO: currently gltf files are imported twice, once to recondition the asset and once to get the model info.
            // We should figure out some way to avoid this duplication, maybe by removing the `create_from_gltf_mesh` call
            // and enforce creation from asset instead of using gltf directly.
            let (document, _buffers, _images) = gltf::import("resource/test_two_cubes.glb").expect("Cannot open glb file.");
            
            let mut gltf_models = Vec::new();
            for mesh in document.meshes() {
                let instances = StaticMeshInstance::create_from_gltf_mesh(
                    &device,
                    &mut pipeline,
                    std::path::Path::new("resource/test_two_cubes.glb"),
                    &mesh,
                    &asset_manager);
                gltf_models.extend(instances);
            }

            println!("Read {} models in total.", gltf_models.len());

            Self {
                window: window.clone(),
                device,
                pipeline,
                gltf_models
            }
        }

        fn render(&mut self) -> Result<(), ()>{
            Err(())
        }
        
        fn get_device_interface(&self) -> &DeviceInterface {
            &self.device
        }
        
        fn get_device_interface_mut(&mut self) -> &mut DeviceInterface {
            &mut self.device
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
