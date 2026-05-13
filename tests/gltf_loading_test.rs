#[cfg(test)]
#[cfg(target_os = "windows")]
mod test {
    use std::sync::Arc;
    use rust_renderer::{app::DefaultAppHandler, renderer::{device_interface::DeviceInterface, mesh::instanced_mesh::InstancedMeshInstance, pipeline::pipeline_state::PipelineStates, window::RendererState}};
    use winit::{event_loop::EventLoop, window::Window};

    pub struct State {
        window: Arc<Window>,
        device: DeviceInterface,
        pipeline: PipelineStates,
        gltf_models: Vec<InstancedMeshInstance>
    }

    impl RendererState for State {
        async fn new(window: Arc<Window>) -> Self {
            let device = DeviceInterface::new(window.clone()).await.unwrap();
            let mut pipeline = PipelineStates::prepare_default_pipelines(&device);

            let (document, buffers, images) = gltf::import("resource/test_two_cubes.glb").expect("Cannot open glb file.");
            
            let mut gltf_models = Vec::new();
            for mesh in document.meshes() {
                let instances = InstancedMeshInstance::create_from_gltf_mesh(
                    &device,
                    pipeline.get_bindless_resource_manager_mut(),
                    &mesh,
                    &buffers,
                    &images);
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
