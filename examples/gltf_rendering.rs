use std::{path::Path, sync::Arc};

use cgmath::{InnerSpace, SquareMatrix, Transform};
use rust_renderer::{
    app::DefaultAppHandler, asset::{
        asset_manager::AssetManager,
        asset_store::{ImportJob, ImportManifest, ManifestEntry},
        asset_types::{AssetData::{SceneAssetType, TextureAssetType}, GpuAssetType},
        importer::{gltf_importer::GltfImporter, image_file_importer::ColorSpace},
    }, renderer::{
        device_interface::DeviceInterface, mesh::{drawable_mesh_traits::DrawableMesh, static_mesh_instance::StaticMeshInstance}, pipeline::{camera::{Camera, CameraPerspective, CameraPerspectiveBuilder, HasViewMatrix, HasViewProjectionMatrix}, pipeline_state::PipelineStates, rasterizer_pipeline::UseRasterizerPipeline, skybox::SkyboxManager}, postfx::{bloom::BloomPass, postfx_stack::{PostFxExtraInputs, PostFxInput, PostFxStack}, tonemap::AcesTonemapPass}, window::RendererState
    },
};
use winit::{application::ApplicationHandler, event::{KeyEvent, WindowEvent}, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::Window};

const GLTF_PATH: &str = "resource/DamagedHelmet.glb";

struct State {
    window: Arc<Window>,
    device: DeviceInterface,
    pipeline: PipelineStates,
    skybox: SkyboxManager,
    postfx: PostFxStack,
    gltf_models: Vec<StaticMeshInstance>,
    camera: CameraPerspective,
    _asset_manager: AssetManager
}

impl State {
    /// Rotate the camera with a rotation in axis-angle (in degree) representation in the world space.
    pub fn rotate_camera_ws(&mut self, rotate_axis_angle: cgmath::Vector3<f32>) {
        let angle = rotate_axis_angle.magnitude();
        let axis = rotate_axis_angle.normalize();

        let rotmat = cgmath::Matrix3::from_axis_angle(axis, cgmath::Deg(angle));
        let new_origin = rotmat.transform_point(self.camera.get_underlying_camera().get_origin());
        let new_up = rotmat * self.camera.get_underlying_camera().get_up();
        self.camera = CameraPerspectiveBuilder::new().set_origin(new_origin).set_up(new_up).build();
    }

    /// Rotate the camera with a rotation in axis-angle (in degree) representation in the camera space.
    pub fn rotate_camera_cs(&mut self, rotate_axis_angle: cgmath::Vector3<f32>) {
        let vec = cgmath::vec4(rotate_axis_angle.x, rotate_axis_angle.y, rotate_axis_angle.z, 0.0);
        let vec_ws = self.camera.get_view_matrix().invert().expect("view matrix should be invertible") * vec;
        self.rotate_camera_ws(vec_ws.truncate());
    }
}

impl RendererState for State {
    async fn new(window: Arc<Window>) -> Self {
        let device = DeviceInterface::new(window.clone(), true).await.unwrap();
        let mut pipeline = PipelineStates::prepare_default_pipelines(&device, wgpu::TextureFormat::Rgba16Float);
        let mut skybox = SkyboxManager::new(&device, pipeline.get_camera_manager(), wgpu::TextureFormat::Rgba16Float, wgpu::TextureFormat::Depth32Float);
        let mut postfx = PostFxStack::new(&device, wgpu::TextureFormat::Rgba16Float);
        postfx.push_pass(Box::new(BloomPass::new(&device, wgpu::TextureFormat::Rgba16Float)));
        postfx.push_pass(Box::new(AcesTonemapPass::new(&device, wgpu::TextureFormat::Rgba16Float, device.get_default_texture_format())));

        // Load through the import manifest: on the first run the jobs are
        // imported and persisted; later runs load the cached containers unless a
        // source file or the importer changed.
        let asset_manager = AssetManager::new();
        let manifest_path = Path::new("resource/import_gltf_manifest.ff");
        let mut manifest = ImportManifest::load(manifest_path).expect("failed to load import manifest");
        if manifest.entries.is_empty() {
            manifest.entries = vec![
                ManifestEntry { job: ImportJob::gltf(GLTF_PATH), outputs: Vec::new() },
                ManifestEntry {
                    job: ImportJob::image_array(
                        Some(String::from("skybox")),
                        [
                            "resource/skybox/right.jpg",
                            "resource/skybox/left.jpg",
                            "resource/skybox/top.jpg",
                            "resource/skybox/bottom.jpg",
                            "resource/skybox/front.jpg",
                            "resource/skybox/back.jpg"
                        ],
                        ColorSpace::Srgb
                    ),
                    outputs: Vec::new(),
                },
            ];
            manifest.save(manifest_path).expect("failed to save import manifest");
        }
        asset_manager.import_manifest(manifest_path).expect("failed to import manifest");

        let gltf_models = {
            let scene_name = GltfImporter::scene_subasset_name(Path::new(GLTF_PATH), 0).expect("failed to resolve scene subasset name");
            let asset = asset_manager.get_asset(&scene_name).expect("gltf scene asset should be present");
            let asset = asset.read().expect("failed to read gltf scene asset");
            if let SceneAssetType(scene) = asset.get_data() {
                StaticMeshInstance::create_from_scene_asset(&device, &mut pipeline, scene, &asset_manager)
            } else {
                panic!("asset '{scene_name}' is not a scene asset");
            }
        };

        let skybox_texture = {
            let asset = asset_manager.get_asset("skybox").expect("skybox array asset should be present");
            let asset = asset.read().expect("failed to read skybox asset");
            if let TextureAssetType(texture) = asset.get_data() {
                texture.upload("skybox", device.get_device(), device.get_queue())
            } else {
                panic!("skybox asset is not a texture asset");
            }
        };
        skybox.set_texture(skybox_texture);

        let camera = CameraPerspectiveBuilder::new().set_origin(cgmath::point3(0.0, 0.0, 3.0)).build();

        Self {
            window: window.clone(),
            device,
            pipeline,
            skybox,
            postfx,
            gltf_models,
            camera,
            _asset_manager: asset_manager
        }
    }

    fn render(&mut self) -> Result<(), ()>{
        self.window.request_redraw();

        // We can't render unless the surface is configured
        if !self.device.is_presentation_ready() {
            return Ok(());
        }
            
        let output = match self.device.get_current_surface_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                self.device.configure_surface();
                surface_texture
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                // Skip this frame
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.device.configure_surface();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                panic!("Lost device");
            }
        };

        self.postfx.configure_with_surface(&self.device);

        let mut encoder = self.device.get_device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.postfx.scene_target(),
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
                    view: self.device.get_depth_texture_view().expect("Unprepared"),
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

            self.pipeline.set_active_camera(self.camera.clone());
            self.pipeline.prepare_render_pass(self.get_device_interface(), &mut rp);
            rp.set_rasterizer_pipeline(self.pipeline.query_pipeline(&String::from("cook_torrance")).expect("cook torrance pipeline not found"));
            for m in &self.gltf_models {
                m.draw(&mut rp);
            }

            self.skybox.draw_skybox(&self.device, &mut rp);
        }

        let final_view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let inputs = PostFxExtraInputs::new()
            .with(PostFxInput::Depth, self.device.get_depth_texture_view().expect("depth view should be ready"));

        self.postfx.run(
            &self.device,
            self.pipeline.get_bindless_resource_manager_mut().get_samplers_mut(),
            &mut encoder,
            &final_view,
            &inputs
        );

        self.device.get_queue().submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
    
    fn get_device_interface(&self) -> &DeviceInterface {
        &self.device
    }
    
    fn get_device_interface_mut(&mut self) -> &mut DeviceInterface {
        &mut self.device
    }
}

struct MyAppState {
    state: Option<State>,
}

impl MyAppState {
    pub fn new() -> Self {
        Self {
            state: None,
        }
    }
}

impl ApplicationHandler<State> for MyAppState {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes();
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        self.state = Some(pollster::block_on(State::new(window)));
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: State) {
        self.state = Some(event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                // state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(_) => {
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => match (code, key_state.is_pressed()) {
                (KeyCode::Escape, true) => event_loop.exit(),
                (KeyCode::KeyA, _) => state.rotate_camera_cs(cgmath::vec3(0.0, -1.0, 0.0)),
                (KeyCode::KeyD, _) => state.rotate_camera_cs(cgmath::vec3(0.0,  1.0, 0.0)),
                (KeyCode::KeyW, _) => state.rotate_camera_cs(cgmath::vec3(-1.0, 0.0, 0.0)),
                (KeyCode::KeyS, _) => state.rotate_camera_cs(cgmath::vec3( 1.0, 0.0, 0.0)),
                _ => {}
            },
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::<State>::with_user_event().build().expect("Failed to build event loop.");
    let mut app = MyAppState::new();
    event_loop.run_app(&mut app).unwrap();
}
