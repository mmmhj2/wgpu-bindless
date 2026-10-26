use std::{path::Path, sync::Arc};

use cgmath::{InnerSpace, SquareMatrix, Transform};
use rust_renderer::{
    asset::{
        asset_manager::AssetManager,
        asset_store::{ImportJob, ImportManifest, ManifestEntry},
        asset_types::{AssetData::{SceneAssetType, TextureAssetType}, GpuAssetType},
        importer::{gltf_importer::GltfImporter, image_file_importer::ColorSpace},
    },
    renderer::{
        mesh::{drawable_mesh_traits::DrawableMesh, static_mesh_instance::StaticMeshInstance},
        orchestrator::{RenderOrchestrator, RendererConfig},
        pipeline::{
            camera::{CameraPerspective, CameraPerspectiveBuilder, HasViewMatrix},
            pipeline_cache::PipelineKey,
            lighting::{Light, LightKindEnum},
            rasterizer_pipeline::UseRasterizerPipeline,
            skybox::SkyboxManager
        },
        postfx::{
            bloom::BloomPass,
            tonemap::AcesTonemapPass
        },
        window::RendererState
    },
};
use winit::{application::ApplicationHandler, event::{KeyEvent, WindowEvent}, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::Window};

const GLTF_PATH: &str = "resource/DamagedHelmet.glb";

struct State {
    orchestrator: RenderOrchestrator,
    skybox: SkyboxManager,
    gltf_models: Vec<StaticMeshInstance>,
    camera: CameraPerspective,
    light: Light,
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

    /// Rotate the directional light with an axis-angle (in degree) rotation.
    pub fn rotate_light_ws(&mut self, rotate_axis_angle: cgmath::Vector3<f32>) {
        let angle = rotate_axis_angle.magnitude();
        let axis = rotate_axis_angle.normalize();

        let rotmat = cgmath::Matrix3::from_axis_angle(axis, cgmath::Deg(angle));
        match &mut self.light.light {
            LightKindEnum::Directional(d) => d.direction = rotmat * d.direction,
            _ => panic!("Light is not directional")
        };
    }
}

impl RendererState for State {
    async fn new(window: Arc<Window>) -> Self {
        let mut orchestrator = RenderOrchestrator::new(
            window,
            RendererConfig {
                request_srgb_surface: true,
                depth_format: wgpu::TextureFormat::Depth32Float,
                scene_format: Some(wgpu::TextureFormat::Rgba16Float),
                enable_postfx: true,
            },
        ).await;

        let scene_format = orchestrator.scene_format();
        let surface_format = orchestrator.surface_format();
        let mut skybox = SkyboxManager::new(
            orchestrator.device(),
            orchestrator.pipelines().get_camera_manager(),
            scene_format,
            wgpu::TextureFormat::Depth32Float
        );
        {
            let (device, postfx) = orchestrator.device_and_postfx_mut();
            let postfx = postfx.expect("postfx should be enabled");
            postfx.push_pass(Box::new(BloomPass::new(device, scene_format)));
            postfx.push_pass(Box::new(AcesTonemapPass::new(device, scene_format, surface_format)));
        }

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
                let (gpu, pipelines) = orchestrator.gpu_and_pipelines_mut();
                StaticMeshInstance::create_from_scene_asset(gpu.device(), gpu.queue(), pipelines, scene, &asset_manager)
            } else {
                panic!("asset '{scene_name}' is not a scene asset");
            }
        };

        let skybox_texture = {
            let asset = asset_manager.get_asset("skybox").expect("skybox array asset should be present");
            let asset = asset.read().expect("failed to read skybox asset");
            if let TextureAssetType(texture) = asset.get_data() {
                texture.upload("skybox", orchestrator.device(), orchestrator.queue())
            } else {
                panic!("skybox asset is not a texture asset");
            }
        };
        skybox.set_texture(skybox_texture);

        let camera = CameraPerspectiveBuilder::new().set_origin(cgmath::point3(0.0, 0.0, 3.0)).build();

        // Scene lighting: a single directional light plus a faint ambient term.
        orchestrator.pipelines_mut().get_lighting_manager_mut().set_ambient([0.03, 0.03, 0.03]);
        let light = Light::directional(cgmath::vec3(1.0, 0.0, 0.0), [1.0, 0.95, 0.9], 3.0);

        Self {
            orchestrator,
            skybox,
            gltf_models,
            camera,
            light,
            _asset_manager: asset_manager
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.orchestrator.resize(width, height);
    }

    fn render(&mut self) -> Result<(), ()>{
        let camera = self.camera.clone();
        let light = self.light.clone();
        let skybox = &self.skybox;
        let models = &self.gltf_models;

        self.orchestrator.with_frame(|frame| {
            frame.render_scene(
                wgpu::Color {
                    r: 0.1,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                },
                |render_pass, gpu, pipelines| {
                    pipelines.set_active_camera(camera);
                    pipelines.get_lighting_manager().upload_lights(
                        gpu.queue(),
                        std::slice::from_ref(&light)
                    );
                    pipelines.prepare_render_pass(gpu.device(), gpu.queue(), render_pass);
                    render_pass.set_rasterizer_pipeline(
                        pipelines.query_pipeline(PipelineKey::CookTorrancePbr).expect("cook torrance pipeline not found")
                    );
                    for model in models {
                        model.draw(render_pass);
                    }

                    skybox.draw_skybox(gpu.device(), render_pass);
                },
            );
        });

        Ok(())
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
                (KeyCode::ArrowUp   , _) => state.rotate_light_ws(cgmath::vec3(-1.0, 0.0, 0.0)),
                (KeyCode::ArrowDown , _) => state.rotate_light_ws(cgmath::vec3( 1.0, 0.0, 0.0)),
                (KeyCode::ArrowLeft , _) => state.rotate_light_ws(cgmath::vec3(1.0, -1.0, 0.0)),
                (KeyCode::ArrowRight, _) => state.rotate_light_ws(cgmath::vec3(0.0,  1.0, 0.0)),
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
