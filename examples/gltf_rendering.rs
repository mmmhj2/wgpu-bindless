use std::{path::Path, sync::Arc};

use cgmath::{InnerSpace, SquareMatrix, Transform};
use rust_renderer::{
    asset::{
        asset_manager::AssetManager,
        asset_store::{ImportJob, ImportManifest, ManifestEntry},
        asset_types::{AssetData::{SceneAssetType, TextureAssetType}, GpuAssetType},
        importer::{gltf_importer::GltfImporter, image_file_importer::ColorSpace},
    }, renderer::{
        drawable::draw_list::{DrawList, DrawableType}, orchestrator::{RenderOrchestrator, RendererConfig}, pipeline::{
            key::{PipelineKey, VertexLayoutKey},
            rasterizer_pipeline::UseRasterizerPipeline,
        }, postfx::{
            bloom::BloomPass,
            tonemap::AcesTonemapPass
        }, resources::{
            camera::{CameraPerspective, CameraPerspectiveBuilder, HasViewMatrix},
            lighting::{Light, LightKindEnum},
        }, scene::{graph::SceneGraph, skybox::SkyboxManager}, window::RendererState
    },
};
use winit::{application::ApplicationHandler, event::{KeyEvent, WindowEvent}, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::Window};

const GLTF_PATH: &str = "resource/DamagedHelmet.glb";

struct State {
    orchestrator: RenderOrchestrator,
    skybox: SkyboxManager,
    draw_list: DrawList,
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
                ..Default::default()
            },
        ).await;

        let scene_format = orchestrator.scene_format();
        let camera_bgl = orchestrator.camera().get_bind_group_layout().clone();
        let mut skybox = {
            let (device, pipelines) = orchestrator.device_and_pipelines_mut();
            SkyboxManager::new(
                device,
                &camera_bgl,
                scene_format,
                wgpu::TextureFormat::Depth32Float,
                pipelines,
            )
        };
        {
            let mut postfx = orchestrator
                .postfx_build_context()
                .expect("postfx should be enabled");
            let bloom = BloomPass::new(&mut postfx);
            postfx.push_pass(bloom);
            let tonemap = AcesTonemapPass::new(&mut postfx);
            postfx.push_pass(tonemap);
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

        let draw_list = {
            let scene_name = GltfImporter::scene_subasset_name(Path::new(GLTF_PATH), 0).expect("failed to resolve scene subasset name");
            let asset = asset_manager.get_asset(&scene_name).expect("gltf scene asset should be present");
            let asset = asset.read().expect("failed to read gltf scene asset");
            if let SceneAssetType(scene) = asset.get_data() {
                let mut ctx = orchestrator.mesh_upload_context(&asset_manager);
                SceneGraph::from_asset(scene).into_draw_list(&mut ctx)
            } else {
                panic!("asset '{scene_name}' is not a scene asset");
            }
        };

        log::info!("Loaded {} static mesh batches.", draw_list.count(DrawableType::OpaqueStatic));

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
        orchestrator.lighting_mut().set_ambient([0.03, 0.03, 0.03]);
        let light = Light::directional(cgmath::vec3(1.0, 0.0, 0.0), [1.0, 0.95, 0.9], 3.0);

        Self {
            orchestrator,
            skybox,
            draw_list,
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

        self.orchestrator.with_frame(|frame| {
            frame.render_scene(
                wgpu::Color {
                    r: 0.1,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                },
                |render_pass, gpu, scene| {
                    scene.camera.set_active_camera(camera);
                    scene.lighting.upload_lights(
                        gpu.queue(),
                        std::slice::from_ref(&light)
                    );
                    scene.bind_globals(gpu.device(), gpu.queue(), render_pass);
                    render_pass.set_rasterizer_pipeline(
                        scene.pipelines.get_or_create(
                            gpu.device(),
                            PipelineKey::CookTorrancePbr,
                            VertexLayoutKey::BASIC,
                        )
                    );
                    self.draw_list.draw(DrawableType::OpaqueStatic, render_pass);

                    skybox.draw_skybox(gpu.device(), scene.pipelines, render_pass);
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
