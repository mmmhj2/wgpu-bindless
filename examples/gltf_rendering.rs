use std::{path::Path, sync::Arc};

use cgmath::{InnerSpace, SquareMatrix, Transform};
use rust_renderer::{
    app::DefaultAppHandler, asset::{
        asset_manager::AssetManager,
        asset_store::{ImportJob, ImportManifest, ManifestEntry},
        asset_types::{AssetData::{SceneAssetType, TextureAssetType}, GpuAssetType},
        importer::{gltf_importer::GltfImporter, image_file_importer::ColorSpace},
    }, renderer::{
        context::{
            frame_context::Backbuffers, gpu_context::GpuContext, surface_context::SurfaceContext,
        },
        mesh::{drawable_mesh_traits::DrawableMesh, static_mesh_instance::StaticMeshInstance},
        pipeline::{
            camera::{Camera, CameraPerspective, CameraPerspectiveBuilder, HasViewMatrix, HasViewProjectionMatrix},
            pipeline_cache::PipelineKey,
            lighting::{Light, LightKindEnum},
            pipeline_state::PipelineStates,
            rasterizer_pipeline::UseRasterizerPipeline,
            skybox::SkyboxManager
        },
        postfx::{
            bloom::BloomPass,
            postfx_stack::{PostFxExtraInputs, PostFxInput, PostFxStack},
            tonemap::AcesTonemapPass
        }, window::RendererState
    },
};
use winit::{application::ApplicationHandler, event::{KeyEvent, WindowEvent}, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::Window};

const GLTF_PATH: &str = "resource/DamagedHelmet.glb";

struct State {
    window: Arc<Window>,
    gpu: GpuContext,
    surface: SurfaceContext<'static>,
    backbuffers: Backbuffers,
    pipeline: PipelineStates,
    skybox: SkyboxManager,
    postfx: PostFxStack,
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

impl RendererState<'static> for State {
    async fn new(window: Arc<Window>) -> Self {
        let mut gpu = GpuContext::new();
        let mut surface = SurfaceContext::new(gpu.instance(), window.clone()).unwrap();
        gpu.create_device(Some(surface.surface())).await.unwrap();
        surface.select_color_format(gpu.try_adapter().expect("adapter should be ready"), true);
        surface.configure(gpu.device());

        let surface_format = surface.get_color_format().expect("surface color format should be selected");
        let mut pipeline = PipelineStates::prepare_default_pipelines(gpu.device(), gpu.queue(), wgpu::TextureFormat::Rgba16Float);
        let mut skybox = SkyboxManager::new(gpu.device(), pipeline.get_camera_manager(), wgpu::TextureFormat::Rgba16Float, wgpu::TextureFormat::Depth32Float);
        let mut postfx = PostFxStack::new(wgpu::TextureFormat::Rgba16Float, surface_format);
        postfx.push_pass(Box::new(BloomPass::new(gpu.device(), wgpu::TextureFormat::Rgba16Float)));
        postfx.push_pass(Box::new(AcesTonemapPass::new(gpu.device(), wgpu::TextureFormat::Rgba16Float, surface_format)));

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
                StaticMeshInstance::create_from_scene_asset(gpu.device(), gpu.queue(), &mut pipeline, scene, &asset_manager)
            } else {
                panic!("asset '{scene_name}' is not a scene asset");
            }
        };

        let skybox_texture = {
            let asset = asset_manager.get_asset("skybox").expect("skybox array asset should be present");
            let asset = asset.read().expect("failed to read skybox asset");
            if let TextureAssetType(texture) = asset.get_data() {
                texture.upload("skybox", gpu.device(), gpu.queue())
            } else {
                panic!("skybox asset is not a texture asset");
            }
        };
        skybox.set_texture(skybox_texture);

        let camera = CameraPerspectiveBuilder::new().set_origin(cgmath::point3(0.0, 0.0, 3.0)).build();

        // Scene lighting: a single directional light plus a faint ambient term.
        pipeline.get_lighting_manager_mut().set_ambient([0.03, 0.03, 0.03]);
        let light = Light::directional(cgmath::vec3(1.0, 0.0, 0.0), [1.0, 0.95, 0.9], 3.0);

        Self {
            window,
            gpu,
            surface,
            backbuffers: Backbuffers::new(wgpu::TextureFormat::Depth32Float),
            pipeline,
            skybox,
            postfx,
            gltf_models,
            camera,
            light,
            _asset_manager: asset_manager
        }
    }

    fn render(&mut self) -> Result<(), ()>{
        self.window.request_redraw();

        let Some(frame) = self.backbuffers.new_frame(self.gpu.device(), &mut self.surface) else {
            return Ok(());
        };

        self.postfx.configure_with_size(self.gpu.device(), frame.get_size());

        let mut encoder = self.gpu.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
                    view: frame.get_depth_view(),
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
            self.pipeline.get_lighting_manager().upload_lights(
                self.gpu.queue(),
                std::slice::from_ref(&self.light)
            );
            self.pipeline.prepare_render_pass(self.gpu.device(), self.gpu.queue(), &mut rp);
            rp.set_rasterizer_pipeline(self.pipeline.query_pipeline(PipelineKey::CookTorrancePbr).expect("cook torrance pipeline not found"));
            for m in &self.gltf_models {
                m.draw(&mut rp);
            }

            self.skybox.draw_skybox(self.gpu.device(), &mut rp);
        }

        let final_view = frame.get_color_view();

        let inputs = PostFxExtraInputs::new()
            .with(PostFxInput::Depth, frame.get_depth_view());

        self.postfx.run(
            self.gpu.device(),
            self.pipeline.get_bindless_resource_manager_mut().get_samplers_mut(),
            &mut encoder,
            final_view,
            &inputs
        );

        self.gpu.queue().submit(std::iter::once(encoder.finish()));
        frame.present();

        Ok(())
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
