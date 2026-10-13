use std::{path::Path, sync::Arc};

use rust_renderer::{
    app::DefaultAppHandler,
    asset::{
        asset_manager::AssetManager,
        asset_store::{ImportJob, ImportManifest, ManifestEntry},
        asset_types::{AssetData::{SceneAssetType, TextureAssetType}, GpuAssetType},
        importer::{gltf_importer::GltfImporter, image_file_importer::ColorSpace},
    },
    renderer::{device_interface::DeviceInterface, mesh::{drawable_mesh_traits::DrawableMesh, static_mesh_instance::StaticMeshInstance}, pipeline::{camera::CameraPerspectiveBuilder, framebuffers::Framebuffers, pipeline_state::PipelineStates, rasterizer_pipeline::UseRasterizerPipeline, skybox::SkyboxManager}, window::RendererState}
};
use winit::{event_loop::EventLoop, window::Window};

const GLTF_PATH: &str = "resource/test_two_cubes.glb";

pub struct State {
    window: Arc<Window>,
    device: DeviceInterface,
    pipeline: PipelineStates,
    skybox: SkyboxManager,
    fb: Framebuffers,
    gltf_models: Vec<StaticMeshInstance>,
    _asset_manager: AssetManager
}

impl RendererState for State {
    async fn new(window: Arc<Window>) -> Self {
        let device = DeviceInterface::new(window.clone(), true).await.unwrap();
        let mut pipeline = PipelineStates::prepare_default_pipelines(&device, wgpu::TextureFormat::Rgba16Float);
        let mut skybox = SkyboxManager::new(&device, pipeline.get_camera_manager(), wgpu::TextureFormat::Rgba16Float, wgpu::TextureFormat::Depth32Float);
        let fb = Framebuffers::new(&device, wgpu::TextureFormat::Rgba16Float);

        // Load through the import manifest: on the first run the jobs are
        // imported and persisted; later runs load the cached containers unless a
        // source file or the importer changed.
        let asset_manager = AssetManager::new();
        let manifest_path = Path::new("resource/import_manifest.ff");
        let mut manifest = ImportManifest::load(manifest_path);
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
            manifest.save(manifest_path);
        }
        asset_manager.import_manifest(manifest_path);

        let gltf_models = {
            let scene_name = GltfImporter::scene_subasset_name(Path::new(GLTF_PATH), 0);
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

        Self {
            window: window.clone(),
            device,
            pipeline,
            skybox,
            fb,
            gltf_models,
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

        self.fb.configure_with_surface(&self.device);

        let mut encoder = self.device.get_device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.fb.get_hdr_framebuffer().expect("hdr framebuffer should be ready"),
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

            let camera = CameraPerspectiveBuilder::new().set_origin(cgmath::point3(0.0, 0.0, -3.0)).build();

            self.pipeline.set_active_camera(camera);
            self.pipeline.prepare_render_pass(self.get_device_interface(), &mut rp);
            rp.set_rasterizer_pipeline(self.pipeline.query_pipeline(&String::from("cook_torrance")).expect("cook torrance pipeline not found"));
            for m in &self.gltf_models {
                m.draw(&mut rp);
            }

            self.skybox.draw_skybox(&self.device, &mut rp);
        }

        self.fb.bloom(&self.device, &mut encoder, 0.003, 0.05);

        let final_view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.fb.tonemap_to(&self.device, &mut encoder, &final_view);

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

fn main() {
    env_logger::init();

    let event_loop = EventLoop::<State>::with_user_event().build().expect("Failed to build event loop.");
    let mut app = DefaultAppHandler::<State>::new();
    event_loop.run_app(&mut app).unwrap();
}
