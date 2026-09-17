use std::{path::Path, sync::Arc};

use rust_renderer::{app::DefaultAppHandler, asset::{asset_manager::AssetManager, asset_types::{AssetData::TextureAssetType, GpuAssetType, texture_asset::TextureAsset}, importer::image_file_importer::ImageFileImporter}, renderer::{device_interface::DeviceInterface, mesh::{drawable_mesh_traits::DrawableMesh, immediate_mesh::ImmediateMeshBuilder, static_mesh_instance::StaticMeshInstance}, pipeline::{camera::CameraPerspectiveBuilder, framebuffers::Framebuffers, pbr_material::PBRMaterial, pipeline_state::PipelineStates, rasterizer_pipeline::UseRasterizerPipeline, sampler::SamplerDescription, skybox::SkyboxManager, texture::{Texture, TextureType}}, window::RendererState}};
use winit::{event_loop::EventLoop, window::Window};

pub struct State {
    window: Arc<Window>,
    device: DeviceInterface,
    pipelines: PipelineStates,
    asset_manager: AssetManager,
    material: PBRMaterial
}

impl RendererState for State {
    async fn new(window: Arc<Window>) -> Self {
        let device = DeviceInterface::new(window.clone(), true).await.unwrap();
        let mut pipelines = PipelineStates::prepare_default_pipelines(&device, device.get_default_texture_format());

        let asset_manager = AssetManager::new();
        asset_manager.import(Path::new("resource/skybox/back.jpg"), ImageFileImporter::new(true, None));

        let texture_view;
        let sampler;
        {
            let asset_db = asset_manager.get_database();
            let texture_asset = asset_db.iter().next().expect("at least one asset should be imported successfully");
            if let TextureAssetType(asset) = texture_asset.1.read().unwrap().get_data() {
                let texture = asset.upload(texture_asset.0, device.get_device(), device.get_queue());
                let wgpu_texture: wgpu::Texture = texture.into();
                texture_view = wgpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
                sampler = asset.sampler.clone().into();
            } else {
                panic!("unexpected asset type");
            }
        }

        let material = PBRMaterial::create_from_views(
            device.get_device(),
            pipelines.get_bindless_resource_manager_mut(),
            Some((texture_view, Some(sampler))),
            None,
            None
        ).expect("failed to allocate material.");

        Self {
            window: window.clone(),
            device,
            pipelines,
            asset_manager,
            material
        }
    }

    fn render(&mut self) -> Result<(), ()> {
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
                // You could recreate the devices and all resources
                // created with it here, but we'll just bail
                panic!("Lost device");
            }
        };

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.get_device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // Do some draw call here
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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

            let mut mesh_builder = ImmediateMeshBuilder::new(
                &self.device,
                self.pipelines.get_bindless_resource_manager_mut(),
                Some(self.material.clone()));
            mesh_builder.color3f([1.0, 1.0, 1.0]);

            mesh_builder.texcoord2f([0.0, 0.0]);
            mesh_builder.vertex3f([-0.5, 0.5, 1.0]);
            mesh_builder.texcoord2f([1.0, 0.0]);
            mesh_builder.vertex3f([-0.5,-0.5, 1.0]);
            mesh_builder.texcoord2f([1.0, 1.0]);
            mesh_builder.vertex3f([ 0.5,-0.5, 1.0]);

            mesh_builder.texcoord2f([0.0, 0.0]);
            mesh_builder.vertex3f([-0.5, 0.5, 1.0]);
            mesh_builder.texcoord2f([1.0, 1.0]);
            mesh_builder.vertex3f([ 0.5,-0.5, 1.0]);
            mesh_builder.texcoord2f([0.0, 1.0]);
            mesh_builder.vertex3f([ 0.5, 0.5, 1.0]);
            let mesh = mesh_builder.commit(&self.device, self.pipelines.get_mesh_manager());

            self.pipelines.prepare_render_pass(&self.device, &mut render_pass);
            render_pass.set_rasterizer_pipeline(self.pipelines.get_default_pipeline());
            mesh.draw(&mut render_pass);
        }

        // submit will accept anything that implements IntoIter
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
