#[cfg(any(target_os = "windows", target_os = "linux"))]
mod common;

#[cfg(test)]
#[cfg(any(target_os = "windows", target_os = "linux"))]
mod test {
    use std::sync::Arc;
    use crate::common;

    use rust_renderer::{
        asset::{
            asset_manager::AssetManager, asset_types::AssetData,
            importer::gltf_importer::GltfImporter,
        }, renderer::{
            drawable::draw_list::{DrawList, DrawableType},
            orchestrator::{RenderOrchestrator, RendererConfig}, window::RendererState,
        },
    };
    use winit::window::Window;
    const FRAMES_TO_RENDER: u32 = 1;

    pub struct State {
        orchestrator: RenderOrchestrator,
        _draw_list: DrawList,
        _asset_manager: AssetManager,
    }

    impl RendererState for State {
        async fn new(window: Arc<Window>) -> Self {
            let mut orchestrator = RenderOrchestrator::new(window, RendererConfig::default()).await;

            let asset_manager = AssetManager::new();
            asset_manager.import(GltfImporter::new("resource/test_two_cubes.glb")).expect("import should succeed");

            // The scene graph is baked into a `SceneAsset` at import time, so
            // the glTF file is parsed exactly once.
            let draw_list = {
                let scene_name = GltfImporter::scene_subasset_name(std::path::Path::new("resource/test_two_cubes.glb"), 0).expect("scene name should resolve");
                let asset = asset_manager.get_asset(&scene_name).expect("gltf scene asset should be present");
                let asset = asset.read().expect("failed to read gltf scene asset");
                if let AssetData::SceneAssetType(scene) = asset.get_data() {
                    orchestrator
                        .prepare_scene(&asset_manager, scene)
                        .expect("failed to prepare the scene")
                } else {
                    panic!("asset '{scene_name}' is not a scene asset");
                }
            };

            println!("Read {} batches in total.", draw_list.count(DrawableType::OpaqueStatic));
            assert_eq!(
                draw_list.count(DrawableType::OpaqueStatic),
                2,
                "test_two_cubes has two distinct primitives"
            );

            Self {
                orchestrator,
                _draw_list: draw_list,
                _asset_manager: asset_manager,
            }
        }

        fn resize(&mut self, width: u32, height: u32) {
            self.orchestrator.resize(width, height);
        }

        fn render(&mut self) -> Result<(), ()> {
            self.orchestrator.with_frame(|frame| {
                frame.render_scene(
                    wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0 },
                    |_, _, _| {},
                );
            });

            Ok(())
        }
    }

    #[test]
    fn gltf_loading_test() {
        env_logger::init();

        let frames = common::run_windowed::<State>(FRAMES_TO_RENDER);
        assert_eq!(
            frames,
            Some(FRAMES_TO_RENDER),
            "the glTF test should render one frame before the limit stops it"
        );
    }
}
