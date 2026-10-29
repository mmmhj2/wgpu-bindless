#[cfg(test)]
mod test {
    use std::path::PathBuf;

use rust_renderer::asset::{asset_manager::AssetManager, asset_types::{Asset, AssetData, texture_asset::{TexelData, TextureDimension, TextureFormat}}, importer::{gltf_importer::GltfImporter, image_file_importer::{ColorSpace, ImageArrayImporter, ImageFileImporter}}};

    #[test]
    fn import_gltf_test() {
        let test_filename: PathBuf;
        {
            let asset_manager = AssetManager::new();
            asset_manager.import(GltfImporter::new("resource/test_two_cubes.glb")).expect("import should succeed");
            asset_manager.save_to_disk().unwrap();
            {
                let db = asset_manager.get_database();
                let a = db.iter().next().expect("some asset should present");
                test_filename = a.1.as_ref().read().unwrap().get_asset_file_path().expect("asset path should present").into();
            }
        }

        Asset::load_from_disk(&test_filename).unwrap();
    }

    #[test]
    fn import_gltf_scene_test() {
        let gltf_path = std::path::Path::new("resource/test_two_cubes.glb");
        let asset_manager = AssetManager::new();
        asset_manager.import(GltfImporter::new(gltf_path)).expect("import should succeed");

        let scene_name = GltfImporter::scene_subasset_name(gltf_path, 0).expect("scene name should resolve");
        let asset = asset_manager.get_asset(&scene_name).expect("scene asset should be present");
        let asset = asset.read().unwrap();
        match asset.get_data() {
            AssetData::SceneAssetType(scene) => {
                // The test scene has two root nodes, each with one primitive.
                assert_eq!(scene.nodes.len(), 2, "scene asset should preserve both glTF nodes.");
                assert_eq!(scene.roots.len(), 2, "both test nodes are scene roots.");
                assert!(
                    scene.nodes.iter().all(|node| !node.mesh_subasset_names.is_empty()),
                    "every test node should reference a mesh primitive."
                );
                for root in &scene.roots {
                    assert!((*root as usize) < scene.nodes.len(), "scene roots must index into nodes.");
                }
                for node in &scene.nodes {
                    for child in &node.children {
                        assert!((*child as usize) < scene.nodes.len(), "scene children must index into nodes.");
                    }
                }

                // Local transforms are preserved instead of being baked into a world matrix.
                let translated = scene.nodes.iter()
                    .find(|node| node.name.as_deref() == Some("cube_textured"))
                    .expect("cube_textured node should be present.");
                assert!(
                    (translated.local_matrix[3][1] - 3.1607275).abs() < 1e-5,
                    "cube_textured should keep its local Y translation."
                );
            }
            _ => panic!("expected a scene asset, got another asset kind."),
        }
    }

    #[test]
    fn import_single_image_test() {
        let test_filename: PathBuf;
        {
            let asset_manager = AssetManager::new();
            asset_manager.import(ImageFileImporter::new("resource/skybox/back.jpg", ColorSpace::Srgb)).expect("import should succeed");
            asset_manager.save_to_disk().unwrap();
            {
                let db = asset_manager.get_database();
                let a = db.iter().next().expect("some asset should present");
                test_filename = a.1.as_ref().read().unwrap().get_asset_file_path().expect("asset path should present").into();
            }
        }

        Asset::load_from_disk(&test_filename).unwrap();
    }

    #[test]
    fn import_image_array_test() {
        let test_filename: PathBuf;
        {
            let asset_manager = AssetManager::new();
            asset_manager.import(ImageArrayImporter::new(
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
            )).expect("import should succeed");
            asset_manager.save_to_disk().unwrap();
            {
                let db = asset_manager.get_database();
                let a = db.iter().next().expect("some asset should present");
                test_filename = a.1.as_ref().read().unwrap().get_asset_file_path().expect("asset path should present").into();
            }
        }

        Asset::load_from_disk(&test_filename).unwrap();
    }

    #[test]
    fn import_gltf_color_texture_test() {
        let (document, _buffers, images) = gltf::import("resource/test_two_cubes.glb").expect("Failed to import GLB file.");

        let mesh = document.meshes()
            .find(|x| x.name().unwrap_or_default() == "cube_textured")
            .expect("Cannot find cube_textured.");
        let primitive = mesh.primitives().next().expect("cube_textured has no primitives.");
        let material = primitive.material();

        let texture = material.pbr_metallic_roughness()
            .base_color_texture().expect("cube_textured has no base color texture.").texture();
        let asset = GltfImporter::import_from_gltf_texture(&texture, &images, true).expect("Texture import unsuccessful.");

        assert_eq!(asset.format, TextureFormat::Rgba8Srgb);
        assert_eq!(asset.width.get(), 32);
        assert_eq!(asset.height.map(|h| h.get()), Some(32));
        assert!(matches!(asset.dimension, TextureDimension::D2));
    }

    #[test]
    fn import_gltf_normal_texture_test() {
        let (document, _buffers, images) = gltf::import("resource/test_two_cubes.glb").expect("Failed to import GLB file.");

        let mesh = document.meshes()
            .find(|x| x.name().unwrap_or_default() == "cube_textured")
            .expect("Cannot find cube_textured.");
        let primitive = mesh.primitives().next().expect("cube_textured has no primitives.");
        let material = primitive.material();

        let texture = material.normal_texture().expect("cube_textured has no normal map texture.").texture();
        let asset = GltfImporter::import_from_gltf_texture(&texture, &images, false).expect("Texture import unsuccessful.");

        // Normal maps are linear 4-channel textures, not sRGB and not Snorm.
        assert_eq!(asset.format, TextureFormat::Rgba8Unorm);
        assert_eq!(asset.width.get(), 32);
        assert_eq!(asset.height.map(|h| h.get()), Some(32));
        assert!(matches!(asset.dimension, TextureDimension::D2));

        match &asset.texels {
            TexelData::Plain(data) => {
                assert_eq!(data[0], 128);
                assert_eq!(data[1], 128);
                assert_eq!(data[2], 255);
            }
            _ => panic!("expected plain texel data for a single image."),
        }
    }
}
