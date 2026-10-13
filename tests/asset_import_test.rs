#[cfg(test)]
mod test {
    use std::path::PathBuf;

use rust_renderer::asset::{asset_manager::AssetManager, asset_types::{Asset, texture_asset::{TexelData, TextureDimension, TextureFormat}}, importer::{gltf_importer::GltfImporter, image_file_importer::{ColorSpace, ImageFileImporter}}};

    #[test]
    fn import_gltf_test() {
        let test_filename: PathBuf;
        {
            let asset_manager = AssetManager::new();
            asset_manager.import(GltfImporter::new("resource/test_two_cubes.glb"));
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
    fn import_single_image_test() {
        let test_filename: PathBuf;
        {
            let asset_manager = AssetManager::new();
            asset_manager.import(ImageFileImporter::new("resource/skybox/back.jpg", ColorSpace::Srgb));
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
