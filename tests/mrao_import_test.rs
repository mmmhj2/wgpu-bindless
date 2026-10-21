#[cfg(test)]
mod test {
    use std::path::{Path, PathBuf};

    use rust_renderer::asset::{
        asset_manager::AssetManager,
        asset_types::{AssetData, texture_asset::{TexelData, TextureFormat}},
        importer::gltf_importer::GltfImporter,
    };

    const SOURCE_GLTF: &str = "resource/test_mrao.gltf";

    fn temp_workspace(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rust_renderer_mrao_{name}"));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).ok();
        }
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(SOURCE_GLTF, dir.join("test_mrao.gltf")).unwrap();
        dir
    }

    fn mrao_texels(manager: &AssetManager, name: &str) -> Vec<u8> {
        let asset = manager.get_asset(name).unwrap_or_else(|| panic!("asset '{name}' not found"));
        let asset = asset.read().expect("asset should be readable");
        match asset.get_data() {
            AssetData::TextureAssetType(texture) => {
                assert_eq!(texture.format, TextureFormat::Rgba8Unorm);
                assert_eq!(texture.width.get(), 2);
                assert_eq!(texture.height.map(|height| height.get()), Some(2));
                match &texture.texels {
                    TexelData::Plain(texels) => texels.to_vec(),
                    TexelData::Mipmapped(_) => panic!("MRAO assets use plain texel data"),
                }
            }
            _ => panic!("asset '{name}' is not a texture asset"),
        }
    }

    #[test]
    fn imports_all_supported_mrao_combinations() {
        let dir = temp_workspace("combinations");
        let gltf = dir.join("test_mrao.gltf");
        let manager = AssetManager::new();
        manager.import(GltfImporter::new(&gltf)).expect("import should succeed");

        // Primitive order follows the five materials in the fixture.
        let mut mesh_materials: Vec<(String, Option<String>)> = {
            let database = manager.get_database();
            database.iter()
                .filter_map(|(name, asset)| {
                    let asset = asset.read().expect("mesh asset should be readable");
                    match asset.get_data() {
                        AssetData::StaticMeshAssetType(mesh) => {
                            Some((name.clone(), mesh.material.mrao_texture_name.clone()))
                        }
                        _ => None,
                    }
                })
                .collect()
        };
        mesh_materials.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(mesh_materials.len(), 6);
        assert_eq!(
            mesh_materials[0].1.as_deref(),
            Some("test_mrao.gltf.mrao_mr0_ao1")
        );
        assert_eq!(
            mesh_materials[1].1.as_deref(),
            Some("test_mrao.gltf.mrao_mr2_ao2")
        );
        assert_eq!(
            mesh_materials[2].1.as_deref(),
            Some("test_mrao.gltf.mrao_mr0_none")
        );
        assert_eq!(
            mesh_materials[3].1.as_deref(),
            Some("test_mrao.gltf.mrao_none_ao1")
        );
        assert_eq!(mesh_materials[4].1, None);
        // The sixth material reuses the first MR/AO pair and must share its asset.
        assert_eq!(
            mesh_materials[5].1.as_deref(),
            Some("test_mrao.gltf.mrao_mr0_ao1")
        );

        fn expected(pixel: [u8; 4]) -> Vec<u8> {
            pixel.repeat(4)
        }

        // Separate MR/AO images: MR B = metallic, MR G = roughness, AO R = occlusion.
        assert_eq!(
            mrao_texels(&manager, "test_mrao.gltf.mrao_mr0_ao1"),
            expected([32, 80, 40, 255])
        );
        // Same ORM image: pure swizzle [R, B, G, 1].
        assert_eq!(
            mrao_texels(&manager, "test_mrao.gltf.mrao_mr2_ao2"),
            expected([32, 80, 40, 255])
        );
        // MR only: AO defaults to 1.
        assert_eq!(
            mrao_texels(&manager, "test_mrao.gltf.mrao_mr0_none"),
            expected([255, 80, 40, 255])
        );
        // AO only: metallic and roughness default to 1.
        assert_eq!(
            mrao_texels(&manager, "test_mrao.gltf.mrao_none_ao1"),
            expected([32, 255, 255, 255])
        );
    }

    #[test]
    fn deduplicates_mrao_assets_and_keeps_raw_textures() {
        let dir = temp_workspace("dedup");
        let gltf = dir.join("test_mrao.gltf");
        let manager = AssetManager::new();
        manager.import(GltfImporter::new(&gltf)).expect("import should succeed");

        let database = manager.get_database();
        let mrao_names: Vec<&str> = database.keys()
            .filter(|name| name.contains(".mrao_"))
            .map(String::as_str)
            .collect();

        assert_eq!(mrao_names.len(), 4, "one asset per unique MR/AO combination");

        for raw_texture in [
            "test_mrao.gltf.tex_0",
            "test_mrao.gltf.tex_1",
            "test_mrao.gltf.tex_2",
        ] {
            assert!(database.contains_key(raw_texture), "raw texture '{raw_texture}' should still be imported");
        }
    }

    #[test]
    fn fixture_path_is_stable() {
        assert!(Path::new(SOURCE_GLTF).is_file());
    }
}
