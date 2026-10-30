#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use rust_renderer::asset::{
        asset_manager::AssetManager,
        asset_types::AssetData,
        importer::gltf_importer::GltfImporter,
    };

    /// `resource/Fox.glb` (glTF-Sample-Assets, CC0) has one skinned primitive,
    /// one skin with 24 joints and three animations.
    const SOURCE_GLTF: &str = "resource/Fox.glb";
    const JOINT_COUNT: usize = 24;

    /// Copy the skinned test model into an isolated temp workspace.
    ///
    /// Importers persist assets next to their source, so copying the source
    /// keeps generated `*.asset` files out of `resource/`.
    fn temp_workspace(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rust_renderer_skinned_{name}"));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).ok();
        }
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(SOURCE_GLTF, dir.join("Fox.glb")).unwrap();
        dir
    }

    #[test]
    fn imports_skin_and_skinned_primitive() {
        let dir = temp_workspace("import");
        let glb = dir.join("Fox.glb");
        let manager = AssetManager::new();
        manager.import(GltfImporter::new(&glb)).expect("skinned glTF import should succeed");

        let skinned_name = GltfImporter::skinned_primitive_subasset_name(&glb, 0, 0)
            .expect("skinned primitive name should resolve");
        let skin_name = GltfImporter::skin_subasset_name(&glb, 0, 0)
            .expect("skin name should resolve");
        let scene_name = GltfImporter::scene_subasset_name(&glb, 0)
            .expect("scene name should resolve");

        let asset = manager.get_asset(&skinned_name).expect("skinned primitive should be imported");
        let asset = asset.read().unwrap();
        let AssetData::SkinnedMeshAssetType(mesh) = asset.get_data() else {
            panic!("'{skinned_name}' is not a skinned mesh asset");
        };

        assert!(!mesh.animation.is_empty(), "skinning attributes should not be empty");
        assert_eq!(
            mesh.animation.len(),
            mesh.restpose.vp.len(),
            "there must be one skinning entry per vertex"
        );
        assert_eq!(mesh.skin, skin_name, "the skinned mesh must reference the imported skin");

        // The compute shader assumes normalized weights.
        for entry in &mesh.animation {
            let sum: f32 = entry.bone_weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "weights must sum to 1, got {sum}");
        }

        let asset = manager.get_asset(&skin_name).expect("skin should be imported");
        let asset = asset.read().unwrap();
        let AssetData::SkinAssetType(skin) = asset.get_data() else {
            panic!("'{skin_name}' is not a skin asset");
        };
        assert_eq!(skin.joints.len(), JOINT_COUNT);
        assert_eq!(skin.scene, scene_name, "the skin is scoped to its owning scene");
    }

    #[test]
    fn scene_node_references_the_skinned_primitive() {
        let dir = temp_workspace("scene");
        let glb = dir.join("Fox.glb");
        let manager = AssetManager::new();
        manager.import(GltfImporter::new(&glb)).expect("skinned glTF import should succeed");

        let skinned_name = GltfImporter::skinned_primitive_subasset_name(&glb, 0, 0)
            .expect("skinned primitive name should resolve");
        let scene_name = GltfImporter::scene_subasset_name(&glb, 0)
            .expect("scene name should resolve");

        let asset = manager.get_asset(&scene_name).expect("scene should be imported");
        let asset = asset.read().unwrap();
        let AssetData::SceneAssetType(scene) = asset.get_data() else {
            panic!("'{scene_name}' is not a scene asset");
        };

        let skinned_node = scene.nodes.iter()
            .find(|node| node.name.as_deref() == Some("fox"))
            .expect("the Fox scene should contain the 'fox' node");
        assert_eq!(skinned_node.mesh_subasset_names, vec![skinned_name]);
    }

    #[test]
    fn skinned_only_mesh_does_not_emit_a_static_variant() {
        let dir = temp_workspace("no_static");
        let glb = dir.join("Fox.glb");
        let manager = AssetManager::new();
        manager.import(GltfImporter::new(&glb)).expect("skinned glTF import should succeed");

        // The Fox mesh is only ever placed on a skinned node, so a static
        // variant would be an orphan asset.
        assert!(
            manager.get_asset("Fox.glb.prim_0_0").is_none(),
            "a skinned-only mesh must not produce a static primitive asset"
        );
    }

    #[test]
    fn skinned_assets_roundtrip_through_disk() {
        let dir = temp_workspace("roundtrip");
        let glb = dir.join("Fox.glb");
        {
            let manager = AssetManager::new();
            manager.import(GltfImporter::new(&glb)).expect("skinned glTF import should succeed");
        }

        // A fresh manager can only succeed if the new asset kinds serialize.
        let loader = AssetManager::new();
        let report = loader.load_directory(&dir);
        assert_eq!(report.failed, 0, "every persisted asset should deserialize");

        let skinned_name = GltfImporter::skinned_primitive_subasset_name(&glb, 0, 0)
            .expect("skinned primitive name should resolve");
        let asset = loader.get_asset(&skinned_name).expect("skinned primitive should load from disk");
        let asset = asset.read().unwrap();
        let AssetData::SkinnedMeshAssetType(mesh) = asset.get_data() else {
            panic!("'{skinned_name}' is not a skinned mesh asset");
        };
        assert_eq!(mesh.animation.len(), mesh.restpose.vp.len());
    }
}
