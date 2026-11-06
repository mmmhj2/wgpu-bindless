//! Morph target import: asset extraction, the morph-only deformable path and
//! the absence of orphaned static variants.

#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use rust_renderer::asset::{
        asset_manager::AssetManager,
        asset_types::AssetData,
        importer::gltf_importer::GltfImporter,
    };

    /// `resource/AnimatedMorphCube.glb` has one mesh
    /// with one primitive carrying two morph targets. The primitive has no
    /// `JOINTS_0` / `WEIGHTS_0` and no skin, so it is a morph-only mesh.
    const SOURCE_GLTF: &str = "resource/AnimatedMorphCube.glb";
    const VERTEX_COUNT: usize = 24;
    const TARGET_COUNT: usize = 2;

    /// Copy the morph test model into an isolated temp workspace.
    ///
    /// Importers persist assets next to their source, so copying the source
    /// keeps generated `*.asset` files out of `resource/`.
    fn temp_workspace(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rust_renderer_morph_{name}"));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).ok();
        }
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(SOURCE_GLTF, dir.join("AnimatedMorphCube.glb")).unwrap();
        dir
    }

    fn import(dir: &std::path::Path) -> (AssetManager, PathBuf) {
        let glb = dir.join("AnimatedMorphCube.glb");
        let manager = AssetManager::new();
        manager.import(GltfImporter::new(&glb)).expect("morph glTF import should succeed");
        (manager, glb)
    }

    #[test]
    fn imports_morph_targets() {
        let dir = temp_workspace("targets");
        let (manager, glb) = import(&dir);

        let target_name = GltfImporter::morph_target_subasset_name(&glb, 0, 0)
            .expect("morph target name should resolve");
        assert!(
            target_name.ends_with(".morphtgt_0_0"),
            "morph targets must follow the `.morphtgt_<mesh>_<prim>` naming, got '{target_name}'"
        );

        let asset = manager.get_asset(&target_name).expect("morph target should be imported");
        let asset = asset.read().unwrap();
        let AssetData::VertexMorphTargetAssetType(target) = asset.get_data() else {
            panic!("'{target_name}' is not a morph target asset");
        };

        assert_eq!(target.targets.len(), TARGET_COUNT, "the cube defines two morph targets");

        for (index, morph) in target.targets.iter().enumerate() {
            let position = morph.position.as_ref()
                .unwrap_or_else(|| panic!("target {index} should have position deltas"));
            let normal = morph.normal.as_ref()
                .unwrap_or_else(|| panic!("target {index} should have normal deltas"));
            let tangent = morph.tangent.as_ref()
                .unwrap_or_else(|| panic!("target {index} should have tangent deltas"));

            // TEXCOORD_n / COLOR_n deltas are dropped by the `gltf` crate.
            assert!(morph.texcoord0.is_none(), "target {index} must not claim UV deltas");
            assert!(morph.color0.is_none(), "target {index} must not claim color deltas");

            assert_eq!(position.len(), VERTEX_COUNT, "target {index} position delta count");
            assert_eq!(normal.len(), VERTEX_COUNT, "target {index} normal delta count");
            assert_eq!(tangent.len(), VERTEX_COUNT, "target {index} tangent delta count");

            for (offset, delta) in position.iter().enumerate() {
                assert!(
                    delta.iter().all(|channel| channel.is_finite()),
                    "target {index} vertex {offset} has a non-finite position delta: {delta:?}"
                );
            }
        }

        // The two targets must come from different accessors, not the same block
        // read twice.
        let first = target.targets[0].position.as_ref().unwrap();
        let second = target.targets[1].position.as_ref().unwrap();
        assert_ne!(first, second, "the two morph targets must not share their deltas");
        assert!(
            first.iter().flatten().any(|channel| *channel != 0.0),
            "the first target's position deltas are all zero"
        );

        // The target belongs to the deformable mesh asset, not to a static one.
        let skinned_name = GltfImporter::skinned_primitive_subasset_name(&glb, 0, 0)
            .expect("skinned primitive name should resolve");
        assert_eq!(target.mesh, skinned_name, "the targets must reference their mesh asset");
    }

    #[test]
    fn morph_only_primitive_uses_the_deformable_asset() {
        let dir = temp_workspace("mesh");
        let (manager, glb) = import(&dir);

        let skinned_name = GltfImporter::skinned_primitive_subasset_name(&glb, 0, 0)
            .expect("skinned primitive name should resolve");
        let target_name = GltfImporter::morph_target_subasset_name(&glb, 0, 0)
            .expect("morph target name should resolve");

        let asset = manager.get_asset(&skinned_name).expect("deformable primitive should be imported");
        let asset = asset.read().unwrap();
        let AssetData::SkinnedMeshAssetType(mesh) = asset.get_data() else {
            panic!("'{skinned_name}' is not a deformable mesh asset");
        };

        assert_eq!(mesh.skin, None, "the cube has no skin, so the asset must not claim one");
        assert_eq!(
            mesh.morph_target.as_deref(),
            Some(target_name.as_str()),
            "the mesh must reference its morph target asset"
        );
        assert!(
            mesh.skinning.is_none(),
            "a morph-only mesh must not synthesize skinning attributes"
        );
        assert_eq!(mesh.restpose.vp.len(), VERTEX_COUNT);
    }

    #[test]
    fn morph_only_primitive_does_not_emit_a_static_variant() {
        let dir = temp_workspace("no_static");
        let (manager, _glb) = import(&dir);

        // The primitive is only ever referenced through its deformable asset, so
        // a static variant would be an orphan.
        assert!(
            manager.get_asset("AnimatedMorphCube.glb.prim_0_0").is_none(),
            "a morph-only primitive must not produce a static primitive asset"
        );
    }

    #[test]
    fn scene_node_references_the_deformable_primitive() {
        let dir = temp_workspace("scene");
        let (manager, glb) = import(&dir);

        let skinned_name = GltfImporter::skinned_primitive_subasset_name(&glb, 0, 0)
            .expect("skinned primitive name should resolve");
        let scene_name = GltfImporter::scene_subasset_name(&glb, 0)
            .expect("scene name should resolve");

        let asset = manager.get_asset(&scene_name).expect("scene should be imported");
        let asset = asset.read().unwrap();
        let AssetData::SceneAssetType(scene) = asset.get_data() else {
            panic!("'{scene_name}' is not a scene asset");
        };

        let node = scene.nodes.iter()
            .find(|node| node.name.as_deref() == Some("AnimatedMorphCube"))
            .expect("the scene should contain the 'AnimatedMorphCube' node");
        assert_eq!(
            node.mesh_subasset_names,
            vec![skinned_name],
            "the node must reference the deformable variant of its primitive"
        );
    }

    #[test]
    fn morph_assets_roundtrip_through_disk() {
        let dir = temp_workspace("roundtrip");
        import(&dir);

        // A fresh manager can only succeed if the new asset kind serializes.
        let loader = AssetManager::new();
        let report = loader.load_directory(&dir);
        assert_eq!(report.failed, 0, "every persisted asset should deserialize");

        let glb = dir.join("AnimatedMorphCube.glb");
        let target_name = GltfImporter::morph_target_subasset_name(&glb, 0, 0)
            .expect("morph target name should resolve");
        let asset = loader.get_asset(&target_name).expect("morph target should load from disk");
        let asset = asset.read().unwrap();
        let AssetData::VertexMorphTargetAssetType(target) = asset.get_data() else {
            panic!("'{target_name}' is not a morph target asset");
        };
        assert_eq!(target.targets.len(), TARGET_COUNT);
        assert_eq!(
            target.targets[0].position.as_ref().map(Vec::len),
            Some(VERTEX_COUNT),
            "dense position deltas must survive serialization"
        );
    }
}
