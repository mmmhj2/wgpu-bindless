#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use rust_renderer::asset::{
        asset_manager::AssetManager,
        asset_types::{AssetData, texture_asset::TextureFormat},
        importer::gltf_importer::GltfImporter,
    };

    /// `resource/Fox.glb` (glTF-Sample-Assets, CC0) has one skinned primitive,
    /// one skin with 24 joints and three animations.
    const SOURCE_GLTF: &str = "resource/Fox.glb";
    const JOINT_COUNT: usize = 24;
    const ANIMATION_COUNT: usize = 3;

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

    #[test]
    fn imports_baked_animations() {
        let dir = temp_workspace("animations");
        let glb = dir.join("Fox.glb");
        let manager = AssetManager::new();
        manager.import(GltfImporter::new(&glb)).expect("skinned glTF import should succeed");

        let skin_name = GltfImporter::skin_subasset_name(&glb, 0, 0)
            .expect("skin name should resolve");

        for animation_index in 0..ANIMATION_COUNT {
            let name = GltfImporter::animation_subasset_name(&glb, 0, animation_index)
                .expect("animation name should resolve");
            let asset = manager.get_asset(&name)
                .unwrap_or_else(|| panic!("animation '{name}' should be imported"));
            let asset = asset.read().unwrap();
            let AssetData::SkinnedAnimationAssetType(animation) = asset.get_data() else {
                panic!("'{name}' is not a skinned animation asset");
            };

            assert_eq!(animation.skin, skin_name);
            assert_eq!(animation.fps, 30.0);
            assert!(animation.frames > 1, "'{name}' should span more than one frame");

            // Four texels per joint, one row per frame, `Rgba16Float`.
            assert_eq!(animation.texture.format, TextureFormat::Rgba16Float);
            assert_eq!(animation.texture.width.get(), JOINT_COUNT as u32 * 4);
            assert_eq!(animation.texture.height.unwrap().get(), animation.frames);

            let texels = match &animation.texture.texels {
                rust_renderer::asset::asset_types::texture_asset::TexelData::Plain(texels) => texels,
                _ => panic!("baked animation must be a plain texture"),
            };
            let frame_bytes = animation.texture.width.get() as usize * 8;
            assert_eq!(texels.len(), frame_bytes * animation.frames as usize);

            // The bake must actually vary over time. Looping animations repeat
            // the first pose at the end, so compare against the middle frame.
            let middle_frame = animation.frames as usize / 2;
            let first = &texels[..frame_bytes];
            let middle = &texels[middle_frame * frame_bytes..(middle_frame + 1) * frame_bytes];
            assert_ne!(first, middle, "'{name}' looks static around the middle frame");
        }
    }

    /// Decode one IEEE-754 binary16 value.
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = ((bits >> 10) & 0x1F) as i32;
        let mantissa = (bits & 0x3FF) as f32;

        let magnitude = if exponent == 0 {
            mantissa * 2f32.powi(-24)
        } else if exponent == 0x1F {
            if mantissa == 0.0 { f32::INFINITY } else { f32::NAN }
        } else {
            (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15)
        };
        sign * magnitude
    }

    /// Read one texel of the baked animation texture as four floats.
    fn texel_at(texels: &[u8], width: usize, x: usize, y: usize) -> [f32; 4] {
        let offset = (y * width + x) * 8;
        let mut value = [0.0f32; 4];
        for (channel, slot) in value.iter_mut().enumerate() {
            let bits = u16::from_le_bytes([texels[offset + channel * 2], texels[offset + channel * 2 + 1]]);
            *slot = f16_to_f32(bits);
        }
        value
    }

    /// Read one column-major joint matrix of the given frame.
    fn joint_matrix(texels: &[u8], width: usize, joint: usize, frame: usize) -> [[f32; 4]; 4] {
        let mut matrix = [[0.0f32; 4]; 4];
        for (column, slot) in matrix.iter_mut().enumerate() {
            *slot = texel_at(texels, width, joint * 4 + column, frame);
        }
        matrix
    }

    fn transform(matrix: &[[f32; 4]; 4], point: [f32; 3]) -> [f32; 3] {
        let vector = [point[0], point[1], point[2], 1.0];
        let mut out = [0.0f32; 3];
        for (row, slot) in out.iter_mut().enumerate() {
            *slot = (0..4).map(|column| matrix[column][row] * vector[column]).sum();
        }
        out
    }

    #[test]
    fn baked_matrices_deform_the_rest_pose_plausibly() {
        let dir = temp_workspace("deform");
        let glb = dir.join("Fox.glb");
        let manager = AssetManager::new();
        manager.import(GltfImporter::new(&glb)).expect("skinned glTF import should succeed");

        let skinned_name = GltfImporter::skinned_primitive_subasset_name(&glb, 0, 0)
            .expect("skinned primitive name should resolve");
        let animation_name = GltfImporter::animation_subasset_name(&glb, 0, 0)
            .expect("animation name should resolve");

        let skinned = manager.get_asset(&skinned_name).expect("skinned mesh should be imported");
        let skinned = skinned.read().unwrap();
        let AssetData::SkinnedMeshAssetType(mesh) = skinned.get_data() else {
            panic!("'{skinned_name}' is not a skinned mesh asset");
        };

        let animation = manager.get_asset(&animation_name).expect("animation should be imported");
        let animation = animation.read().unwrap();
        let AssetData::SkinnedAnimationAssetType(animation) = animation.get_data() else {
            panic!("'{animation_name}' is not a skinned animation asset");
        };
        let texels = match &animation.texture.texels {
            rust_renderer::asset::asset_types::texture_asset::TexelData::Plain(texels) => texels,
            _ => panic!("baked animation must be a plain texture"),
        };
        let width = animation.texture.width.get() as usize;

        // Skin the rest pose with frame 0 and compare the two bounding boxes.
        // A transposed matrix, a wrong texture layout or a wrong space would
        // throw vertices far outside the bind pose.
        let mut rest_bounds = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        let mut posed_bounds = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);

        for (vertex, skinning) in mesh.restpose.vp.iter().zip(&mesh.animation) {
            let rest = vertex.position;
            let mut posed = [0.0f32; 3];
            for influence in 0..4 {
                let weight = skinning.bone_weights[influence];
                if weight <= 0.0 {
                    continue;
                }
                let matrix = joint_matrix(texels, width, skinning.bone_indices[influence] as usize, 0);
                let transformed = transform(&matrix, rest);
                for axis in 0..3 {
                    posed[axis] += weight * transformed[axis];
                }
            }

            for axis in 0..3 {
                rest_bounds.0[axis] = rest_bounds.0[axis].min(rest[axis]);
                rest_bounds.1[axis] = rest_bounds.1[axis].max(rest[axis]);
                posed_bounds.0[axis] = posed_bounds.0[axis].min(posed[axis]);
                posed_bounds.1[axis] = posed_bounds.1[axis].max(posed[axis]);
            }
        }

        for axis in 0..3 {
            let rest_extent = rest_bounds.1[axis] - rest_bounds.0[axis];
            let posed_extent = posed_bounds.1[axis] - posed_bounds.0[axis];
            let rest_center = (rest_bounds.0[axis] + rest_bounds.1[axis]) * 0.5;
            let posed_center = (posed_bounds.0[axis] + posed_bounds.1[axis]) * 0.5;

            assert!(
                posed_extent <= rest_extent * 3.0 + 1.0,
                "axis {axis}: posed extent {posed_extent} is far larger than the rest extent {rest_extent}"
            );
            assert!(
                (posed_center - rest_center).abs() <= rest_extent * 2.0 + 1.0,
                "axis {axis}: posed center {posed_center} drifted away from the rest center {rest_center}"
            );
        }
    }
}
