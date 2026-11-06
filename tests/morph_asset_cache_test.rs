//! Headless GPU tests for the morph target cache: deduplication across instances,
//! the layout/delta upload shape and the base-weight allocation policy.

#[cfg(test)]
#[cfg(any(target_os = "windows", target_os = "linux"))]
mod test {
    use std::{num::NonZeroU32, path::PathBuf};

    use rust_renderer::{
        asset::{
            asset_manager::AssetManager,
            asset_types::{
                animation_asset::{MorphTarget, VertexMorphTargetAsset},
                scene_asset::{SceneAsset, SceneNodeAsset},
            },
            importer::gltf_importer::GltfImporter,
        },
        renderer::{
            context::gpu_context::GpuContext,
            drawable::draw_list::DrawableType,
            resources::{
                asset_cache::{GpuAssetStore, SceneUploadContext},
                bindless::BindlessResources,
            },
            upload::{GpuAssetType, UploadContext, UploadError, MAX_MORPH_TARGETS},
        },
    };

    /// `resource/AnimatedMorphCube.glb` has one morph-only primitive with two
    /// targets and `mesh.weights = [0, 0]`.
    const SOURCE_GLTF: &str = "resource/AnimatedMorphCube.glb";

    /// TODO: extract this into a new common.rs function
    fn temp_workspace() -> PathBuf {
        let dir = std::env::temp_dir().join("rust_renderer_morph_cache");
        if dir.exists() {
            std::fs::remove_dir_all(&dir).expect("failed to remove old temp dir");
        }
        std::fs::create_dir(&dir).expect("failed to create new temp dir");
        std::fs::copy(SOURCE_GLTF, dir.join("AnimatedMorphCube.glb")).unwrap();
        dir
    }

    fn setup() -> (GpuContext, BindlessResources) {
        let mut gpu = GpuContext::new();
        pollster::block_on(gpu.create_device(None)).expect("device creation should succeed");
        let bindless = BindlessResources::new(gpu.device(), gpu.queue());
        (gpu, bindless)
    }

    /// A scene with two instances of the same mesh asset.
    fn two_instance_scene(mesh_name: &str) -> SceneAsset {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let translated = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [5.0, 0.0, 0.0, 1.0],
        ];

        SceneAsset {
            nodes: vec![
                SceneNodeAsset {
                    name: Some("instance_a".into()),
                    local_matrix: identity,
                    mesh_subasset_names: vec![mesh_name.to_string()],
                    children: vec![],
                },
                SceneNodeAsset {
                    name: Some("instance_b".into()),
                    local_matrix: translated,
                    mesh_subasset_names: vec![mesh_name.to_string()],
                    children: vec![],
                },
            ],
            roots: vec![0, 1],
        }
    }

    fn manual_target(default_weights: Option<Vec<f32>>) -> VertexMorphTargetAsset {
        let vertices = vec![[0.0f32, 0.0, 0.0]; 3];
        VertexMorphTargetAsset {
            mesh: "test.skpr".into(),
            targets: vec![
                MorphTarget { position: Some(vertices.clone()), ..Default::default() },
                MorphTarget { position: Some(vertices), ..Default::default() },
            ],
            default_weights,
        }
    }

    #[test]
    fn shared_morph_target_is_uploaded_once() {
        let dir = temp_workspace();
        let glb = dir.join("AnimatedMorphCube.glb");
        let assets = AssetManager::new();
        assets.import(GltfImporter::new(&glb)).expect("morph glTF import should succeed");

        let mesh_name = GltfImporter::skinned_primitive_subasset_name(&glb, 0, 0)
            .expect("mesh name should resolve");
        let target_name = GltfImporter::morph_target_subasset_name(&glb, 0, 0)
            .expect("morph target name should resolve");

        let (gpu, mut bindless) = setup();
        let mut store = GpuAssetStore::new();
        let scene = two_instance_scene(&mesh_name);

        let draw_list = {
            let mut ctx = SceneUploadContext::new(
                UploadContext::new(gpu.device(), gpu.queue(), &mut bindless),
                &assets,
            );
            store.prepare_scene(&mut ctx, &scene).expect("scene preparation should succeed")
        };

        assert_eq!(draw_list.count(DrawableType::OpaqueSkinned), 2, "both instances are drawn");
        assert_eq!(store.skinned_mesh_count(), 1, "the rest pose is shared between instances");
        assert_eq!(
            store.morph_target_count(),
            1,
            "two instances of one mesh must upload its morph target once"
        );
        assert_eq!(store.morph_animation_count(), 1, "the weight track is shared too");

        let players = draw_list.animations();
        assert_eq!(players.player_count(), 2, "one player per instance");
        assert_eq!(
            players.player(0).rig(),
            players.player(1).rig(),
            "both instances play the same rig"
        );

        let target = store.morph_target(&target_name).expect("morph target should be uploaded");
        assert!(target.deltas.size() >= 4, "the delta buffer is never zero-sized");
        assert_eq!(target.channel_count, 2, "the cube has two morph targets");

        // The cube's `mesh.weights` is all zero, so no base texture is allocated.
        assert!(target.base_weights.is_none(), "zero base weights need no texture");
        assert_eq!(target.base_channels, None);
    }

    #[test]
    fn non_zero_base_weights_allocate_a_texture() {
        let (gpu, mut bindless) = setup();
        let mut upload = UploadContext::new(gpu.device(), gpu.queue(), &mut bindless);

        let target = manual_target(Some(vec![0.25, 0.75]));
        let uploaded = target.upload("test.morphtgt", &mut upload).expect("upload should succeed");

        assert_eq!(uploaded.channel_count, 2);
        assert_eq!(uploaded.base_channels, Some(NonZeroU32::new(2).unwrap()), "both targets have a base weight");
        assert!(
            uploaded.base_weights.is_some(),
            "non-zero base weights must allocate the one-row texture"
        );
        assert_eq!(uploaded.layout.size(), 196, "49 u32 words");
    }

    #[test]
    fn zero_or_absent_base_weights_allocate_nothing() {
        let (gpu, mut bindless) = setup();

        for (label, weights) in [
            ("all zero", Some(vec![0.0, 0.0])),
            ("absent", None),
        ] {
            let mut upload = UploadContext::new(gpu.device(), gpu.queue(), &mut bindless);
            let uploaded = manual_target(weights)
                .upload("test.morphtgt", &mut upload)
                .expect("upload should succeed");

            assert!(uploaded.base_weights.is_none(), "{label}: no texture expected");
            assert_eq!(uploaded.base_channels, None, "{label}: no base channels expected");
            assert_eq!(uploaded.channel_count, 2, "{label}: the delta layout still exists");
        }
    }

    #[test]
    fn layout_packs_sparse_targets() {
        let (gpu, mut bindless) = setup();
        let mut upload = UploadContext::new(gpu.device(), gpu.queue(), &mut bindless);

        // Target 0 carries positions, target 1 carries normals: each gets its own
        // block and its own mask bit.
        let target = VertexMorphTargetAsset {
            mesh: "test.skpr".into(),
            targets: vec![
                MorphTarget { position: Some(vec![[1.0, 2.0, 3.0]; 2]), ..Default::default() },
                MorphTarget { normal: Some(vec![[0.0, 0.0, 1.0]; 2]), ..Default::default() },
            ],
            default_weights: None,
        };
        let uploaded = target.upload("test.sparse", &mut upload).expect("upload should succeed");

        // Two position blocks (2 vertices * 3 floats) plus one normal block of the
        // same size: sparse storage only pays for what exists.
        assert_eq!(uploaded.deltas.size(), 6 * 4 * 2, "two blocks of six floats");
    }

    fn target_with_count(count: usize) -> VertexMorphTargetAsset {
        let vertex = vec![[0.0f32, 0.0, 0.0]];
        VertexMorphTargetAsset {
            mesh: "test.skpr".into(),
            targets: (0..count)
                .map(|_| MorphTarget { position: Some(vertex.clone()), ..Default::default() })
                .collect(),
            default_weights: None,
        }
    }

    #[test]
    fn rejects_target_counts_outside_the_supported_range() {
        let (gpu, mut bindless) = setup();
        let mut upload = UploadContext::new(gpu.device(), gpu.queue(), &mut bindless);

        for count in [0, MAX_MORPH_TARGETS + 1] {
            let error = target_with_count(count)
                .upload("test.invalid", &mut upload)
                .expect_err("an out-of-range target count must not be uploaded");
            match error {
                UploadError::MorphTargetCount { count: actual, max, .. } => {
                    assert_eq!(actual, count);
                    assert_eq!(max, MAX_MORPH_TARGETS);
                }
                other => panic!("expected MorphTargetCount for {count} targets, got {other:?}"),
            }
        }
    }
}
