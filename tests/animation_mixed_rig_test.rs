//! Headless GPU tests for the parts of the deformation path that the single-rig
//! morph test cannot reach:
//!
//! - a rig with both stages, where a joints-only clip makes morphing
//!   neutral and a weights-only clip makes skinning neutral;
//! - a static `mesh.weights` base shape used as the neutral morph input when the
//!   player has no clip at all;
//! - two instances of one rig sharing one clip catalog.
//!
//! The fixtures are hand-built assets saved through the normal container path,
//! so the tests exercise `GpuAssetStore::prepare_scene` exactly like a scene
//! loaded from disk.

#[cfg(any(target_os = "windows", target_os = "linux"))]
mod common;

#[cfg(test)]
#[cfg(any(target_os = "windows", target_os = "linux"))]
mod test {
    use std::{
        num::NonZero,
        path::{Path, PathBuf},
    };

    use rust_renderer::{
        asset::{
            asset_manager::AssetManager,
            asset_types::{
                animation_asset::{
                    MorphTarget, SkinnedAnimationAsset, VertexMorphAnimationAsset, VertexMorphTargetAsset,
                },
                scene_asset::{SceneAsset, SceneNodeAsset},
                skin_asset::{SkinAsset, SkinJoint},
                deformable_mesh_asset::DeformableMeshAsset,
                static_mesh_asset::{StaticMeshAsset, StaticMeshMaterial},
                texture_asset::{Sampler, TexelData, TextureAsset, TextureDimension, TextureFormat},
                vertex::{VertexBufferOthers, VertexBufferPosition, VertexBufferSkinning},
                Asset, AssetData, AssetMetadata,
            },
        },
        renderer::{
            context::gpu_context::GpuContext,
            drawable::{
                draw_list::{DrawList, DrawableType},
                drawable_mesh_traits::Mesh,
            },
            pipeline::compute_library::ComputePipelineLibrary,
            resources::{
                bindless::BindlessResources,
                gpu_asset_store::{GpuAssetStore, SceneUploadContext},
                mesh_layout::MeshLayout,
                rig_plan::{FinalBuffer, RigStages},
            },
            upload::UploadContext,
        },
    };
    use crate::common;

    fn identity() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// Unique temp directory per test: these tests run in parallel.
    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rust_renderer_mixed_rig_{name}"));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).expect("failed to clear the previous fixture");
        }
        std::fs::create_dir_all(&dir).expect("failed to create the fixture directory");
        dir
    }

    fn save_asset(dir: &Path, name: &str, data: AssetData) {
        let path = dir.join(format!("{name}.asset"));
        let asset = Asset::new(
            AssetMetadata::new(name.to_string(), Some(path.clone().into_boxed_path())),
            data,
        );
        asset.save_to_disk().expect("fixture asset should save");
    }

    fn load_assets(dir: &Path) -> AssetManager {
        let manager = AssetManager::new();
        for entry in std::fs::read_dir(dir).expect("fixture directory should be readable") {
            let path = entry.expect("fixture entry").path();
            manager
                .load_asset_file(&path)
                .unwrap_or_else(|error| panic!("fixture asset '{}' should load: {error}", path.display()));
        }
        manager
    }

    /// Build an RGBA16F texture from rows of target weights.
    fn weight_texture(width: u32, rows: &[Vec<f32>]) -> TextureAsset {
        let mut texels = Vec::with_capacity(rows.len() * width as usize * 8);
        for row in rows {
            for x in 0..width as usize {
                for lane in 0..4 {
                    let value = row.get(x * 4 + lane).copied().unwrap_or(0.0);
                    texels.extend_from_slice(&common::f16_bits(value).to_le_bytes());
                }
            }
        }

        TextureAsset {
            format: TextureFormat::Rgba16Float,
            dimension: TextureDimension::D2,
            width: NonZero::new(width).expect("a texture needs a width"),
            height: Some(NonZero::new(rows.len() as u32).expect("a texture needs a height")),
            depth_or_array_slice: None,
            sampler: Sampler::default(),
            texels: TexelData::Plain(texels.into_boxed_slice()),
        }
    }

    /// One joint whose matrix translates by (1, 0, 0), with the wrap row that
    /// the skinning shader always reads as `floor(frame) + 1`.
    fn joint_texture() -> TextureAsset {
        let columns = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ];
        let row: Vec<f32> = columns.iter().flat_map(|column| column.iter().copied()).collect();
        weight_texture(4, &[row.clone(), row])
    }

    fn restpose(position: [f32; 3]) -> StaticMeshAsset {
        StaticMeshAsset {
            vp: vec![VertexBufferPosition { position }],
            va: vec![VertexBufferOthers {
                color: [1.0; 4],
                normal: [0.0, 1.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                uv0: [0.0, 0.0],
            }],
            vi: None,
            material: StaticMeshMaterial {
                diffuse_texture_name: None,
                normal_texture_name: None,
                mrao_texture_name: None,
            },
            vertex_draw_count: 1,
            vertex_color_scale: [1.0; 4],
        }
    }

    fn scene(mesh_names: &[&str]) -> SceneAsset {
        SceneAsset {
            nodes: mesh_names
                .iter()
                .enumerate()
                .map(|(index, mesh)| SceneNodeAsset {
                    name: Some(format!("node{index}")),
                    local_matrix: identity(),
                    mesh_subasset_names: vec![(*mesh).to_string()],
                    children: Vec::new(),
                })
                .collect(),
            roots: (0..mesh_names.len() as u32).collect(),
        }
    }

    /// A rig with a skin, a morph target, one joints-only clip and one
    /// weights-only clip.
    fn save_mixed_fixture(dir: &Path) {
        save_asset(
            dir,
            "mixed.skin",
            AssetData::SkinAssetType(SkinAsset {
                scene: "mixed.scene".into(),
                joints: vec![SkinJoint {
                    scene_node: 0,
                    inverse_bind_matrix: identity(),
                }],
            }),
        );
        save_asset(
            dir,
            "mixed.morphtgt",
            AssetData::VertexMorphTargetAssetType(VertexMorphTargetAsset {
                mesh: "mixed.skpr".into(),
                targets: vec![MorphTarget {
                    position: Some(vec![[0.5, 0.0, 0.0]]),
                    ..Default::default()
                }],
                default_weights: None,
            }),
        );
        save_asset(
            dir,
            "mixed.skpr",
            AssetData::SkinnedMeshAssetType(DeformableMeshAsset {
                skin: Some("mixed.skin".into()),
                morph_target: Some("mixed.morphtgt".into()),
                restpose: restpose([0.0, 0.0, 0.0]),
                skinning: Some(vec![VertexBufferSkinning {
                    bone_indices: [0; 4],
                    bone_weights: [1.0, 0.0, 0.0, 0.0],
                }]),
            }),
        );
        save_asset(
            dir,
            "mixed.anim",
            AssetData::SkinnedAnimationAssetType(SkinnedAnimationAsset {
                skin: "mixed.skin".into(),
                source: "scene0/anim0".into(),
                frames: 1,
                fps: 30.0,
                texture: joint_texture(),
            }),
        );
        save_asset(
            dir,
            "mixed.morphanim",
            AssetData::VertexMorphAnimationAssetType(VertexMorphAnimationAsset {
                mesh: "mixed.skpr".into(),
                target: "mixed.morphtgt".into(),
                source: "scene0/anim1".into(),
                frames: 1,
                fps: 30.0,
                texture: weight_texture(1, &[vec![1.0], vec![1.0]]),
            }),
        );
        save_asset(dir, "mixed.scene", AssetData::SceneAssetType(scene(&["mixed.skpr"])));
    }

    /// A morph-only rig whose neutral shape comes from `mesh.weights`, with no
    /// clip at all.
    fn save_base_weight_fixture(dir: &Path) {
        save_asset(
            dir,
            "base.morphtgt",
            AssetData::VertexMorphTargetAssetType(VertexMorphTargetAsset {
                mesh: "base.skpr".into(),
                targets: vec![MorphTarget {
                    position: Some(vec![[1.0, 0.0, 0.0]]),
                    ..Default::default()
                }],
                default_weights: Some(vec![0.25]),
            }),
        );
        save_asset(
            dir,
            "base.skpr",
            AssetData::SkinnedMeshAssetType(DeformableMeshAsset {
                skin: None,
                morph_target: Some("base.morphtgt".into()),
                restpose: restpose([0.0, 0.0, 0.0]),
                skinning: None,
            }),
        );
        save_asset(dir, "base.scene", AssetData::SceneAssetType(scene(&["base.skpr"])));
    }

    fn setup() -> (GpuContext, BindlessResources, ComputePipelineLibrary) {
        let mut gpu = GpuContext::new();
        pollster::block_on(gpu.create_device(None)).expect("device creation should succeed");

        let bindless = BindlessResources::new(gpu.device(), gpu.queue());
        let compute = ComputePipelineLibrary::new(gpu.device(), bindless.get_bind_group_layout());
        (gpu, bindless, compute)
    }

    fn prepare(
        gpu: &GpuContext,
        bindless: &mut BindlessResources,
        manager: &AssetManager,
        scene_name: &str,
    ) -> (GpuAssetStore, DrawList) {
        let mut store = GpuAssetStore::new();

        let draw_list = {
            let asset = manager.get_asset(scene_name).expect("the fixture scene should exist");
            let asset = asset.read().expect("the fixture scene should be readable");
            let AssetData::SceneAssetType(scene) = asset.get_data() else {
                panic!("'{scene_name}' is not a scene asset");
            };
            let mut ctx = SceneUploadContext::new(
                UploadContext::new(gpu.device(), gpu.queue(), bindless),
                manager,
            );
            store
                .prepare_scene(&mut ctx, scene)
                .expect("the fixture scene should prepare")
        };

        (store, draw_list)
    }

    fn ensure_bound(draw_list: &mut DrawList, gpu: &GpuContext, compute: &ComputePipelineLibrary) {
        let mesh_layout = MeshLayout::new(gpu.device());
        draw_list
            .ensure_bound(gpu.device(), &mesh_layout)
            .ensure_deformation_bound(gpu.device(), compute);
    }

    fn dispatch(
        draw_list: &DrawList,
        gpu: &GpuContext,
        bindless: &BindlessResources,
        compute: &ComputePipelineLibrary,
    ) {
        let bindless_bg = bindless.get_bind_group(gpu.device());
        let mut encoder = gpu.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mixed rig deformation encoder"),
        });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mixed rig deformation"),
                timestamp_writes: None,
            });
            cp.set_bind_group(0, Some(&bindless_bg), &[]);
            draw_list.record_deformation(&mut cp, compute);
        }
        gpu.queue().submit(std::iter::once(encoder.finish()));
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll should succeed");
    }

    fn read_positions(gpu: &GpuContext, buffer: &wgpu::Buffer) -> Vec<[f32; 3]> {
        let staging = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("mixed rig readback"),
            size: buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mixed rig readback encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
        gpu.queue().submit(std::iter::once(encoder.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |result| {
            result.expect("readback mapping should succeed");
        });
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll should succeed");

        staging
            .slice(..)
            .get_mapped_range()
            .chunks_exact(std::mem::size_of::<VertexBufferPosition>())
            .map(bytemuck::pod_read_unaligned::<VertexBufferPosition>)
            .map(|position| position.position)
            .collect()
    }

    fn assert_position(actual: [f32; 3], expected: [f32; 3], label: &str) {
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() < 1e-4,
                "{label}: expected {expected:?}, got {actual:?}"
            );
        }
    }

    #[test]
    fn mixed_rig_neutralizes_the_missing_track() {
        let dir = temp_dir("neutral");
        save_mixed_fixture(&dir);
        let manager = load_assets(&dir);

        let (gpu, mut bindless, compute) = setup();
        let (_store, mut draw_list) = prepare(&gpu, &mut bindless, &manager, "mixed.scene");

        assert_eq!(draw_list.count(DrawableType::OpaqueSkinned), 1);
        {
            let mesh = &draw_list.get_skinned_meshes()[0];
            assert_eq!(mesh.stages(), RigStages::MorphThenSkin);
            assert_eq!(mesh.final_buffer(), FinalBuffer::SkinOut);
            assert_eq!(mesh.deformed_buffer_count(), 4, "morph_out + skin_out");
        }
        assert_eq!(draw_list.animations().clips().len(), 2, "two mixed-catalog clips");

        let player = draw_list.get_skinned_meshes()[0].player();
        let default_clip = draw_list.animations().player(player).clip().expect("a default clip");
        assert_eq!(
            draw_list.animations().clip(default_clip).name(),
            "mixed.anim",
            "the joints-only clip sorts first"
        );

        ensure_bound(&mut draw_list, &gpu, &compute);

        let rest_buffer = draw_list.get_skinned_meshes()[0].restpose_vertex_buffers()[0].clone();
        let final_buffer = draw_list.get_skinned_meshes()[0].get_vertex_buffer()[0].clone();
        let rest = read_positions(&gpu, &rest_buffer);
        assert_position(rest[0], [0.0, 0.0, 0.0], "rest pose");

        // Clip A drives joints only. Morph must still run with `channels = 0`
        // and copy the rest pose, then skinning applies the +1 X translation.
        dispatch(&draw_list, &gpu, &bindless, &compute);
        let deformed = read_positions(&gpu, &final_buffer);
        assert_position(
            deformed[0],
            [1.0, 0.0, 0.0],
            "joints-only clip: neutral morph + real skinning",
        );

        // Clip B drives weights only. Morph applies +0.5 X, then skinning takes
        // the identity path and copies its input through.
        assert!(
            draw_list.animations_mut().play_by_name(player, "mixed.morphanim"),
            "the weights-only clip should be registered for the rig"
        );
        dispatch(&draw_list, &gpu, &bindless, &compute);
        let deformed = read_positions(&gpu, &final_buffer);
        assert_position(
            deformed[0],
            [0.5, 0.0, 0.0],
            "weights-only clip: real morphing + identity skinning",
        );
    }

    #[test]
    fn base_weights_are_used_without_a_clip() {
        let dir = temp_dir("base");
        save_base_weight_fixture(&dir);
        let manager = load_assets(&dir);

        let (gpu, mut bindless, compute) = setup();
        let (_store, mut draw_list) = prepare(&gpu, &mut bindless, &manager, "base.scene");

        {
            let mesh = &draw_list.get_skinned_meshes()[0];
            assert_eq!(mesh.stages(), RigStages::Morph);
            assert_eq!(mesh.final_buffer(), FinalBuffer::MorphOut);
            assert_eq!(mesh.deformed_buffer_count(), 2);
        }
        let player = draw_list.get_skinned_meshes()[0].player();
        assert_eq!(draw_list.animations().player_count(), 1);
        assert_eq!(draw_list.animations().clips().len(), 0);
        assert!(
            draw_list.animations().frame(player).is_none(),
            "a player without a clip must report no tracks"
        );

        ensure_bound(&mut draw_list, &gpu, &compute);

        let final_buffer = draw_list.get_skinned_meshes()[0].get_vertex_buffer()[0].clone();
        dispatch(&draw_list, &gpu, &bindless, &compute);
        let deformed = read_positions(&gpu, &final_buffer);
        assert_position(
            deformed[0],
            [0.25, 0.0, 0.0],
            "static base weights are the neutral morph input",
        );
    }

    #[test]
    fn instances_share_one_clip_catalog() {
        let dir = temp_dir("shared");
        save_mixed_fixture(&dir);
        // Overwrite the fixture scene with two nodes referencing the same mesh.
        save_asset(
            &dir,
            "mixed.scene",
            AssetData::SceneAssetType(scene(&["mixed.skpr", "mixed.skpr"])),
        );
        let manager = load_assets(&dir);

        let (gpu, mut bindless, _compute) = setup();
        let (store, draw_list) = prepare(&gpu, &mut bindless, &manager, "mixed.scene");

        assert_eq!(draw_list.count(DrawableType::OpaqueSkinned), 2);
        assert_eq!(draw_list.animations().player_count(), 2);
        assert_eq!(
            draw_list.animations().clips().len(),
            2,
            "two unique clips, not one copy per instance"
        );
        let first = draw_list.get_skinned_meshes()[0].player();
        let second = draw_list.get_skinned_meshes()[1].player();
        assert_eq!(
            draw_list.animations().player(first).clip(),
            draw_list.animations().player(second).clip(),
            "both instances start on the same clip id"
        );
        assert_eq!(store.skinned_mesh_count(), 1, "the rest pose is shared");
        assert_eq!(store.morph_animation_count(), 1, "the weight track is shared");
        assert_eq!(store.morph_target_count(), 1, "the target is shared");
    }
}
