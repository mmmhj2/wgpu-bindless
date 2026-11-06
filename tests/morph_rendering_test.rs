//! Headless end-to-end test for the vertex morphing runtime path.
//!
//! The scene is prepared from the real `AnimatedMorphCube.glb` asset, the draw
//! list wires the morph stage and its per-instance buffers, and three morphing
//! dispatches run through the real compute pipeline. No window is involved.

#[cfg(test)]
#[cfg(any(target_os = "windows", target_os = "linux"))]
mod test {
    use std::path::PathBuf;

    use rust_renderer::{
        asset::{
            asset_manager::AssetManager,
            asset_types::{vertex::VertexBufferPosition, AssetData},
            importer::gltf_importer::GltfImporter,
        },
        renderer::{
            context::gpu_context::GpuContext,
            drawable::{
                draw_list::{DrawList, DrawableType},
                drawable_mesh_traits::Mesh,
            },
            pipeline::compute_library::ComputePipelineLibrary,
            resources::{
                asset_cache::{FinalBuffer, GpuAssetStore, RigStages, SceneUploadContext},
                bindless::BindlessResources,
                mesh_layout::MeshLayout,
            },
            upload::UploadContext,
        },
    };

    const SOURCE_GLTF: &str = "resource/AnimatedMorphCube.glb";
    const FRAMES_TO_DISPATCH: u32 = 3;
    const FRAME_SECONDS: f32 = 1.0 / 30.0;

    /// Copy the morph test model into an isolated temp workspace.
    ///
    /// Importers persist assets next to their source; `resource/` is shared with
    /// other tests, so the model is imported from a temp copy.
    fn temp_workspace() -> PathBuf {
        let dir = std::env::temp_dir().join("rust_renderer_morph_rendering");
        if dir.exists() {
            std::fs::remove_dir_all(&dir).expect("failed to remove old temp dir");
        }
        std::fs::create_dir(&dir).expect("failed to create new temp dir");
        std::fs::copy(SOURCE_GLTF, dir.join("AnimatedMorphCube.glb")).unwrap();
        dir
    }

    fn setup() -> (GpuContext, BindlessResources, ComputePipelineLibrary) {
        let mut gpu = GpuContext::new();
        pollster::block_on(gpu.create_device(None)).expect("device creation should succeed");

        let bindless = BindlessResources::new(gpu.device(), gpu.queue());
        let compute = ComputePipelineLibrary::new(gpu.device(), bindless.get_bind_group_layout());
        (gpu, bindless, compute)
    }

    fn read_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
    ) -> Vec<u8> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("morph readback"),
            size: buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("morph readback encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
        queue.submit(std::iter::once(encoder.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |result| {
            result.expect("readback mapping should succeed");
        });
        device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll should succeed");

        staging.slice(..).get_mapped_range().to_vec()
    }

    fn read_positions(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
    ) -> Vec<[f32; 3]> {
        let bytes = read_bytes(device, queue, buffer);
        bytes
            .chunks_exact(std::mem::size_of::<VertexBufferPosition>())
            .map(bytemuck::pod_read_unaligned::<VertexBufferPosition>)
            .map(|position| position.position)
            .collect()
    }

    fn dispatch_deformation(
        draw_list: &DrawList,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bindless: &BindlessResources,
        compute: &ComputePipelineLibrary,
    ) {
        let bindless_bg = bindless.get_bind_group(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("morph rendering encoder"),
        });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("morph rendering deformation"),
                timestamp_writes: None,
            });
            cp.set_bind_group(0, Some(&bindless_bg), &[]);
            draw_list.record_deformation(&mut cp, compute);
        }
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll should succeed");
    }

    #[test]
    fn morph_rendering_test() {
        let dir = temp_workspace();
        let glb = dir.join("AnimatedMorphCube.glb");
        let assets = AssetManager::new();
        assets.import(GltfImporter::new(&glb)).expect("morph glTF import should succeed");

        let (gpu, mut bindless, compute) = setup();
        let mut store = GpuAssetStore::new();
        let mut draw_list = {
            let scene_name = GltfImporter::scene_subasset_name(&glb, 0)
                .expect("scene name should resolve");
            let asset = assets.get_asset(&scene_name).expect("scene asset should be present");
            let asset = asset.read().expect("failed to read scene asset");
            let AssetData::SceneAssetType(scene) = asset.get_data() else {
                panic!("asset '{scene_name}' is not a scene asset");
            };
            let mut ctx = SceneUploadContext::new(
                UploadContext::new(gpu.device(), gpu.queue(), &mut bindless),
                &assets,
            );
            store.prepare_scene(&mut ctx, scene).expect("scene preparation should succeed")
        };

        assert_eq!(
            draw_list.count(DrawableType::OpaqueSkinned),
            1,
            "the cube has one morph-only primitive"
        );

        let mesh = &draw_list.get_skinned_meshes()[0];
        assert_eq!(mesh.stages(), RigStages::Morph);
        assert_eq!(mesh.final_buffer(), FinalBuffer::MorphOut);
        assert_eq!(mesh.deformed_buffer_count(), 2, "morph only allocates one buffer pair");
        assert_eq!(mesh.get_vertex_buffer().len(), 2);

        let tracks = draw_list.animations().frame(0).expect("the cube has an animation clip");
        assert!(tracks.weights.is_some(), "the clip drives morph weights");
        assert!(tracks.joints.is_none(), "a morph-only clip has no joints track");
        assert!(draw_list.animations().skinned_frame(0).is_none());

        let first_frame = tracks.weights.expect("weights track").frame();
        draw_list.update_animations(FRAME_SECONDS);
        let second_frame = draw_list
            .animations()
            .frame(0)
            .expect("the clip is still playing")
            .weights
            .expect("weights track")
            .frame();
        assert!(second_frame > first_frame, "playback must advance the morph frame");

        let mesh_layout = MeshLayout::new(gpu.device());
        draw_list
            .ensure_bound(gpu.device(), &mesh_layout)
            .ensure_deformation_bound(gpu.device(), &compute);

        let (rest_buffer, final_buffer) = {
            let mesh = &draw_list.get_skinned_meshes()[0];
            (
                mesh.restpose_vertex_buffers()[0].clone(),
                mesh.get_vertex_buffer()[0].clone(),
            )
        };
        let rest_pose = read_positions(gpu.device(), gpu.queue(), &rest_buffer);
        draw_list.update_animations(0.5);
        dispatch_deformation(&draw_list, gpu.device(), gpu.queue(), &bindless, &compute);
        let deformed = read_positions(gpu.device(), gpu.queue(), &final_buffer);
        assert!(
            rest_pose.iter().zip(deformed.iter()).any(|(rest, deformed)| rest != deformed),
            "morphing must modify at least one final vertex; rest={:?}, deformed={:?}",
            &rest_pose[..rest_pose.len().min(4)],
            &deformed[..deformed.len().min(4)],
        );

        for _ in 0..FRAMES_TO_DISPATCH {
            dispatch_deformation(&draw_list, gpu.device(), gpu.queue(), &bindless, &compute);
            draw_list.update_animations(FRAME_SECONDS);
        }
    }
}
