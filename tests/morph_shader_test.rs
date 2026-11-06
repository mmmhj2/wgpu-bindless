#[cfg(any(target_os = "windows", target_os = "linux"))]
mod common;

#[cfg(test)]
#[cfg(any(target_os = "windows", target_os = "linux"))]
mod test {
    use std::num::NonZero;

    use rust_renderer::{
        asset::asset_types::{
            animation_asset::{MorphTarget, VertexMorphTargetAsset},
            texture_asset::{Sampler, TexelData, TextureAsset, TextureDimension, TextureFormat},
            vertex::{VertexBufferOthers, VertexBufferPosition, VertexBufferSkinning},
        },
        renderer::{
            context::gpu_context::GpuContext,
            pipeline::compute_library::{ComputePipelineKey, ComputePipelineLibrary},
            resources::bindless::{BindlessResources, SampledTextureHandle},
            upload::{GpuAssetType, GpuMorphTargets, UploadContext},
        },
    };
    use crate::common;

    const WORKGROUP_SIZE: u32 = 64;

    fn storage_buffer<T: bytemuck::Pod>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
        data: &[T],
        extra_usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: std::mem::size_of_val(data) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra_usage,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        buffer
    }

    fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: NonZero::new(buffer.size()),
            }),
        }
    }

    /// Build an RGBA16F weight texture. Rows are laid out like the baked
    /// animation asset: four target weights per texel, one row per frame.
    fn weight_texture(rows: &[Vec<f32>], channels: usize) -> TextureAsset {
        let width = channels.div_ceil(4).max(1) as u32;
        let mut texels = Vec::with_capacity(rows.len() * width as usize * 8);
        for row in rows {
            for x in 0..width as usize {
                for lane in 0..4usize {
                    let value = row.get(x * 4 + lane).copied().unwrap_or(0.0);
                    texels.extend_from_slice(&common::f16_bits(value).to_le_bytes());
                }
            }
        }

        TextureAsset {
            format: TextureFormat::Rgba16Float,
            dimension: TextureDimension::D2,
            width: NonZero::new(width).expect("weight texture is at least one texel wide"),
            height: Some(NonZero::new(rows.len() as u32).expect("weight texture has at least one row")),
            depth_or_array_slice: None,
            sampler: Sampler::default(),
            texels: TexelData::Plain(texels.into_boxed_slice()),
        }
    }

    fn upload_target(
        gpu: &GpuContext,
        bindless: &mut BindlessResources,
        target: &VertexMorphTargetAsset,
        weights: &TextureAsset,
    ) -> (GpuMorphTargets, SampledTextureHandle) {
        let mut upload = UploadContext::new(gpu.device(), gpu.queue(), bindless);
        let target = target
            .upload("test.morph", &mut upload)
            .expect("morph target upload should succeed");
        let weights = upload
            .upload_texture("test.weight", weights)
            .expect("weight texture should upload")
            .handle;
        (target, weights)
    }

    fn run_morph(
        gpu: &GpuContext,
        bindless: &mut BindlessResources,
        compute: &ComputePipelineLibrary,
        target: &GpuMorphTargets,
        weights: SampledTextureHandle,
        frame: f32,
        channels: u32,
        positions: &[VertexBufferPosition],
        attributes: &[VertexBufferOthers],
    ) -> (Vec<[f32; 3]>, Vec<VertexBufferOthers>) {
        let device = gpu.device();
        let queue = gpu.queue();
        assert_eq!(positions.len(), attributes.len());

        let position_bytes = std::mem::size_of_val(positions);
        let attribute_bytes = std::mem::size_of_val(attributes);

        let position_in = storage_buffer(device, queue, "test position in", positions, wgpu::BufferUsages::empty());
        let attribute_in = storage_buffer(device, queue, "test attribute in", attributes, wgpu::BufferUsages::empty());

        // Zero the outputs so a partially written dispatch cannot pass by
        // accidentally matching the input contents.
        let zero_positions = vec![VertexBufferPosition { position: [0.0; 3] }; positions.len()];
        let zero_attributes = vec![VertexBufferOthers {
            color: [0.0; 4],
            normal: [0.0; 3],
            tangent: [0.0; 4],
            uv0: [0.0; 2],
        }; attributes.len()];
        let position_out = storage_buffer(device, queue, "test position out", &zero_positions, wgpu::BufferUsages::COPY_SRC);
        let attribute_out = storage_buffer(device, queue, "test attribute out", &zero_attributes, wgpu::BufferUsages::COPY_SRC);

        // Morphing declares the skinning buffer to share group 1, but never
        // reads it.
        let bones = [VertexBufferSkinning {
            bone_indices: [0; 4],
            bone_weights: [0.0; 4],
        }];
        let bones_buffer = storage_buffer(device, queue, "test bones", &bones, wgpu::BufferUsages::empty());

        let group1_layout = compute
            .get_dispatch_bind_group_layout(ComputePipelineKey::GpuVertexMorphing, 0)
            .expect("morphing pipeline should expose a group-1 layout");
        let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("test morph group 1"),
            layout: group1_layout,
            entries: &[
                binding(0, &position_in),
                binding(1, &attribute_in),
                binding(2, &position_out),
                binding(3, &attribute_out),
                binding(4, &bones_buffer),
            ],
        });

        let group2_layout = compute
            .get_dispatch_bind_group_layout(ComputePipelineKey::GpuVertexMorphing, 1)
            .expect("morphing pipeline should expose a group-2 layout");
        let group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("test morph group 2"),
            layout: group2_layout,
            entries: &[binding(0, &target.layout), binding(1, &target.deltas)],
        });

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test morph readback"),
            size: (position_bytes + attribute_bytes) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bindless_bg = bindless.get_bind_group(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test morph encoder"),
        });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("test morph pass"),
                timestamp_writes: None,
            });
            cp.set_pipeline(
                compute
                    .get_pipeline(ComputePipelineKey::GpuVertexMorphing)
                    .expect("morphing pipeline should be registered"),
            );
            cp.set_bind_group(0, Some(&bindless_bg), &[]);
            cp.set_bind_group(1, Some(&group1), &[]);
            cp.set_bind_group(2, &group2, &[]);

            // Layout must stay in sync with `Immediate` in
            // `gpu_vertex_morph.wgsl`.
            let mut immediates = [0u8; 12];
            immediates[..4].copy_from_slice(&(weights.index() as u32).to_le_bytes());
            immediates[4..8].copy_from_slice(&frame.to_le_bytes());
            immediates[8..].copy_from_slice(&channels.to_le_bytes());
            cp.set_immediates(0, &immediates);

            cp.dispatch_workgroups((positions.len() as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&position_out, 0, &staging, 0, position_bytes as u64);
        encoder.copy_buffer_to_buffer(&attribute_out, 0, &staging, position_bytes as u64, attribute_bytes as u64);
        queue.submit(std::iter::once(encoder.finish()));

        staging.slice(..).map_async(wgpu::MapMode::Read, |result| {
            result.expect("readback mapping should succeed");
        });
        device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll should succeed");

        let mapped = staging.slice(..).get_mapped_range();
        let positions: &[VertexBufferPosition] = bytemuck::cast_slice(&mapped[..position_bytes]);
        let attributes: &[VertexBufferOthers] = bytemuck::cast_slice(&mapped[position_bytes..]);
        (
            positions.iter().map(|position| position.position).collect(),
            attributes.to_vec(),
        )
    }

    fn assert_vec3(actual: [f32; 3], expected: [f32; 3], label: &str) {
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() < 1e-4,
                "{label}: expected {expected:?}, got {actual:?}"
            );
        }
    }

    fn setup() -> (GpuContext, BindlessResources, ComputePipelineLibrary) {
        let mut gpu = GpuContext::new();
        pollster::block_on(gpu.create_device(None)).expect("device creation should succeed");

        let bindless = BindlessResources::new(gpu.device(), gpu.queue());
        let compute = ComputePipelineLibrary::new(gpu.device(), bindless.get_bind_group_layout());
        (gpu, bindless, compute)
    }

    fn position(x: f32, y: f32, z: f32) -> VertexBufferPosition {
        VertexBufferPosition { position: [x, y, z] }
    }

    fn attribute() -> VertexBufferOthers {
        VertexBufferOthers {
            color: [1.0, 1.0, 1.0, 1.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            uv0: [0.0, 0.0],
        }
    }

    fn target(targets: Vec<MorphTarget>, default_weights: Option<Vec<f32>>) -> VertexMorphTargetAsset {
        VertexMorphTargetAsset {
            mesh: "test.skpr".into(),
            targets,
            default_weights,
        }
    }

    #[test]
    fn single_channel_moves_the_position() {
        let (gpu, mut bindless, compute) = setup();
        let asset = target(
            vec![MorphTarget { position: Some(vec![[1.0, 0.0, 0.0]]), ..Default::default() }],
            None,
        );
        let weights = weight_texture(&[vec![1.0]], 1);
        let (gpu_target, handle) = upload_target(&gpu, &mut bindless, &asset, &weights);

        let (positions, _) = run_morph(
            &gpu,
            &mut bindless,
            &compute,
            &gpu_target,
            handle,
            0.0,
            1,
            &[position(0.0, 0.0, 0.0)],
            &[attribute()],
        );

        assert_vec3(positions[0], [1.0, 0.0, 0.0], "single channel");
    }

    #[test]
    fn per_vertex_deltas_use_each_vertex_offset() {
        let (gpu, mut bindless, compute) = setup();
        let asset = target(
            vec![MorphTarget {
                position: Some(vec![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
                normal: Some(vec![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                texcoord0: Some(vec![[1.0, 0.0], [0.0, 1.0]]),
                color0: Some(vec![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
                ..Default::default()
            }],
            None,
        );
        let weights = weight_texture(&[vec![1.0]], 1);
        let (gpu_target, handle) = upload_target(&gpu, &mut bindless, &asset, &weights);
        let mut attribute0 = attribute();
        attribute0.color = [0.0, 0.0, 0.0, 0.0];
        attribute0.uv0 = [0.0, 0.0];
        let mut attribute1 = attribute();
        attribute1.color = [0.0, 0.0, 0.0, 0.0];
        attribute1.normal = [0.0, 0.0, 1.0];
        attribute1.uv0 = [0.0, 0.0];

        let (positions, attributes) = run_morph(
            &gpu,
            &mut bindless,
            &compute,
            &gpu_target,
            handle,
            0.0,
            1,
            &[position(0.0, 0.0, 0.0), position(0.0, 0.0, 0.0)],
            &[attribute0, attribute1],
        );

        assert_vec3(positions[0], [1.0, 0.0, 0.0], "vertex 0 position delta");
        assert_vec3(positions[1], [0.0, 2.0, 0.0], "vertex 1 must use its own position delta");
        assert_vec3(attributes[0].normal, [0.0, 1.0, 0.0], "vertex 0 normal delta");
        assert_vec3(attributes[1].normal, [0.0, 0.0, 1.0], "vertex 1 normal delta");
        assert_eq!(attributes[0].uv0, [1.0, 0.0], "vertex 0 uv delta");
        assert_eq!(attributes[1].uv0, [0.0, 1.0], "vertex 1 uv delta");
        assert_eq!(attributes[0].color, [1.0, 0.0, 0.0, 0.0], "vertex 0 color delta");
        assert_eq!(attributes[1].color, [0.0, 1.0, 0.0, 0.0], "vertex 1 color delta");
    }

    #[test]
    fn two_channels_accumulate() {
        let (gpu, mut bindless, compute) = setup();
        let asset = target(
            vec![
                MorphTarget { position: Some(vec![[1.0, 0.0, 0.0]]), ..Default::default() },
                MorphTarget { position: Some(vec![[0.0, 1.0, 0.0]]), ..Default::default() },
            ],
            None,
        );
        let weights = weight_texture(&[vec![0.5, 0.5]], 2);
        let (gpu_target, handle) = upload_target(&gpu, &mut bindless, &asset, &weights);

        let (positions, _) = run_morph(
            &gpu,
            &mut bindless,
            &compute,
            &gpu_target,
            handle,
            0.0,
            2,
            &[position(0.0, 0.0, 0.0)],
            &[attribute()],
        );

        assert_vec3(positions[0], [0.5, 0.5, 0.0], "two channel sum");
    }

    #[test]
    fn fractional_frame_interpolates_between_rows() {
        let (gpu, mut bindless, compute) = setup();
        let asset = target(
            vec![MorphTarget { position: Some(vec![[1.0, 0.0, 0.0]]), ..Default::default() }],
            None,
        );
        // Two rows: frame 0 has weight 0, frame 1 has weight 1.
        let weights = weight_texture(&[vec![0.0], vec![1.0]], 1);
        let (gpu_target, handle) = upload_target(&gpu, &mut bindless, &asset, &weights);

        let (positions, _) = run_morph(
            &gpu,
            &mut bindless,
            &compute,
            &gpu_target,
            handle,
            0.5,
            1,
            &[position(0.0, 0.0, 0.0)],
            &[attribute()],
        );

        assert_vec3(positions[0], [0.5, 0.0, 0.0], "half-way between the two rows");
    }

    #[test]
    fn sparse_target_leaves_missing_attributes_untouched() {
        let (gpu, mut bindless, compute) = setup();
        let asset = target(
            vec![MorphTarget { position: Some(vec![[1.0, 0.0, 0.0]]), ..Default::default() }],
            None,
        );
        let weights = weight_texture(&[vec![1.0]], 1);
        let (gpu_target, handle) = upload_target(&gpu, &mut bindless, &asset, &weights);
        let input = attribute();

        let (positions, attributes) = run_morph(
            &gpu,
            &mut bindless,
            &compute,
            &gpu_target,
            handle,
            0.0,
            1,
            &[position(0.0, 0.0, 0.0)],
            &[input],
        );

        assert_vec3(positions[0], [1.0, 0.0, 0.0], "position");
        assert_eq!(attributes[0].normal, input.normal, "missing normal must pass through");
        assert_eq!(attributes[0].tangent, input.tangent, "missing tangent must pass through");
        assert_eq!(attributes[0].color, input.color, "missing color must pass through");
        assert_eq!(attributes[0].uv0, input.uv0, "missing uv must pass through");
    }

    #[test]
    fn normal_delta_is_renormalized() {
        let (gpu, mut bindless, compute) = setup();
        let asset = target(
            vec![MorphTarget { normal: Some(vec![[1.0, 0.0, 0.0]]), ..Default::default() }],
            None,
        );
        let weights = weight_texture(&[vec![1.0]], 1);
        let (gpu_target, handle) = upload_target(&gpu, &mut bindless, &asset, &weights);

        let (_, attributes) = run_morph(
            &gpu,
            &mut bindless,
            &compute,
            &gpu_target,
            handle,
            0.0,
            1,
            &[position(0.0, 0.0, 0.0)],
            &[attribute()],
        );

        let expected = std::f32::consts::FRAC_1_SQRT_2;
        assert_vec3(attributes[0].normal, [expected, expected, 0.0], "normalized normal");
    }

    #[test]
    fn zero_channels_copy_the_rest_pose_byte_for_byte() {
        let (gpu, mut bindless, compute) = setup();
        let asset = target(
            vec![MorphTarget { position: Some(vec![[5.0, 5.0, 5.0]]), ..Default::default() }],
            None,
        );
        let weights = weight_texture(&[vec![1.0]], 1);
        let (gpu_target, handle) = upload_target(&gpu, &mut bindless, &asset, &weights);
        let input_position = position(1.0, 2.0, 3.0);
        let input_attribute = attribute();

        let (positions, attributes) = run_morph(
            &gpu,
            &mut bindless,
            &compute,
            &gpu_target,
            handle,
            0.0,
            0,
            &[input_position],
            &[input_attribute],
        );

        assert_eq!(positions[0], input_position.position);
        assert_eq!(attributes[0].color, input_attribute.color);
        assert_eq!(attributes[0].normal, input_attribute.normal);
        assert_eq!(attributes[0].tangent, input_attribute.tangent);
        assert_eq!(attributes[0].uv0, input_attribute.uv0);
    }

    #[test]
    fn color_delta_keeps_the_initial_alpha() {
        let (gpu, mut bindless, compute) = setup();
        let asset = target(
            vec![MorphTarget { color0: Some(vec![[1.0, 0.0, 0.0, 0.0]]), ..Default::default() }],
            None,
        );
        let weights = weight_texture(&[vec![1.0]], 1);
        let (gpu_target, handle) = upload_target(&gpu, &mut bindless, &asset, &weights);
        let mut input = attribute();
        input.color = [0.0, 0.0, 0.0, 0.0];

        let (_, attributes) = run_morph(
            &gpu,
            &mut bindless,
            &compute,
            &gpu_target,
            handle,
            0.0,
            1,
            &[position(0.0, 0.0, 0.0)],
            &[input],
        );

        assert_eq!(attributes[0].color, [1.0, 0.0, 0.0, 0.0], "VEC3 source delta has alpha 0");
    }

    #[test]
    fn texcoord_delta_is_not_normalized() {
        let (gpu, mut bindless, compute) = setup();
        let asset = target(
            vec![MorphTarget { texcoord0: Some(vec![[2.0, 0.0]]), ..Default::default() }],
            None,
        );
        let weights = weight_texture(&[vec![1.0]], 1);
        let (gpu_target, handle) = upload_target(&gpu, &mut bindless, &asset, &weights);

        let (_, attributes) = run_morph(
            &gpu,
            &mut bindless,
            &compute,
            &gpu_target,
            handle,
            0.0,
            1,
            &[position(0.0, 0.0, 0.0)],
            &[attribute()],
        );

        assert_eq!(attributes[0].uv0, [2.0, 0.0], "uv deltas must not be normalized");
    }

    #[test]
    fn base_weights_deform_at_frame_zero() {
        let (gpu, mut bindless, compute) = setup();
        let asset = target(
            vec![
                MorphTarget { position: Some(vec![[1.0, 0.0, 0.0]]), ..Default::default() },
                MorphTarget { position: Some(vec![[0.0, 1.0, 0.0]]), ..Default::default() },
            ],
            Some(vec![0.25, 0.75]),
        );
        let (gpu_target, _) = upload_target(&gpu, &mut bindless, &asset, &weight_texture(&[vec![0.0]], 1));

        let base_weights = gpu_target.base_weights.expect("non-zero base weights must allocate a texture");
        let channels = gpu_target.base_channels.expect("base channels must be present").get();
        assert_eq!(channels, 2);

        let (positions, _) = run_morph(
            &gpu,
            &mut bindless,
            &compute,
            &gpu_target,
            base_weights,
            0.0,
            channels,
            &[position(0.0, 0.0, 0.0)],
            &[attribute()],
        );

        assert_vec3(positions[0], [0.25, 0.75, 0.0], "static base weights");
    }
}
