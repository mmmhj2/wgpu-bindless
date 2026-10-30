//! Headless GPU tests for the skinning compute shader.
//!
//! They drive the real pipeline with hand-built buffers and a one-frame animation
//! texture, then read the deformed vertex buffers back and compare them against a
//! CPU reference. No window or surface is involved.

#[cfg(target_os = "windows")]
mod common;

#[cfg(test)]
#[cfg(target_os = "windows")]
mod test {
    use std::num::NonZero;

    use rust_renderer::{
        asset::asset_types::{
            GpuAssetType,
            texture_asset::TextureAsset,
            vertex::{VertexBufferOthers, VertexBufferPosition, VertexBufferSkinning},
        },
        renderer::{
            context::gpu_context::GpuContext,
            pipeline::compute_library::{ComputePipelineKey, ComputePipelineLibrary},
            resources::bindless::BindlessResources,
        },
    };

    use crate::common;

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

    /// Run one skinning dispatch and read the deformed buffers back.
    fn run_skinning(
        gpu: &GpuContext,
        bindless: &mut BindlessResources,
        compute: &ComputePipelineLibrary,
        animation: &TextureAsset,
        positions: &[VertexBufferPosition],
        attributes: &[VertexBufferOthers],
        bones: &[VertexBufferSkinning],
    ) -> (Vec<[f32; 3]>, Vec<VertexBufferOthers>) {
        let device = gpu.device();
        let queue = gpu.queue();
        assert_eq!(positions.len(), attributes.len());
        assert_eq!(positions.len(), bones.len());

        let position_bytes = std::mem::size_of_val(positions);
        let attribute_bytes = std::mem::size_of_val(attributes);

        let texture = animation.upload("test joint matrices", device, queue);
        let animation_handle = bindless
            .create_sampled_texture(texture.create_view(&Default::default()))
            .expect("animation texture should fit in the bindless table");

        let position_in = storage_buffer(device, queue, "test position in", positions, wgpu::BufferUsages::empty());
        let attribute_in = storage_buffer(device, queue, "test attribute in", attributes, wgpu::BufferUsages::empty());
        let position_out = storage_buffer(device, queue, "test position out", positions, wgpu::BufferUsages::COPY_SRC);
        let attribute_out = storage_buffer(device, queue, "test attribute out", attributes, wgpu::BufferUsages::COPY_SRC);
        let bones_buffer = storage_buffer(device, queue, "test bones", bones, wgpu::BufferUsages::empty());

        let skinning_layout = compute
            .get_dispatch_bind_group_layout(ComputePipelineKey::GpuSkinning)
            .expect("skinning pipeline should be registered");
        let skinning_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("test skinning bind group"),
            layout: skinning_layout,
            entries: &[
                binding(0, &position_in),
                binding(1, &attribute_in),
                binding(2, &position_out),
                binding(3, &attribute_out),
                binding(4, &bones_buffer),
            ],
        });

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test readback"),
            size: (position_bytes + attribute_bytes) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bindless_bg = bindless.get_bind_group(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test skinning encoder"),
        });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("test skinning pass"),
                timestamp_writes: None,
            });
            cp.set_pipeline(
                compute.get_pipeline(ComputePipelineKey::GpuSkinning)
                    .expect("skinning pipeline should be registered"),
            );
            cp.set_bind_group(0, Some(&bindless_bg), &[]);
            cp.set_bind_group(1, Some(&skinning_bg), &[]);

            // Layout must stay in sync with `Immediate` in `gpu_skinning.wgsl`.
            let mut immediates = [0u8; 8];
            immediates[..4].copy_from_slice(&(animation_handle.index() as u32).to_le_bytes());
            immediates[4..].copy_from_slice(&0u32.to_le_bytes());
            cp.set_immediates(0, &immediates);

            cp.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&position_out, 0, &staging, 0, position_bytes as u64);
        encoder.copy_buffer_to_buffer(&attribute_out, 0, &staging, position_bytes as u64, attribute_bytes as u64);
        queue.submit(std::iter::once(encoder.finish()));

        staging.slice(..).map_async(wgpu::MapMode::Read, |result| {
            result.expect("readback mapping should succeed");
        });
        device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll should succeed");

        let mapped = staging.slice(..).get_mapped_range();
        let deformed_positions: &[VertexBufferPosition] = bytemuck::cast_slice(&mapped[..position_bytes]);
        let deformed_attributes: &[VertexBufferOthers] = bytemuck::cast_slice(&mapped[position_bytes..]);

        (
            deformed_positions.iter().map(|p| p.position).collect(),
            deformed_attributes.to_vec(),
        )
    }

    fn assert_vec3(actual: [f32; 3], expected: [f32; 3], label: &str) {
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() < 1e-3,
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

    /// 90 degree rotation about +Z, column-major.
    const ROTATION_Z_90: [[f32; 4]; 4] = [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    #[test]
    fn skinning_shader_matches_cpu_reference() {
        let (gpu, mut bindless, compute) = setup();

        let positions = [
            VertexBufferPosition { position: [1.0, 0.0, 0.0] },
            VertexBufferPosition { position: [0.0, 1.0, 0.0] },
            VertexBufferPosition { position: [0.0, 0.0, 1.0] },
            VertexBufferPosition { position: [2.0, 3.0, 0.0] },
            // Last vertex has no influence at all and must keep its rest pose.
            VertexBufferPosition { position: [5.0, 6.0, 7.0] },
        ];
        let attributes = [
            VertexBufferOthers { color: [1.0, 0.0, 0.0, 1.0], normal: [0.0, 1.0, 0.0], tangent: [1.0, 0.0, 0.0, -1.0], uv0: [0.0, 0.0] },
            VertexBufferOthers { color: [0.0, 1.0, 0.0, 1.0], normal: [1.0, 0.0, 0.0], tangent: [0.0, 1.0, 0.0, 1.0], uv0: [1.0, 0.0] },
            VertexBufferOthers { color: [0.0, 0.0, 1.0, 1.0], normal: [0.0, 0.0, 1.0], tangent: [1.0, 0.0, 0.0, 1.0], uv0: [0.0, 1.0] },
            VertexBufferOthers { color: [1.0, 1.0, 1.0, 1.0], normal: [0.0, 1.0, 0.0], tangent: [1.0, 0.0, 0.0, -1.0], uv0: [0.5, 0.5] },
            VertexBufferOthers { color: [0.0, 0.0, 0.0, 0.0], normal: [0.0, 1.0, 0.0], tangent: [1.0, 0.0, 0.0, 1.0], uv0: [0.25, 0.75] },
        ];
        let full_influence = VertexBufferSkinning {
            bone_indices: [0, 0, 0, 0],
            bone_weights: [1.0, 0.0, 0.0, 0.0],
        };
        let bones = [
            full_influence,
            full_influence,
            full_influence,
            full_influence,
            VertexBufferSkinning { bone_indices: [0, 0, 0, 0], bone_weights: [0.0, 0.0, 0.0, 0.0] },
        ];

        let animation = common::joint_matrix_texture(&ROTATION_Z_90);
        let (deformed_positions, deformed_attributes) =
            run_skinning(&gpu, &mut bindless, &compute, &animation, &positions, &attributes, &bones);

        // 90 degree rotation about +Z maps (x, y, z) to (-y, x, z).
        assert_vec3(deformed_positions[0], [0.0, 1.0, 0.0], "p0");
        assert_vec3(deformed_positions[1], [-1.0, 0.0, 0.0], "p1");
        assert_vec3(deformed_positions[2], [0.0, 0.0, 1.0], "p2");
        assert_vec3(deformed_positions[3], [-3.0, 2.0, 0.0], "p3");
        assert_vec3(deformed_positions[4], [5.0, 6.0, 7.0], "p4 (no influence)");

        assert_vec3(deformed_attributes[0].normal, [-1.0, 0.0, 0.0], "n0");
        assert_vec3(deformed_attributes[1].normal, [0.0, 1.0, 0.0], "n1");
        assert_vec3(deformed_attributes[2].normal, [0.0, 0.0, 1.0], "n2");

        assert_vec3(
            [deformed_attributes[0].tangent[0], deformed_attributes[0].tangent[1], deformed_attributes[0].tangent[2]],
            [0.0, 1.0, 0.0],
            "t0",
        );
        // Handedness must survive the transform.
        assert!((deformed_attributes[0].tangent[3] + 1.0).abs() < 1e-6, "t0 handedness should be preserved");
        assert!((deformed_attributes[1].tangent[3] - 1.0).abs() < 1e-6, "t1 handedness should be preserved");

        // Attributes that are not transformed must be passed through verbatim.
        for (index, (before, after)) in attributes.iter().zip(deformed_attributes.iter()).enumerate() {
            for channel in 0..4 {
                assert!(
                    (before.color[channel] - after.color[channel]).abs() < 1e-6,
                    "vertex {index}: color must pass through"
                );
            }
            for channel in 0..2 {
                assert!(
                    (before.uv0[channel] - after.uv0[channel]).abs() < 1e-6,
                    "vertex {index}: uv0 must pass through"
                );
            }
        }
    }

    #[test]
    fn skinning_weights_blend_two_joints() {
        let (gpu, mut bindless, compute) = setup();

        // Joint 0 is the identity, joint 1 rotates 90 degrees about +Z.
        let mut columns: Vec<[f32; 4]> = common::IDENTITY_COLUMNS.to_vec();
        columns.extend_from_slice(&ROTATION_Z_90);
        let animation = common::joint_matrix_texture(&columns);

        let positions = [
            // Half identity, half rotation: (1,0,0) -> 0.5*(1,0,0) + 0.5*(0,1,0).
            VertexBufferPosition { position: [1.0, 0.0, 0.0] },
            // Only the second slot is used, so the weight must still be applied.
            VertexBufferPosition { position: [1.0, 0.0, 0.0] },
            // Zero weights in the first slot must be skipped, not multiplied in.
            VertexBufferPosition { position: [1.0, 0.0, 0.0] },
        ];
        let attributes = [
            VertexBufferOthers { color: [1.0, 1.0, 1.0, 1.0], normal: [0.0, 1.0, 0.0], tangent: [1.0, 0.0, 0.0, 1.0], uv0: [0.0, 0.0] },
            VertexBufferOthers { color: [1.0, 1.0, 1.0, 1.0], normal: [0.0, 1.0, 0.0], tangent: [1.0, 0.0, 0.0, 1.0], uv0: [0.0, 0.0] },
            VertexBufferOthers { color: [1.0, 1.0, 1.0, 1.0], normal: [0.0, 1.0, 0.0], tangent: [1.0, 0.0, 0.0, 1.0], uv0: [0.0, 0.0] },
        ];
        let bones = [
            VertexBufferSkinning { bone_indices: [0, 1, 0, 0], bone_weights: [0.5, 0.5, 0.0, 0.0] },
            VertexBufferSkinning { bone_indices: [0, 1, 0, 0], bone_weights: [0.0, 1.0, 0.0, 0.0] },
            VertexBufferSkinning { bone_indices: [0, 1, 0, 0], bone_weights: [0.0, 1.0, 0.0, 0.0] },
        ];

        let (deformed_positions, deformed_attributes) =
            run_skinning(&gpu, &mut bindless, &compute, &animation, &positions, &attributes, &bones);

        assert_vec3(deformed_positions[0], [0.5, 0.5, 0.0], "blended");
        assert_vec3(deformed_positions[1], [0.0, 1.0, 0.0], "second slot only");
        assert_vec3(deformed_positions[2], [0.0, 1.0, 0.0], "zero weight skipped");

        // The blended normal is normalized after accumulation:
        // 0.5*(0,1,0) + 0.5*(-1,0,0) = (-0.5,0.5,0) -> (-1,1,0)/sqrt(2).
        let expected_normal = [-0.707_106_8, 0.707_106_8, 0.0];
        assert_vec3(deformed_attributes[0].normal, expected_normal, "blended normal");
    }
}
