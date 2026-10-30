use std::{num::NonZero, sync::Arc};

use crate::{
    asset::asset_types::{skinned_mesh_asset::GpuSkinnedMesh, vertex::{VertexBufferOthers, VertexBufferPosition}},
    renderer::{
        drawable::{drawable_mesh_traits::{DrawableMesh, Mesh}, vertex_types::VertexType},
        pipeline::{compute_library::ComputePipelineKey, pbr_material::PBRMaterial},
        resources::asset_cache::MeshUploadContext,
    },
};

const SKINNING_WORKGROUP_SIZE: u32 = 64;
const MODEL_MATRIX_SIZE: u64 = std::mem::size_of::<[[f32; 4]; 4]>() as u64;

/// Skinned animated mesh.
/// 
/// Skinned meshes store both a rest-pose reference mesh in `geometry`, and the
/// intermediate vertex attributes that are produced by the compute shader.
/// 
/// Contrary to static meshes, skinned meshes are *never* instanced.
#[derive(Debug)]
pub struct SkinnedMesh {
    geometry: Arc<GpuSkinnedMesh>,
    /// Deformed vertex streams written by the skinning compute shader and read
    /// by the rasterizer, in the same order as the rest pose streams.
    temp_buffer: Vec<wgpu::Buffer>,
    material: PBRMaterial,

    #[allow(dead_code)]
    model_matrix_buffer: wgpu::Buffer,

    // TODO: building these bind group on construction means heavy dependency
    // on both the mesh layout and the compute shader library. Maybe defer
    // their creation on use with OnceCell can clean up the design.
    model_matrix_bg: wgpu::BindGroup,
    skinning_dispatch_bg: wgpu::BindGroup,
}

impl SkinnedMesh {

    pub(crate) fn new(
        ctx: &mut MeshUploadContext<'_>,
        geometry: Arc<GpuSkinnedMesh>,
        material: PBRMaterial,
        world_matrix: [[f32; 4]; 4],
    ) -> Self {
        let vertex_count = u64::from(geometry.vertex_count);

        let deformed_position = ctx.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("skinned mesh deformed position"),
            size: vertex_count * std::mem::size_of::<VertexBufferPosition>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false
        });
        let deformed_attribute = ctx.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("skinned mesh deformed attribute"),
            size: vertex_count * std::mem::size_of::<VertexBufferOthers>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false
        });
        let temp_buffer = vec![deformed_position, deformed_attribute];

        let model_matrix_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("skinned mesh model matrix"),
            size: MODEL_MATRIX_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });
        ctx.queue.write_buffer(&model_matrix_buffer, 0, bytemuck::cast_slice(&world_matrix));
        let model_matrix_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("skinned mesh model matrix bind group"),
            layout: ctx.mesh_layout.get_bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &model_matrix_buffer,
                        offset: 0,
                        size: NonZero::new(model_matrix_buffer.size()),
                    }),
                },
            ],
        });

        let skinning_layout = ctx.compute_pipeline
            .get_dispatch_bind_group_layout(ComputePipelineKey::GpuSkinning)
            .expect("the GPU skinning compute pipeline should be registered");
        let skinning_dispatch_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("skinning dispatch bind group"),
            layout: skinning_layout,
            entries: &[
                Self::storage_entry(0, &geometry.restpose_vertex_buffer[0]),
                Self::storage_entry(1, &geometry.restpose_vertex_buffer[1]),
                Self::storage_entry(2, &temp_buffer[0]),
                Self::storage_entry(3, &temp_buffer[1]),
                Self::storage_entry(4, &geometry.skinning_buffer),
            ],
        });

        Self {
            geometry,
            temp_buffer,
            material,
            model_matrix_buffer,
            model_matrix_bg,
            skinning_dispatch_bg,
        }
    }

    fn storage_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: NonZero::new(buffer.size()),
            }),
        }
    }

    /// Record the skinning dispatch that produces the deformed vertex buffers.
    /// 
    /// This method assumes that the compute pass is set up correctly:
    /// - The skinning compute pipeline is bound;
    /// - Bindless bind group (group 0) is bound.
    ///
    /// It binds the per-mesh bind group (group 1), sets the animation immediates
    /// and dispatches one workgroup per 64 vertices.
    ///
    /// TODO: revisit this API design after Skinning animation is actually running.
    pub fn skin_with(&self, cp: &mut wgpu::ComputePass, anim_texture: u32, frame: u32) {
        cp.set_bind_group(1, Some(&self.skinning_dispatch_bg), &[]);

        // Layout must stay in sync with `Immediate` in `gpu_skinning.wgsl`.
        let mut immediates = [0u8; 8];
        immediates[..4].copy_from_slice(&anim_texture.to_le_bytes());
        immediates[4..].copy_from_slice(&frame.to_le_bytes());
        cp.set_immediates(0, &immediates);

        cp.dispatch_workgroups(self.geometry.vertex_count.div_ceil(SKINNING_WORKGROUP_SIZE), 1, 1);
    }

    pub fn get_material(&self) -> &PBRMaterial { &self.material }
}

impl Mesh for SkinnedMesh {
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer] { &self.temp_buffer }
    fn get_vertex_draw_count(&self) -> u32 { self.geometry.vertex_draw_count }
    fn get_index_buffer(&self) -> Option<&wgpu::Buffer> { self.geometry.index_buffer.as_ref() }
    fn get_vertex_type(&self) -> VertexType { VertexType::Basic }
}

impl DrawableMesh for SkinnedMesh {
    fn get_model_matrix_bind_group(&self) -> &wgpu::BindGroup { &self.model_matrix_bg }
    fn get_instance_range(&self) -> std::ops::Range<u32> { 0..1 }
    fn get_material(&self) -> &PBRMaterial { &self.material }
}
