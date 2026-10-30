use serde::{Serialize, Deserialize};

use crate::asset::asset_types::{ConcreteAssetType, GpuAssetType, static_mesh_asset::StaticMeshAsset, vertex::VertexBufferSkinning};

#[derive(Serialize, Deserialize, Debug)]
pub struct SkinnedMeshAsset {
    /// Name of the skin asset that this mesh is bound to.
    pub skin: String,
    /// Rest pose geometry, in the object space of the skinned mesh.
    pub restpose: StaticMeshAsset,
    /// Per-vertex joint indices and weights, indexed like the rest pose vertex
    /// buffers. Consumed by the skinning compute shader as a storage buffer.
    pub animation : Vec<VertexBufferSkinning>,
}

/// GPU-resident geometry produced by uploading a [`SkinnedMeshAsset`].
///
/// The `restpose_vertex_buffer` are feed to the skinning compute buffer, and
/// then consumed by the rasterizer. These buffers are assumed to have
/// storage (UAV) and vertex usages enabled.
#[derive(Debug)]
pub struct GpuSkinnedMesh {
    /// Rest pose vertex streams, ordered like the renderer's vertex buffer
    /// layouts (position, then attributes).
    pub restpose_vertex_buffer: Vec<wgpu::Buffer>,
    /// Joint indices and weights, one [`VertexBufferSkinning`] per vertex.
    pub skinning_buffer: wgpu::Buffer,
    pub index_buffer: Option<wgpu::Buffer>,
    /// Number of vertices in the rest pose. Drives the compute dispatch size,
    /// which must not be confused with `vertex_draw_count` (an index count).
    pub vertex_count: u32,
    pub vertex_draw_count: u32
}

/// Size of one [`VertexBufferSkinning`] entry.
const SKINNING_ENTRY_SIZE: u64 = std::mem::size_of::<VertexBufferSkinning>() as u64;

impl ConcreteAssetType for SkinnedMeshAsset {
    const SCHEMA_VERSION: u32 = 1;
}

impl GpuAssetType for SkinnedMeshAsset {
    type GpuObjectType = GpuSkinnedMesh;
    
    /// Upload the rest-pose vertex buffers onto the GPU.
    /// 
    /// In case of an animated mesh, each instance of the mesh must be allocated
    /// separate buffers to store the intermediate vertex attributes calculated
    /// by the compute shader. These buffers should ideally be handled by the
    /// runtime mesh/scene manager, instead of being created here.
    ///
    /// Rest pose vertex buffers are additionally flagged as `STORAGE` so the
    /// skinning compute shader can read them as typed storage buffers.
    fn upload(&self, name: &str, device: &wgpu::Device, queue: &wgpu::Queue) -> Self::GpuObjectType {
        let restpose = &self.restpose;
        assert_eq!(
            restpose.vp.len(),
            restpose.va.len(),
            "position and attribute buffers must have the same length."
        );
        assert_eq!(
            restpose.vp.len(),
            self.animation.len(),
            "skinned mesh '{name}' has {} vertices but {} skinning entries.",
            restpose.vp.len(),
            self.animation.len()
        );

        let position_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some(&format!("{name}.restpose.position")),
            size: std::mem::size_of_val(restpose.vp.as_slice()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&position_buffer, 0, bytemuck::cast_slice(&restpose.vp));

        let attribute_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some(&format!("{name}.restpose.attribute")),
            size: std::mem::size_of_val(restpose.va.as_slice()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&attribute_buffer, 0, bytemuck::cast_slice(&restpose.va));

        // A zero-sized buffer cannot be created; fall back to a single zeroed
        // entry so the bind group stays valid for malformed assets.
        let skinning_size = SKINNING_ENTRY_SIZE * self.animation.len().max(1) as u64;
        let skinning_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some(&format!("{name}.skinning")),
            size: skinning_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !self.animation.is_empty() {
            queue.write_buffer(&skinning_buffer, 0, bytemuck::cast_slice(&self.animation));
        }

        let index_buffer = restpose.vi.as_ref().map(|indices| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor{
                label: Some(&format!("{name}.index")),
                size: std::mem::size_of_val(indices.as_slice()) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(indices));
            buffer
        });

        GpuSkinnedMesh {
            restpose_vertex_buffer: vec![position_buffer, attribute_buffer],
            skinning_buffer,
            index_buffer,
            vertex_count: restpose.vp.len() as u32,
            vertex_draw_count: restpose.vertex_draw_count
        }
    }
}
