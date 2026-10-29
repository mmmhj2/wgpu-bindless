use serde::{Serialize, Deserialize};

use crate::asset::asset_types::{ConcreteAssetType, GpuAssetType, static_mesh_asset::StaticMeshMaterial, vertex::{VertexBufferAnimation, VertexBufferOthers, VertexBufferPosition}};

#[derive(Serialize, Deserialize, Debug)]
pub struct SkinnedMeshAsset {
    pub position : Vec<VertexBufferPosition>,
    pub attribute : Vec<VertexBufferOthers>,
    pub animation : Vec<VertexBufferAnimation>,
    pub index  : Option<Vec<u32>>,
    pub material : StaticMeshMaterial,
    pub vertex_draw_count : u32,
    pub vertex_color_scale: [f32; 4]
}

/// GPU-resident geometry produced by uploading a [`SkinnedMeshAsset`].
///
/// The `restpose_vertex_buffer` are feed to the skinning compute buffer, and
/// then consumed by the rasterizer. These buffers are assumed to have
/// storage (UAV) usages instead of vertex usages enabled.
#[derive(Debug)]
pub struct GpuSkinnedMesh {
    pub restpose_vertex_buffer: Vec<wgpu::Buffer>,
    pub index_buffer: Option<wgpu::Buffer>,
    pub vertex_draw_count: u32
}

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
    fn upload(&self, name: &str, device: &wgpu::Device, queue: &wgpu::Queue) -> Self::GpuObjectType {
        todo!()
    }
}
