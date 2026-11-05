use crate::{
    asset::asset_types::animation_asset::{SkinnedAnimationAsset, VertexMorphAnimationAsset, VertexMorphTargetAsset},
    renderer::upload::{context::UploadContext, error::UploadError, gpu_type::GpuAssetType},
};

impl GpuAssetType for SkinnedAnimationAsset {
    type GpuObjectType = wgpu::Texture;

    /// Upload the baked animation texture onto the GPU.
    ///
    /// Joint animations are baked into a 2D texture at import time, one row per
    /// frame.
    ///
    /// The texture has the format of RGBA16 by default. If bandwidth turns out to
    /// be a problem, RGBA8 may be used in the future. All transform data is baked
    /// *in model space*, to be used directly in the compute shader.
    fn upload(
        &self,
        name: &str,
        ctx: &mut UploadContext<'_>,
    ) -> Result<Self::GpuObjectType, UploadError> {
        self.texture.upload(name, ctx)
    }
}

/// TODO: maybe move to util module
pub struct BufferWithOffset {
    pub buffer: wgpu::Buffer,
    pub offset: usize
}

pub enum MorphTargetBuffer {
    Position(BufferWithOffset),
    Normal(BufferWithOffset),
    Tangent(BufferWithOffset)
}

impl GpuAssetType for VertexMorphTargetAsset {
    /// Storage buffers for vertex morph targets
    type GpuObjectType = Vec<MorphTargetBuffer>;

    fn upload(
        &self,
        name: &str,
        ctx: &mut UploadContext<'_>,
    ) -> Result<Self::GpuObjectType, UploadError> {
        todo!()
    }
}

impl GpuAssetType for VertexMorphAnimationAsset {
    type GpuObjectType = wgpu::Texture;

    fn upload(
        &self,
        name: &str,
        ctx: &mut UploadContext<'_>,
    ) -> Result<Self::GpuObjectType, UploadError> {
        self.texture.upload(name, ctx)
    }
}
