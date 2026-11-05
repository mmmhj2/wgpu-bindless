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

impl GpuAssetType for VertexMorphTargetAsset {
    /// TODO: the upload produces a `GpuMorphTargets` holding the layout
    /// buffer (masks and offsets), the packed delta buffer and the optional
    /// base-weight texture.
    type GpuObjectType = ();

    fn upload(
        &self,
        _name: &str,
        _ctx: &mut UploadContext<'_>,
    ) -> Result<Self::GpuObjectType, UploadError> {
        todo!("upload morph target layout / deltas / base weights")
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
