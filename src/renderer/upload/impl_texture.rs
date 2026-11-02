use crate::{
    asset::asset_types::texture_asset::TextureAsset,
    renderer::upload::{
        context::UploadContext, error::UploadError, gpu_type::GpuAssetType, texture::create_texture,
    },
};

impl GpuAssetType for TextureAsset {
    type GpuObjectType = wgpu::Texture;

    /// Create the texture the asset describes and write every mip level (if exists).
    ///
    /// Simply delegate to [`create_texture`] for resource uploading.
    /// Use [`UploadContext::upload_texture`] when a bindless slot and a sampler
    /// are needed.
    fn upload(
        &self,
        name: &str,
        ctx: &mut UploadContext<'_>,
    ) -> Result<Self::GpuObjectType, UploadError> {
        create_texture(ctx.device(), ctx.queue(), name, self)
    }
}
