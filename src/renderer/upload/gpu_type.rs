use crate::{
    asset::asset_types::ConcreteAssetType,
    renderer::upload::{context::UploadContext, error::UploadError},
};

/// An asset type that can be turned into a GPU object.
pub trait GpuAssetType: ConcreteAssetType {
    /// The GPU-resident result of uploading this asset.
    type GpuObjectType;

    /// Create the GPU object for this asset, writing its payload with `ctx`.
    ///
    /// `name` is the asset name; it is expected to be used as the resource label
    /// so GPU captures stay readable, and may be used to derive per-stream labels
    /// (e.g. `<name>.position`).
    fn upload(
        &self,
        name: &str,
        ctx: &mut UploadContext<'_>,
    ) -> Result<Self::GpuObjectType, UploadError>;
}
