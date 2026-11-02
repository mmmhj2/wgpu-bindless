use crate::renderer::resources::bindless::{SampledTextureHandle, SamplerHandle};

/// A texture uploaded from an asset, together with its bindless slots.
///
/// TODO: implement real RAII with this resource handle. Do need to carefully
/// consider lifetime problems, as it basically re-introduces what is eliminated
/// by a bindless approach.
#[derive(Debug)]
pub struct TextureBinding {
    /// The uploaded texture.
    ///
    /// Dropping this handle is safe while the binding is still registered as
    /// the bindless table owns its own [`wgpu::TextureView`], which keeps the
    /// underlying texture alive.
    pub texture: wgpu::Texture,
    /// The view registered in the bindless table.
    pub view: wgpu::TextureView,
    /// Bindless slot of [`Self::view`].
    pub handle: SampledTextureHandle,
    /// Bindless slot of the sampler described by the asset.
    pub sampler: SamplerHandle,
}
