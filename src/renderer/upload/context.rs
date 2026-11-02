use crate::{
    asset::asset_types::texture_asset::TextureAsset,
    renderer::{
        resources::{
            bindless::{BindlessResources, SamplerHandle},
            sampler::SamplerDescription,
        },
        upload::{
            buffer::upload_buffer, error::UploadError, gpu_type::GpuAssetType,
            texture::TextureBinding,
        },
    },
};

/// Borrowed handle to the GPU resources an upload needs.
pub struct UploadContext<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    bindless: &'a mut BindlessResources,
}

impl<'a> UploadContext<'a> {
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        bindless: &'a mut BindlessResources,
    ) -> Self {
        Self {
            device,
            queue,
            bindless,
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        self.queue
    }

    /// Read only access to bindless tables this context allocates from.
    ///
    /// Writes to the tables should be delegated to
    /// [`Self::upload_texture`]/[`Self::upload_sampler`] so that slot bookkeeping
    /// and bind-group invalidation stay in one place.
    pub fn bindless(&self) -> &BindlessResources {
        self.bindless
    }

    /// Upload a one-shot buffer. See [`upload_buffer`] for the rationale.
    pub fn upload_buffer<T: bytemuck::Pod>(
        &self,
        label: &str,
        usage: wgpu::BufferUsages,
        data: &[T],
    ) -> Result<wgpu::Buffer, UploadError> {
        upload_buffer(self.device, self.queue, label, usage, data)
    }

    /// Register a sampler in the bindless table.
    ///
    /// Identical descriptions return the same handle, so passes that need the
    /// same filtering share one GPU sampler.
    pub fn upload_sampler(
        &mut self,
        desc: SamplerDescription,
    ) -> Result<SamplerHandle, UploadError> {
        Ok(self.bindless.create_sampler(self.device, desc)?)
    }

    /// Upload a texture asset and register it (and its sampler) in the bindless
    /// table.
    ///
    /// Create the texture and the view, and allocate the bindless slot and
    /// register the sampler the asset describes in that order.
    /// 
    /// NOTE: The bindless table only holds `texture_2d_array` bindings, so this
    /// upload path creates only 2D views to textures. A pass that needs other
    /// views (the skybox) must manage on its own.
    pub fn upload_texture(
        &mut self,
        name: &str,
        asset: &TextureAsset,
    ) -> Result<TextureBinding, UploadError> {
        let texture = asset.upload(name, self)?;

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let handle = self.bindless.create_sampled_texture(view.clone())?;
        let sampler = self
            .bindless
            .create_sampler(self.device, asset.sampler.clone().into())?;

        Ok(TextureBinding {
            texture,
            view,
            handle,
            sampler,
        })
    }
}
