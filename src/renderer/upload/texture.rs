use std::num::NonZero;

use crate::{
    asset::asset_types::texture_asset::{TexelData, TextureAsset},
    renderer::{
        resources::bindless::{SampledTextureHandle, SamplerHandle},
        upload::error::UploadError,
    },
};

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

/// Create the GPU texture an asset describes and write every mip level into it.
///
/// Intented for texture asset uploads, the texture gets
/// `COPY_DST | TEXTURE_BINDING` only. Render target textures must be handled
/// separately.
/// 
/// You might have to register the texture to the bindless resource manager
/// manually after its creation, if it is to be used as such.
pub fn create_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    name: &str,
    asset: &TextureAsset,
) -> Result<wgpu::Texture, UploadError> {
    let format: wgpu::TextureFormat = asset.format.into();
    let dimension: wgpu::TextureDimension = asset.dimension.into();
    let size = wgpu::Extent3d {
        width: asset.width.get(),
        height: asset.height.map_or(1, NonZero::get),
        depth_or_array_layers: asset.depth_or_array_slice.map_or(1, NonZero::get),
    };

    let mip_level_count = match &asset.texels {
        TexelData::Plain(_) => 1,
        TexelData::Mipmapped(miplevels) => miplevels.len() as u32,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(name),
        size,
        mip_level_count,
        sample_count: 1,
        dimension,
        format,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    match &asset.texels {
        TexelData::Plain(texels) => {
            let (bytes_per_row, rows_per_image) = upload_layout(format, size);
            check_texel_size(name, size, bytes_per_row, rows_per_image, texels.len())?;

            queue.write_texture(
                wgpu::TexelCopyTextureInfoBase {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                texels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(rows_per_image),
                },
                size,
            );
        },
        TexelData::Mipmapped(miplevels) => {
            let mut mip_size = size;
            for (level, data) in miplevels.iter().enumerate() {
                if mip_size.width == 0
                    || mip_size.height == 0
                    || mip_size.depth_or_array_layers == 0
                {
                    return Err(UploadError::MipChainTooLong {
                        name: name.to_string(),
                        level: level as u32,
                    });
                }

                let (bytes_per_row, rows_per_image) = upload_layout(format, mip_size);
                check_texel_size(name, mip_size, bytes_per_row, rows_per_image, data.texels().len())?;

                queue.write_texture(
                    wgpu::TexelCopyTextureInfoBase {
                        texture: &texture,
                        mip_level: level as u32,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &data.texels(),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(rows_per_image),
                    },
                    mip_size,
                );

                mip_size.width >>= 1;
                mip_size.height >>= 1;
                // A 2D array does not shrink its layer count across mips; only a
                // volume texture shrinks its depth.
                if dimension == wgpu::TextureDimension::D3 {
                    mip_size.depth_or_array_layers >>= 1;
                }
            }
        },
    }

    Ok(texture)
}

/// Number of bytes per row and rows per image for one mip level, respecting
/// block-compressed formats.
fn upload_layout(format: wgpu::TextureFormat, size: wgpu::Extent3d) -> (u32, u32) {
    let block_copy_size = format
        .block_copy_size(None)
        .expect("designated format should have a block copy size");
    let block_dimensions = format.block_dimensions();

    let bytes_per_row = block_copy_size * size.width.div_ceil(block_dimensions.0);
    let rows_per_image = size.height.div_ceil(block_dimensions.1);
    (bytes_per_row, rows_per_image)
}

/// Reject a texel payload whose length does not match the declared layout.
fn check_texel_size(
    name: &str,
    size: wgpu::Extent3d,
    bytes_per_row: u32,
    rows_per_image: u32,
    actual: usize,
) -> Result<(), UploadError> {
    let expected = (bytes_per_row as usize) * (rows_per_image as usize)
        * (size.depth_or_array_layers as usize);
    if expected != actual {
        return Err(UploadError::TexelSize {
            name: name.to_string(),
            expected,
            actual,
        });
    }
    Ok(())
}
