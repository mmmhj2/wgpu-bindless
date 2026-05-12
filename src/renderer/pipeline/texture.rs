use crate::renderer::device_interface::DeviceInterface;


/// A simple wrapper around wgpu::Texture
pub struct Texture {
    inner:  wgpu::Texture
}

impl From<Texture> for wgpu::Texture {
    fn from(value: Texture) -> Self {
        value.inner
    }
}

impl Texture {

    const PACKED_TEXEL_LAYOUT: wgpu::TexelCopyBufferLayout = wgpu::TexelCopyBufferLayout {
        offset: 0, bytes_per_row: None, rows_per_image: None
    };

    const SINGLE_TEXEL_EXTENT: wgpu::Extent3d = wgpu::Extent3d {
        width: 1, height: 1, depth_or_array_layers: 1
    };

    pub fn create_from_gltf() -> Self {
        todo!()
    }

    /// Create a texture from a single texel.
    /// 
    /// This convenient method creates a texture from specification with a single texel.
    /// It first create the texture instance, and then issue a copy command via `write_texture()`
    pub fn create_from_single_color_texel(
        di : &DeviceInterface,
        texel: &[u8],
        dimension: wgpu::TextureDimension,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        label: Option<&str>
    ) -> Self {
        assert!(usage.contains(wgpu::TextureUsages::COPY_DST));
        assert!(
            format.block_dimensions() == (1, 1),
            "Block-compressed texture formats are not supported."
        );
        assert_eq!(
            texel.len(),
            format.block_copy_size(None).expect("Non-color aspects are not supported.") as usize,
            "Texel size you supplied is not equal to the texel size of the format."
        );

        let texture = di.get_device().create_texture(&wgpu::TextureDescriptor{
            label,
            size: Self::SINGLE_TEXEL_EXTENT,
            mip_level_count: 1,
            sample_count: 1,
            dimension,
            format,
            usage,
            view_formats: &[],
        });

        let texture_copy_info = wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        };
        di.get_queue().write_texture(
            texture_copy_info,
            texel,
            Self::PACKED_TEXEL_LAYOUT,
            Self::SINGLE_TEXEL_EXTENT
        );

        Self {inner: texture}
    }
}
