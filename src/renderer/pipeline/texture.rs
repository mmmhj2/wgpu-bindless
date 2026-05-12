use crate::renderer::device_interface::DeviceInterface;


/// A simple wrapper around wgpu::Texture
pub struct Texture {
    inner:  wgpu::Texture
}

pub enum TextureType {
    Linear,
    ColorSrgb,
    Normal
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

    pub fn extract_texture_info<'a>(
        texture: &gltf::Texture,
        images: &'a Vec<gltf::image::Data>,
        ttype: TextureType
    ) -> (wgpu::TextureDescriptor<'static>, &'a [u8]) {
        let image = &images[texture.index()];

        fn match_texture_format(f: gltf::image::Format) -> Option<wgpu::TextureFormat> {
            use wgpu::TextureFormat;
            match f {
                gltf::image::Format::R8 => Some(TextureFormat::R8Unorm),
                gltf::image::Format::R8G8 => Some(TextureFormat::Rg8Unorm),
                gltf::image::Format::R8G8B8 => None,
                gltf::image::Format::R8G8B8A8 => Some(TextureFormat::Rgba8Unorm),
                gltf::image::Format::R16 => Some(TextureFormat::R16Unorm),
                gltf::image::Format::R16G16 => Some(TextureFormat::Rg16Unorm),
                gltf::image::Format::R16G16B16 => None,
                gltf::image::Format::R16G16B16A16 => Some(TextureFormat::Rgba16Unorm),
                gltf::image::Format::R32G32B32FLOAT => None,
                gltf::image::Format::R32G32B32A32FLOAT => Some(TextureFormat::Rgba32Float),
            }
        }

        let converted_format = match_texture_format(image.format).expect("Unsupported texture format. 3-channel textures are not supported by WGPU.");

        let final_format = match ttype {
            TextureType::Linear => { converted_format },
            TextureType::ColorSrgb => { converted_format.add_srgb_suffix() },
            TextureType::Normal => { 
                match converted_format {
                    wgpu::TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Snorm,
                    _ => converted_format
                }
             },
        };

        let descriptor: wgpu::TextureDescriptor = wgpu::TextureDescriptor{
            label: None,
            size: wgpu::Extent3d { width: image.width, height: image.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: final_format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        (descriptor, &image.pixels)
    }

    pub fn create_from_gltf(
        di: &DeviceInterface,
        texture: &gltf::Texture,
        buffers: &Vec<gltf::buffer::Data>,
        images: &Vec<gltf::image::Data>
    ) -> Self {
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_color_texture_import() {
        let (document, buffers, images) = gltf::import("resource/test_two_cubes.glb").expect("Failed to import GLB file.");

        let mesh = document.meshes()
            .find(|x| x.name().unwrap_or_default() == "cube_textured")
            .expect("Cannot find cube_textured.");

        let primitive = mesh.primitives().next().expect("cube_textured has not primitives.");
        let material = primitive.material();

        let (desc, raw_data) = Texture::extract_texture_info(
            &material.pbr_metallic_roughness().base_color_texture().expect("cube_textured has no base color texture.").texture(),
            &images, TextureType::ColorSrgb
        );
        assert_eq!(desc.size.width, 32);
        assert_eq!(desc.size.height, 32);
        assert_eq!(desc.size.depth_or_array_layers, 1);
        assert_eq!(desc.format, wgpu::TextureFormat::Rgba8UnormSrgb);
    }

    #[test]
    fn test_normal_texture_import() {
        let (document, buffers, images) = gltf::import("resource/test_two_cubes.glb").expect("Failed to import GLB file.");

        let mesh = document.meshes()
            .find(|x| x.name().unwrap_or_default() == "cube_textured")
            .expect("Cannot find cube_textured.");

        let primitive = mesh.primitives().next().expect("cube_textured has not primitives.");
        let material = primitive.material();

        let (desc, raw_data) = Texture::extract_texture_info(
            &material.normal_texture().expect("cube_textured has no normal map texture.").texture(),
            &images, TextureType::Normal
        );
        assert_eq!(desc.size.width, 32);
        assert_eq!(desc.size.height, 32);
        assert_eq!(desc.size.depth_or_array_layers, 1);
        assert_eq!(desc.format, wgpu::TextureFormat::Rgba8Snorm);

        println!("{} {} {} {}", raw_data[0], raw_data[1], raw_data[2], raw_data[3]);
        assert_eq!(raw_data[0], 128);
        assert_eq!(raw_data[1], 128);
        assert_eq!(raw_data[2], 255);
    }
}
