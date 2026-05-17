use image::EncodableLayout;

use crate::renderer::device_interface::DeviceInterface;


#[derive(Debug)]
pub enum TextureImportError {
    FailedToOpenImage(std::io::Error),
    FailedToDecodeImage(image::ImageError),
    /// The texture has only three channels, which is not supported by wGPU.
    UnsupportedFormat,
    /// Texture has less channels than expected.
    /// For example, color and normal textures should have at least 3 channels.
    ChannelNotSufficient,
    /// Texel size not correct.
    /// The texel buffer obtained from GLTF is either too large or too small for the texture.
    UnfitTexelDataSize,
    /// A 2D texture array should be created,
    /// but cannot due to different images having mismatched width, height or formats.
    IncompatibleArrayFormat
}
impl std::fmt::Display for TextureImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextureImportError::FailedToOpenImage(e) => write!(f, "failed to open image: {}", e),
            TextureImportError::FailedToDecodeImage(e) => write!(f, "failed to decode image: {}", e),
            TextureImportError::UnsupportedFormat => write!(f, "texture has an unsupported texel format. For example, 3 channel RGB textures are not supported for GLTF import."),
            TextureImportError::ChannelNotSufficient => write!(f, "texture has less channels than expected."),
            TextureImportError::UnfitTexelDataSize => write!(f, "texel buffer size is not correct."),
            TextureImportError::IncompatibleArrayFormat => write!(f, "a 2D texture array should be created, but cannot due to different images having mismatched width, height or formats."),
        }
    }
}
impl std::error::Error for TextureImportError {}

struct ImportedTextureDescriptor {
    pub data: Vec<u8>,
    pub format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32
}

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
impl From<Texture> for wgpu::TextureView {
    fn from(value: Texture) -> Self {
        value.inner.create_view(&Default::default())
    }
}

impl Texture {
    const PACKED_TEXEL_LAYOUT: wgpu::TexelCopyBufferLayout = wgpu::TexelCopyBufferLayout {
        offset: 0, bytes_per_row: None, rows_per_image: None
    };
    const SINGLE_TEXEL_EXTENT: wgpu::Extent3d = wgpu::Extent3d {
        width: 1, height: 1, depth_or_array_layers: 1
    };

    fn load_from_file<P>(
        path: P
    ) -> Result<ImportedTextureDescriptor, TextureImportError> where P: AsRef<std::path::Path> {
        let image = image::ImageReader::open(path);
        if let Err(e) = image {
            return Err(TextureImportError::FailedToOpenImage(e));
        }
        let image = image.unwrap().decode();
        if let Err(e) = image {
            return Err(TextureImportError::FailedToDecodeImage(e));
        }
        let image = image.unwrap();
        let width = image.width();
        let height = image.height();

        let format = match image.color() {
            image::ColorType::Rgb8 => wgpu::TextureFormat::Rgba8Unorm,
            image::ColorType::Rgba8 => wgpu::TextureFormat::Rgba8Unorm,
            image::ColorType::Rgb16 => wgpu::TextureFormat::Rgba16Unorm,
            image::ColorType::Rgba16 => wgpu::TextureFormat::Rgba16Unorm,
            image::ColorType::Rgb32F => wgpu::TextureFormat::Rgba32Float,
            image::ColorType::Rgba32F => wgpu::TextureFormat::Rgba32Float,
            _ => return Err(TextureImportError::UnsupportedFormat),
        };

        let data= match image.color() {
            image::ColorType::Rgb8 | image::ColorType::Rgba8 => image.into_rgba8().into_vec(),
            // TODO: eliminate the extra copy here.
            image::ColorType::Rgb16 | image::ColorType::Rgba16 => image.into_rgba16().as_bytes().to_vec(),
            image::ColorType::Rgb32F | image::ColorType::Rgba32F => image.into_rgba32f().as_bytes().to_vec(),
            _ => return Err(TextureImportError::UnsupportedFormat),
        };

        return Ok(ImportedTextureDescriptor { data, format, width, height });
    }
    
    /// Create a texture from a file path.
    /// 
    /// The texture will be created, and a new write texture command will be recorded on the queue.
    /// The format of the texture will be automatically detected.
    /// If the texture has only RGB channels, an extra alpha channel will be added.
    /// 
    /// The texture_type specified only affects intepretation of texel data in case of 8-bit per channel texels.
    /// No gamma correction is performed.
    pub fn create_from_file<P>(
        di: &DeviceInterface,
        path: P,
        texture_type: TextureType
    ) -> Result<Self, TextureImportError> where P: AsRef<std::path::Path> {
        let idesc = Self::load_from_file(path)?;

        let descriptor = wgpu::TextureDescriptor{
            label: None,
            size: wgpu::Extent3d {
                width: idesc.width,
                height: idesc.height,
                depth_or_array_layers: 1
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: match texture_type {
                TextureType::ColorSrgb => idesc.format.add_srgb_suffix(),
                _ => idesc.format,
            },
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[]
        };

        let texture = di.get_device().create_texture(&descriptor);

        di.get_queue().write_texture(
            wgpu::TexelCopyTextureInfo{
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            idesc.data.as_slice(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(
                    descriptor.format.block_copy_size(None).expect(
                        "Designated format does not have a copy size."
                    ) * descriptor.size.width),
                rows_per_image: None
            },
            wgpu::Extent3d { width: idesc.width, height: idesc.height, depth_or_array_layers: 1 }
        );

        Ok(Self{ inner: texture })
    }

    pub fn create_from_file_array_rgba8<P>(
        di: &DeviceInterface,
        path: &[P],
        texture_type: TextureType
    ) -> Result<Self, TextureImportError> where P: AsRef<std::path::Path> {
        let descriptor_slice: Vec<Result<_, _>> = path.iter().map(|p| Self::load_from_file(p)).collect();
        let descriptor_slice: Result<_, _> = descriptor_slice.into_iter().collect();
        let descriptor_slice: Vec<_> = descriptor_slice?;
        
        let format = descriptor_slice[0].format;
        let width = descriptor_slice[0].width;
        let height = descriptor_slice[0].height;

        for desc in &descriptor_slice {
            if !(width == desc.width && height == desc.height && format == desc.format) {
                return Err(TextureImportError::IncompatibleArrayFormat);
            }
        }

        let descriptor = wgpu::TextureDescriptor{
            label: None,
            size: wgpu::Extent3d { width, height, depth_or_array_layers: descriptor_slice.len() as u32 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: match texture_type {
                TextureType::ColorSrgb => format.add_srgb_suffix(),
                _ => format,
            },
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[]
        };
        let texture = di.get_device().create_texture(&descriptor);

        for (i, desc) in descriptor_slice.iter().enumerate() {
            di.get_queue().write_texture(
                wgpu::TexelCopyTextureInfo{
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d{x: 0, y: 0, z: i as u32},
                    aspect: wgpu::TextureAspect::All,
                },
                desc.data.as_slice(),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        descriptor.format.block_copy_size(None).expect(
                            "Designated format does not have a copy size."
                        ) * descriptor.size.width),
                    rows_per_image: None
                },
                wgpu::Extent3d { width, height, depth_or_array_layers: 1 }
            )
        }

        Ok(Self{ inner: texture })
    }

    /// Extract texture descriptor from a GLTF texture.
    /// 
    /// Returns a wgpu texture descriptor, and a u8 slice containing texel data.
    /// The texture descriptor will have *no* views attached to and a usage of `COPY_DST` and `TEXTURE_BINDING`.
    pub fn extract_texture_info<'a>(
        texture: &gltf::Texture,
        images: &'a Vec<gltf::image::Data>,
        ttype: TextureType
    ) -> Result<(wgpu::TextureDescriptor<'static>, &'a [u8]), TextureImportError> {
        let image = &images[texture.index()];

        fn match_texture_format(f: gltf::image::Format) -> Result<wgpu::TextureFormat, TextureImportError> {
            use wgpu::TextureFormat;
            match f {
                gltf::image::Format::R8 => Ok(TextureFormat::R8Unorm),
                gltf::image::Format::R8G8 => Ok(TextureFormat::Rg8Unorm),
                gltf::image::Format::R8G8B8 => Err(TextureImportError::UnsupportedFormat),
                gltf::image::Format::R8G8B8A8 => Ok(TextureFormat::Rgba8Unorm),
                gltf::image::Format::R16 => Ok(TextureFormat::R16Unorm),
                gltf::image::Format::R16G16 => Ok(TextureFormat::Rg16Unorm),
                gltf::image::Format::R16G16B16 => Err(TextureImportError::UnsupportedFormat),
                gltf::image::Format::R16G16B16A16 => Ok(TextureFormat::Rgba16Unorm),
                gltf::image::Format::R32G32B32FLOAT => Err(TextureImportError::UnsupportedFormat),
                gltf::image::Format::R32G32B32A32FLOAT => Ok(TextureFormat::Rgba32Float),
            }
        }

        let converted_format = match_texture_format(image.format)?;

        let final_format = match ttype {
            TextureType::Linear => { Ok(converted_format) },
            TextureType::ColorSrgb => { Ok(converted_format.add_srgb_suffix()) },
            TextureType::Normal => { 
                match converted_format {
                    x @ (wgpu::TextureFormat::Rgba8Unorm 
                    | wgpu::TextureFormat::Rgba16Unorm 
                    | wgpu::TextureFormat::Rgba32Float) => Ok(x),
                    _ => Err(TextureImportError::ChannelNotSufficient)
                }
             },
        }?;

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

        // Check whether texel size matches.
        let expected_pixel_size = final_format.block_copy_size(None).expect("") as usize * image.width as usize * image.height as usize;
        if image.pixels.len() != expected_pixel_size {
            return Err(TextureImportError::UnfitTexelDataSize);
        }

        Ok((descriptor, &image.pixels))
    }

    /// Create a texture from a GLTF file.
    pub fn create_from_gltf(
        di: &DeviceInterface,
        texture_type: TextureType,
        texture: &gltf::Texture,
        images: &Vec<gltf::image::Data>,
        label: Option<&str>
    ) -> Result<Self, TextureImportError> {
        let (descriptor, pixels) = Texture::extract_texture_info(texture, images, texture_type)?;
        let texture = di.get_device().create_texture(
            &wgpu::TextureDescriptor {
                label,
                ..descriptor
            }
        );

        let texture_copy_info = wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        };
        di.get_queue().write_texture(
            texture_copy_info,
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(
                    descriptor.format.block_copy_size(None).expect(
                        "Designated format does not have a copy size."
                    ) * descriptor.size.width),
                rows_per_image: None
            },
            descriptor.size
        );

        Ok(Self{inner: texture})
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
        let (document, _, images) = gltf::import("resource/test_two_cubes.glb").expect("Failed to import GLB file.");

        let mesh = document.meshes()
            .find(|x| x.name().unwrap_or_default() == "cube_textured")
            .expect("Cannot find cube_textured.");

        let primitive = mesh.primitives().next().expect("cube_textured has not primitives.");
        let material = primitive.material();

        let (desc, _) = Texture::extract_texture_info(
            &material.pbr_metallic_roughness().base_color_texture().expect("cube_textured has no base color texture.").texture(),
            &images, TextureType::ColorSrgb
        ).expect("Texture import unsuccessful.");
        assert_eq!(desc.size.width, 32);
        assert_eq!(desc.size.height, 32);
        assert_eq!(desc.size.depth_or_array_layers, 1);
        assert_eq!(desc.format, wgpu::TextureFormat::Rgba8UnormSrgb);
    }

    #[test]
    fn test_normal_texture_import() {
        let (document, _, images) = gltf::import("resource/test_two_cubes.glb").expect("Failed to import GLB file.");

        let mesh = document.meshes()
            .find(|x| x.name().unwrap_or_default() == "cube_textured")
            .expect("Cannot find cube_textured.");

        let primitive = mesh.primitives().next().expect("cube_textured has not primitives.");
        let material = primitive.material();

        let (desc, raw_data) = Texture::extract_texture_info(
            &material.normal_texture().expect("cube_textured has no normal map texture.").texture(),
            &images, TextureType::Normal
        ).expect("Texture import unsuccessful.");
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
