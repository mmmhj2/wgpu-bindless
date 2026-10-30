//! Texture, sampler and MRAO import.

use std::num::NonZeroU32;

use crate::asset::asset_types::texture_asset::{
    AddressMode, FilterMode, Sampler, TexelData, TextureAsset, TextureDimension::D2, TextureFormat,
};

use super::common::{GltfImageView, GltfImportError};
use super::importer::GltfImporter;
use super::mrao::build_mrao_texture_asset;

/// Source textures combined into one renderer-layout MRAO texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct MraoKey {
    pub(super) mr: Option<usize>,
    pub(super) ao: Option<usize>,
}

impl Into<AddressMode> for gltf::texture::WrappingMode {
    fn into(self) -> AddressMode {
        match self {
            gltf::texture::WrappingMode::ClampToEdge => AddressMode::ClampToEdge,
            gltf::texture::WrappingMode::MirroredRepeat => AddressMode::MirrorRepeat,
            gltf::texture::WrappingMode::Repeat => AddressMode::Repeat,
        }
    }
}

impl GltfImporter {
    pub(super) fn import_from_gltf_mrao(
        document: &gltf::Document,
        images: &[gltf::image::Data],
        key: MraoKey,
    ) -> Result<TextureAsset, GltfImportError> {
        let mr_texture = key.mr
            .map(|index| document.textures().nth(index).ok_or(GltfImportError::IndexOutOfRange(index)))
            .transpose()?;
        let ao_texture = key.ao
            .map(|index| document.textures().nth(index).ok_or(GltfImportError::IndexOutOfRange(index)))
            .transpose()?;

        let mr_image = mr_texture.as_ref()
            .map(|texture| {
                let image_index = texture.source().index();
                images.get(image_index).ok_or(GltfImportError::IndexOutOfRange(image_index))
            })
            .transpose()?;
        let ao_image = ao_texture.as_ref()
            .map(|texture| {
                let image_index = texture.source().index();
                images.get(image_index).ok_or(GltfImportError::IndexOutOfRange(image_index))
            })
            .transpose()?;

        let mr_view = mr_image.map(GltfImageView::from_gltf).transpose()?;
        let ao_view = ao_image.map(GltfImageView::from_gltf).transpose()?;

        let sampler = match (&mr_texture, &ao_texture) {
            (Some(mr), Some(ao)) => {
                if !samplers_equal(&mr.sampler(), &ao.sampler()) {
                    log::warn!(
                        "metallic-roughness and occlusion textures use different samplers; using the metallic-roughness sampler"
                    );
                }
                Self::import_from_gltf_sampler(&mr.sampler())
            }
            (Some(mr), None) => Self::import_from_gltf_sampler(&mr.sampler()),
            (None, Some(ao)) => Self::import_from_gltf_sampler(&ao.sampler()),
            (None, None) => return Err(GltfImportError::BadMraoTexture(super::common::GltfImageImportError::InvalidTexture)),
        };

        Ok(build_mrao_texture_asset(mr_view.as_ref(), ao_view.as_ref(), sampler)?)
    }

    pub fn import_from_gltf_sampler_filter_mode(sampler: &gltf::texture::Sampler) -> (FilterMode, FilterMode, FilterMode) {
        let mipmap = match sampler.min_filter() {
            Some(x) => match x {
                gltf::texture::MinFilter::Nearest
                 | gltf::texture::MinFilter::Linear
                 | gltf::texture::MinFilter::NearestMipmapNearest
                 | gltf::texture::MinFilter::LinearMipmapNearest  => FilterMode::Nearest,
                _ => FilterMode::Linear,
            },
            None => FilterMode::Nearest,
        };

        let mag = match sampler.mag_filter() {
            Some(x) => match x {
                gltf::texture::MagFilter::Nearest => FilterMode::Nearest,
                gltf::texture::MagFilter::Linear => FilterMode::Linear,
            },
            None => FilterMode::Nearest,
        };

        let min = match sampler.min_filter() {
            Some(x) => match x {
                gltf::texture::MinFilter::Nearest
                 | gltf::texture::MinFilter::NearestMipmapNearest
                 | gltf::texture::MinFilter::NearestMipmapLinear => FilterMode::Nearest,
                _ => FilterMode::Linear
            },
            None => FilterMode::Nearest,
        };

        (mag, min, mipmap)
    }

    fn import_from_gltf_sampler(sampler: &gltf::texture::Sampler) -> Sampler {
        Sampler { 
            address_mode: (sampler.wrap_s().into(), sampler.wrap_t().into(), AddressMode::ClampToEdge),
            filter: Self::import_from_gltf_sampler_filter_mode(sampler)
        }
    }

    /// Map a glTF texel format onto an asset texture format.
    ///
    /// The sRGB suffix only applies to 8-bit color textures; all other formats
    /// (including normal maps) are imported as linear formats.
    fn match_texture_format(f: gltf::image::Format, is_srgb: bool) -> Result<TextureFormat, GltfImportError> {
        Ok(match f {
            gltf::image::Format::R8 => TextureFormat::R8Unorm,
            gltf::image::Format::R8G8 => TextureFormat::Rg8Unorm,
            gltf::image::Format::R8G8B8A8 => if is_srgb {TextureFormat::Rgba8Srgb} else {TextureFormat::Rgba8Unorm},
            gltf::image::Format::R16 => TextureFormat::R16Unorm,
            gltf::image::Format::R16G16 => TextureFormat::Rg16Unorm,
            gltf::image::Format::R16G16B16A16 => TextureFormat::Rgba16Unorm,
            gltf::image::Format::R32G32B32A32FLOAT => TextureFormat::Rgba32Float,
            _ => return Err(GltfImportError::UnsupportedTextureFormat(f))
        })
    }

    fn recondition_rgb8_texture(width: NonZeroU32, height: NonZeroU32, pixels: &Vec<u8>) -> Vec<u8> {
        let mut ret = Vec::with_capacity((width.get() * height.get() * 4) as usize);
        assert_eq!(ret.capacity(), pixels.len() / 3 * 4);

        for i in (0..pixels.len()).step_by(3) {
            ret.push(pixels[i]);
            ret.push(pixels[i+1]);
            ret.push(pixels[i+2]);
            ret.push(255);
        };

        ret
    }

    pub fn import_from_gltf_texture(
        texture: &gltf::Texture,
        images: &[gltf::image::Data],
        is_srgb: bool
    ) -> Result<TextureAsset, GltfImportError> {
        // The format must be read from the image the texture sources from,
        // not from the texture's own index into the image list.
        let image = images.get(texture.source().index())
            .ok_or(GltfImportError::IndexOutOfRange(texture.source().index()))?;
        let (converted_format, texels) = match image.format {
            gltf::image::Format::R8G8B8 => {
                log::info!("Reconditioning RGB8 texture");
                (
                    if is_srgb {TextureFormat::Rgba8Srgb} else {TextureFormat::Rgba8Unorm},
                    &Self::recondition_rgb8_texture(
                        image.width.try_into().unwrap(),
                        image.height.try_into().unwrap(),
                        &image.pixels)
                )
            },
            _ => (
                Self::match_texture_format(image.format, is_srgb)?,
                &image.pixels
            )
        };

        Ok(TextureAsset {
            format: converted_format,
            dimension: D2,
            width: image.width.try_into().map_err(|_| GltfImportError::ZeroDimension)?,
            height: Some(image.height.try_into().map_err(|_| GltfImportError::ZeroDimension)?),
            depth_or_array_slice: None,
            sampler: Self::import_from_gltf_sampler(&texture.sampler()),
            // One gltf image can correspond to multiple textures with different sampler.
            // So we must clone its data for safety.
            texels: TexelData::Plain(texels.clone().into_boxed_slice()),
        })
    }
}

fn samplers_equal(a: &gltf::texture::Sampler, b: &gltf::texture::Sampler) -> bool {
    a.wrap_s() == b.wrap_s()
        && a.wrap_t() == b.wrap_t()
        && a.mag_filter() == b.mag_filter()
        && a.min_filter() == b.min_filter()
}
