use std::num::NonZero;

use crate::asset::asset_types::texture_asset::{
    Sampler, TexelData, TextureAsset, TextureDimension, TextureFormat,
};

/// Errors produced while building a renderer-layout MRAO texture from glTF images.
#[derive(Debug, thiserror::Error)]
pub enum GltfMraoError {
    #[error("unsupported glTF image format for MRAO import: {0:?}")]
    UnsupportedFormat(gltf::image::Format),
    #[error("texture supplied for MRAO generation is invalid")]
    InvalidTexture,
    #[error("texture data length {actual} does not match {width}x{height} with {channels} channels")]
    TexelSizeMismatch {
        width: u32,
        height: u32,
        channels: usize,
        actual: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum ChannelFormat {
    U8,
    U16,
    F32,
}

impl ChannelFormat {
    fn bytes_per_channel(self) -> usize {
        match self {
            ChannelFormat::U8 => 1,
            ChannelFormat::U16 => 2,
            ChannelFormat::F32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ChannelCount {
    One,
    Two,
    Four,
}

impl ChannelCount {
    fn count(self) -> usize {
        match self {
            ChannelCount::One => 1,
            ChannelCount::Two => 2,
            ChannelCount::Four => 4,
        }
    }
}

/// Borrowed, format-normalized view over a decoded glTF image.
///
/// Internal to the glTF importer.
pub(super) struct GltfImageView<'a> {
    pub format: ChannelFormat,
    pub channels: ChannelCount,
    pub width: NonZero<u32>,
    pub height: NonZero<u32>,
    pub texels: &'a [u8],
}

impl<'a> GltfImageView<'a> {
    pub fn from_gltf(image: &'a gltf::image::Data) -> Result<Self, GltfMraoError> {
        let (format, channels) = match image.format {
            gltf::image::Format::R8 => (ChannelFormat::U8, ChannelCount::One),
            gltf::image::Format::R8G8 => (ChannelFormat::U8, ChannelCount::Two),
            gltf::image::Format::R8G8B8A8 => (ChannelFormat::U8, ChannelCount::Four),
            gltf::image::Format::R16 => (ChannelFormat::U16, ChannelCount::One),
            gltf::image::Format::R16G16 => (ChannelFormat::U16, ChannelCount::Two),
            gltf::image::Format::R16G16B16A16 => (ChannelFormat::U16, ChannelCount::Four),
            gltf::image::Format::R32G32B32A32FLOAT => (ChannelFormat::F32, ChannelCount::Four),
            // WGPU does not support RGB textures by default, and the existing
            // texture import path does not support them either.
            other => return Err(GltfMraoError::UnsupportedFormat(other)),
        };

        Self::new(format, channels, image.width, image.height, &image.pixels)
    }

    fn new(
        format: ChannelFormat,
        channels: ChannelCount,
        width: u32,
        height: u32,
        texels: &'a [u8],
    ) -> Result<Self, GltfMraoError> {
        let width = NonZero::new(width).ok_or(GltfMraoError::InvalidTexture)?;
        let height = NonZero::new(height).ok_or(GltfMraoError::InvalidTexture)?;

        let expected_len = width.get() as u64
            * height.get() as u64
            * channels.count() as u64
            * format.bytes_per_channel() as u64;
        if expected_len != texels.len() as u64 {
            return Err(GltfMraoError::TexelSizeMismatch {
                width: width.get(),
                height: height.get(),
                channels: channels.count(),
                actual: texels.len(),
            });
        }

        Ok(Self {
            format,
            channels,
            width,
            height,
            texels,
        })
    }

    fn channel_or(&self, x: u32, y: u32, channel: usize, default: f32) -> f32 {
        if channel >= self.channels.count() {
            return default;
        }

        let pixel_index = (y as usize) * (self.width.get() as usize) + (x as usize);
        let channel_index = pixel_index * self.channels.count() + channel;
        let offset = channel_index * self.format.bytes_per_channel();

        match self.format {
            ChannelFormat::U8 => self.texels[offset] as f32 / 255.0,
            ChannelFormat::U16 => {
                let value = u16::from_ne_bytes([self.texels[offset], self.texels[offset + 1]]);
                value as f32 / 65535.0
            }
            ChannelFormat::F32 => {
                let bytes = [
                    self.texels[offset],
                    self.texels[offset + 1],
                    self.texels[offset + 2],
                    self.texels[offset + 3],
                ];
                f32::from_ne_bytes(bytes)
            }
        }
    }
}

pub(super) struct MraoData {
    pub format: TextureFormat,
    pub width: NonZero<u32>,
    pub height: NonZero<u32>,
    pub texels: Box<[u8]>,
}

/// Merge glTF metallic-roughness and occlusion sources into the renderer layout:
/// R = AO, G = metallic, B = roughness, A = 1.
pub(super) fn merge_mrao(
    mr: Option<&GltfImageView<'_>>,
    ao: Option<&GltfImageView<'_>>,
) -> Result<MraoData, GltfMraoError> {
    let (width, height) = match (mr, ao) {
        (Some(mr), _) => (mr.width, mr.height),
        (None, Some(ao)) => (ao.width, ao.height),
        (None, None) => return Err(GltfMraoError::InvalidTexture),
    };

    let output_format = match (mr.map(|m| m.format), ao.map(|a| a.format)) {
        (Some(mr_format), Some(ao_format)) => mr_format.max(ao_format),
        (Some(format), None) | (None, Some(format)) => format,
        (None, None) => unreachable!("at least one source is present"),
    };

    if let Some(mr) = mr {
        if mr.channels == ChannelCount::One {
            log::warn!(
                "metallic-roughness texture only has one channel; roughness and metallic default to 1.0"
            );
        } else if mr.channels == ChannelCount::Two {
            log::warn!(
                "metallic-roughness texture has no blue channel; metallic defaults to 1.0"
            );
        }
    }

    if let (Some(mr), Some(ao)) = (mr, ao) {
        if mr.width != ao.width || mr.height != ao.height {
            todo!("Mismatched sizes for MR and AO textures. Resampling is not implemented yet.");
        }
    }

    let target_format = match output_format {
        ChannelFormat::U8 => TextureFormat::Rgba8Unorm,
        ChannelFormat::U16 => TextureFormat::Rgba16Unorm,
        ChannelFormat::F32 => TextureFormat::Rgba32Float,
    };

    let pixel_count = width.get() as usize * height.get() as usize;
    let mut texels = Vec::with_capacity(pixel_count * 4 * output_format.bytes_per_channel());

    for y in 0..height.get() {
        for x in 0..width.get() {
            let r = ao
                .map(|ao| ao.channel_or(x, y,  0, 1.0))
                .unwrap_or(1.0);
            let g = mr
                .map(|mr| mr.channel_or(x, y, 2, 1.0))
                .unwrap_or(1.0);
            let b = mr
                .map(|mr| mr.channel_or(x, y, 1, 1.0))
                .unwrap_or(1.0);

            write_texel(&mut texels, target_format, r, g, b);
        }
    }

    Ok(MraoData {
        format: target_format,
        width,
        height,
        texels: texels.into_boxed_slice(),
    })
}

fn write_texel(texels: &mut Vec<u8>, format: TextureFormat, r: f32, g: f32, b: f32) {
    match format {
        TextureFormat::Rgba8Unorm => {
            texels.push(to_u8(r));
            texels.push(to_u8(g));
            texels.push(to_u8(b));
            texels.push(u8::MAX);
        }
        TextureFormat::Rgba16Unorm => {
            push_u16(texels, to_u16(r));
            push_u16(texels, to_u16(g));
            push_u16(texels, to_u16(b));
            push_u16(texels, u16::MAX);
        }
        TextureFormat::Rgba32Float => {
            texels.extend_from_slice(&clamp_unit(r).to_ne_bytes());
            texels.extend_from_slice(&clamp_unit(g).to_ne_bytes());
            texels.extend_from_slice(&clamp_unit(b).to_ne_bytes());
            texels.extend_from_slice(&1.0f32.to_ne_bytes());
        }
        other => unreachable!("unsupported MRAO output format: {other:?}"),
    }
}

fn clamp_unit(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn to_u8(value: f32) -> u8 {
    (clamp_unit(value) * 255.0).round() as u8
}

fn to_u16(value: f32) -> u16 {
    (clamp_unit(value) * 65535.0).round() as u16
}

fn push_u16(texels: &mut Vec<u8>, value: u16) {
    texels.extend_from_slice(&value.to_ne_bytes());
}

pub(super) fn build_mrao_texture_asset(
    mr: Option<&GltfImageView<'_>>,
    ao: Option<&GltfImageView<'_>>,
    sampler: Sampler,
) -> Result<TextureAsset, GltfMraoError> {
    let data = merge_mrao(mr, ao)?;

    Ok(TextureAsset {
        format: data.format,
        dimension: TextureDimension::D2,
        width: data.width,
        height: Some(data.height),
        depth_or_array_slice: None,
        sampler,
        texels: TexelData::Plain(data.texels),
    })
}

#[cfg(test)]
mod test {
    use super::*;

    fn view_u8(pixels: &[u8], width: u32, height: u32, channels: ChannelCount) -> GltfImageView<'_> {
        GltfImageView::new(ChannelFormat::U8, channels, width, height, pixels).unwrap()
    }

    fn mrao_texels(asset: &TextureAsset) -> &[u8] {
        match &asset.texels {
            TexelData::Plain(texels) => texels,
            TexelData::Mipmapped(_) => panic!("expected plain texel data"),
        }
    }

    #[test]
    fn can_merge_and_swizzle() {
        // GLTF Style MRAO encoding:
        let mr_source = [0u8, 40 /* Green: roughness */, 80 /* Blue: metalness */, 255];
        let ao_source = [32u8 /* Red: occlusion */, 0, 0, 255];
        let mr = view_u8(&mr_source, 1, 1, ChannelCount::Four);
        let ao = view_u8(&ao_source, 1, 1, ChannelCount::Four);

        let asset = build_mrao_texture_asset(Some(&mr), Some(&ao), Sampler::default())
            .expect("MRAO build should succeed");

        assert_eq!(&mrao_texels(&asset), &[32 /* Red: occlusion */, 80 /* Green: metalness */, 40 /* Blue: roughness */, 255]);
    }

    #[test]
    fn defualts_to_one() {
        let mr_source = [0u8, 40, 80, 255];
        let mr = view_u8(&mr_source, 1, 1, ChannelCount::Four);

        let asset = build_mrao_texture_asset(Some(&mr), None, Sampler::default())
            .expect("MRAO build should succeed");

        assert_eq!(&mrao_texels(&asset), &[255, 80, 40, 255]);

        let ao_source = [32u8, 0, 0, 255];
        let ao = view_u8(&ao_source, 1, 1, ChannelCount::Four);

        let asset = build_mrao_texture_asset(None, Some(&ao), Sampler::default())
            .expect("MRAO build should succeed");

        assert_eq!(&mrao_texels(&asset), &[32, 255, 255, 255]);

        // R8G8: R is unused, G is roughness, B (metallic) is missing.
        let mr_source = [0u8, 40];
        let mr = view_u8(&mr_source, 1, 1, ChannelCount::Two);

        let asset = build_mrao_texture_asset(Some(&mr), None, Sampler::default())
            .expect("MRAO build should succeed");

        assert_eq!(&mrao_texels(&asset), &[255, 255, 40, 255]);
    }

    #[test]
    fn preserves_16_bit_precision() {
        let mr_values = [0u16, 0x4000, 0x8000, 0xffff];
        let ao_values = [0x2000u16, 0, 0, 0xffff];
        let mut mr_bytes = Vec::new();
        let mut ao_bytes = Vec::new();
        for value in mr_values {
            mr_bytes.extend_from_slice(&value.to_ne_bytes());
        }
        for value in ao_values {
            ao_bytes.extend_from_slice(&value.to_ne_bytes());
        }

        let mr = GltfImageView::new(ChannelFormat::U16, ChannelCount::Four, 1, 1, &mr_bytes).unwrap();
        let ao = GltfImageView::new(ChannelFormat::U16, ChannelCount::Four, 1, 1, &ao_bytes).unwrap();

        let asset = build_mrao_texture_asset(Some(&mr), Some(&ao), Sampler::default())
            .expect("MRAO build should succeed");

        assert_eq!(asset.format, TextureFormat::Rgba16Unorm);
        let values: Vec<u16> = mrao_texels(&asset)
            .chunks_exact(2)
            .map(|bytes| u16::from_ne_bytes([bytes[0], bytes[1]]))
            .collect();
        assert_eq!(values, vec![0x2000, 0x8000, 0x4000, 0xffff]);
    }
}
