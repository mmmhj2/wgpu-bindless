use std::{num::NonZero, path::PathBuf};

use crate::asset::importer::vertex_reconditioner::ReconditionerError;


/// Errors produced while building a texture from glTF images.
#[derive(Debug, thiserror::Error)]
pub enum GltfImageImportError {
    #[error("unsupported glTF image format for MRAO import: {0:?}")]
    UnsupportedFormat(gltf::image::Format),
    #[error("texture supplied for reconditioning is invalid")]
    InvalidTexture,
    #[error("texture data length {actual} does not match {width}x{height} with {channels} channels")]
    TexelSizeMismatch {
        width: u32,
        height: u32,
        channels: usize,
        actual: usize,
    },
}


/// Errors produced while importing a glTF document.
#[derive(Debug, thiserror::Error)]
pub enum GltfImportError {
    #[error("glTF source '{0}' is not a file")]
    SourceNotAFile(PathBuf),

    #[error("cannot read glTF file '{path}'")]
    Read {
        path: PathBuf,
        #[source]
        source: gltf::Error,
    },

    #[error("this gltf file depends on an external resource '{0}'.")]
    ExternalResourceUnavailable(String),

    #[error("generated subresource name '{0}' is invalid.")]
    InvalidSubresourceName(PathBuf),

    #[error("gltf index {0} is out of range.")]
    IndexOutOfRange(usize),

    #[error("unsupported glTF texture format {0:?}")]
    UnsupportedTextureFormat(gltf::image::Format),

    #[error("texture has a zero dimension")]
    ZeroDimension,

    #[error("failed to recondition vertices")]
    ReconditionFailed(#[from] ReconditionerError),

    #[error("failed to import MRAO image")]
    BadMraoTexture(#[from] GltfImageImportError),

    #[error("skinned primitive is missing JOINTS_0 / WEIGHTS_0")]
    MissingSkinningAttributes,

    #[error("skin has {joints} joints but {matrices} inverse bind matrices")]
    SkinJointCountMismatch { joints: usize, matrices: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum ChannelFormat {
    U8,
    U16,
    F32,
}

impl ChannelFormat {
    pub(super) fn bytes_per_channel(self) -> usize {
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
    Three,
    Four,
}

impl ChannelCount {
    pub(super) fn count(self) -> usize {
        match self {
            ChannelCount::One => 1,
            ChannelCount::Two => 2,
            ChannelCount::Three => 3,
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
    pub fn from_gltf(image: &'a gltf::image::Data) -> Result<Self, GltfImageImportError> {
        let (format, channels) = match image.format {
            gltf::image::Format::R8 => (ChannelFormat::U8, ChannelCount::One),
            gltf::image::Format::R8G8 => (ChannelFormat::U8, ChannelCount::Two),
            // While the GPU texture does not support RGB three channel images, reading from them is fine.
            gltf::image::Format::R8G8B8 => (ChannelFormat::U8, ChannelCount::Three),
            gltf::image::Format::R8G8B8A8 => (ChannelFormat::U8, ChannelCount::Four),
            gltf::image::Format::R16 => (ChannelFormat::U16, ChannelCount::One),
            gltf::image::Format::R16G16 => (ChannelFormat::U16, ChannelCount::Two),
            gltf::image::Format::R16G16B16 => (ChannelFormat::U16, ChannelCount::Three),
            gltf::image::Format::R16G16B16A16 => (ChannelFormat::U16, ChannelCount::Four),
            gltf::image::Format::R32G32B32A32FLOAT => (ChannelFormat::F32, ChannelCount::Four),
            other => return Err(GltfImageImportError::UnsupportedFormat(other)),
        };

        Self::new(format, channels, image.width, image.height, &image.pixels)
    }

    pub(super) fn new(
        format: ChannelFormat,
        channels: ChannelCount,
        width: u32,
        height: u32,
        texels: &'a [u8],
    ) -> Result<Self, GltfImageImportError> {
        let width = NonZero::new(width).ok_or(GltfImageImportError::InvalidTexture)?;
        let height = NonZero::new(height).ok_or(GltfImageImportError::InvalidTexture)?;

        let expected_len = width.get() as u64
            * height.get() as u64
            * channels.count() as u64
            * format.bytes_per_channel() as u64;
        if expected_len != texels.len() as u64 {
            return Err(GltfImageImportError::TexelSizeMismatch {
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

    pub(super) fn channel_or(&self, x: u32, y: u32, channel: usize, default: f32) -> f32 {
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
