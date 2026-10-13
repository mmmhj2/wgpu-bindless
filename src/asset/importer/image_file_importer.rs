use std::{num::NonZero, path::Path, sync::Arc};

use crate::asset::{
    asset_types::{
        Asset, AssetData::TextureAssetType, AssetMetadata,
        texture_asset::{TexelData, TextureAsset, TextureDimension, TextureFormat},
    },
    importer::AssetImporter,
};

/// How the texel data of an imported image should be interpreted by the GPU.
#[derive(Clone, Copy, Debug)]
pub enum ColorSpace {
    Linear,
    /// Gamma-corrected sRGB color space.
    /// Only used if the texel is 8-bit per channel, otherwise the texel is 
    /// always treated as linear.
    Srgb,
}

/// An image decoded into 4-channel texel data.
///
/// RGB images are expanded to RGBA by the underlying decoder, so the texel
/// format is always one of the 4-channel formats.
/// 
/// **TODO:** Currently we are only supporting RGBA formats without mipmaps.
/// Maybe we should support more formats and mipmaps in the future.
/// See [`TexelData`] and [`TextureAsset`] for more info.
struct DecodedImage {
    format: TextureFormat,
    width: NonZero<u32>,
    height: NonZero<u32>,
    texels: Box<[u8]>,
}

/// Decode a single image file into texel data.
fn decode_image(path: &Path, color_space: ColorSpace) -> DecodedImage {
    let image = image::ImageReader::open(path);
    if let Err(e) = image {
        panic!("failed to open image file: {e}");
    }
    let image = image.unwrap().decode();
    if let Err(e) = image {
        panic!("failed to decode image file: {e}");
    }
    let image = image.unwrap();
    let width = image.width();
    let height = image.height();

    let format = match image.color() {
        image::ColorType::Rgb8 | image::ColorType::Rgba8 => match color_space {
            ColorSpace::Srgb => TextureFormat::Rgba8Srgb,
            ColorSpace::Linear => TextureFormat::Rgba8Unorm,
        },
        image::ColorType::Rgb16 | image::ColorType::Rgba16 => TextureFormat::Rgba16Unorm,
        image::ColorType::Rgb32F | image::ColorType::Rgba32F => TextureFormat::Rgba32Float,
        _ => panic!("unsupported image format {:?}", image.color()),
    };

    let texels = match image.color() {
        image::ColorType::Rgb8 | image::ColorType::Rgba8 => image.into_rgba8().into_raw().into_boxed_slice(),
        image::ColorType::Rgb16 | image::ColorType::Rgba16 =>
            bytemuck::cast_slice_box(image.into_rgba16().into_raw().into_boxed_slice()),
        image::ColorType::Rgb32F | image::ColorType::Rgba32F =>
            bytemuck::cast_slice_box(image.into_rgba32f().into_raw().into_boxed_slice()),
        _ => panic!("unsupported image format {:?}", image.color()),
    };

    DecodedImage {
        format,
        width: width.try_into().expect("width of the texture should be at least 1"),
        height: height.try_into().expect("height of the texture should be at least 1"),
        texels,
    }
}

/// Derive the asset name from a source path.
///
/// The extension is stripped, e.g. `resource/skybox/back.jpg` -> `back`.
fn asset_name_from_path(path: &Path) -> String {
    path.with_extension("").file_name()
        .expect("imported asset should have a file name")
        .to_str().expect("name of imported asset should contain UTF-8 chars only")
        .to_string()
}

fn insert_texture_asset(
    context: &super::ImporterContext,
    asset_name: String,
    asset_file_path: Box<Path>,
    texture: TextureAsset,
) {
    let asset = Arc::new(Asset::new(
        AssetMetadata { asset_name: asset_name.clone(), asset_file_path: Some(asset_file_path) },
        TextureAssetType(texture),
    ).into());
    context.asset_batch.write().expect("failed to acquire write lock for asset database").insert(asset_name, asset);
}

/// Import a single image file into a `TextureAsset`.
pub struct ImageFileImporter {
    source: Box<Path>,
    color_space: ColorSpace,
}

impl ImageFileImporter {
    pub fn new(source: impl AsRef<Path>, color_space: ColorSpace) -> Self {
        Self { source: source.as_ref().into(), color_space }
    }
}

impl AssetImporter for ImageFileImporter {
    fn import(self, context: &super::ImporterContext) {
        let image = decode_image(&self.source, self.color_space);
        let asset_name = asset_name_from_path(&self.source);
        let asset_file_path = self.source.with_added_extension("asset").into_boxed_path();

        insert_texture_asset(
            context,
            asset_name,
            asset_file_path,
            TextureAsset {
                format: image.format,
                dimension: TextureDimension::D2,
                width: image.width,
                height: Some(image.height),
                depth_or_array_slice: None,
                sampler: Default::default(),
                texels: TexelData::Plain(image.texels),
            },
        );
    }
}

/// Import a collection of images as a 2D texture array (e.g. a cubemap).
///
/// All slices must share the same dimensions and texel format.
pub struct ImageArrayImporter {
    /// Optional name of the resulting texture asset.
    /// If `None`, the name is derived from the first source path.
    name: Option<String>,
    sources: Vec<Box<Path>>,
    color_space: ColorSpace,
}

impl ImageArrayImporter {
    pub fn new<P: AsRef<Path>>(
        name: Option<String>,
        sources: impl IntoIterator<Item = P>,
        color_space: ColorSpace,
    ) -> Self {
        Self { name, sources: sources.into_iter().map(|source| source.as_ref().into()).collect(), color_space }
    }
}

impl AssetImporter for ImageArrayImporter {
    fn import(self, context: &super::ImporterContext) {
        assert!(!self.sources.is_empty(), "an image array importer requires at least one source.");

        let decoded: Vec<DecodedImage> = self.sources.iter()
            .map(|source| decode_image(source, self.color_space))
            .collect();

        let format = decoded[0].format;
        let width = decoded[0].width;
        let height = decoded[0].height;
        for image in &decoded {
            if image.width != width || image.height != height || image.format != format {
                panic!("all slices of an image array must share width, height and texel format.");
            }
        }

        let slice_count = decoded.len() as u32;
        let mut texels: Vec<u8> = Vec::new();
        for image in decoded {
            texels.extend_from_slice(&image.texels);
        }

        let asset_name = self.name.unwrap_or_else(|| format!("{}.array", asset_name_from_path(&self.sources[0])));
        let asset_file_path = self.sources[0].with_file_name(&asset_name).with_added_extension("asset").into_boxed_path();

        insert_texture_asset(
            context,
            asset_name,
            asset_file_path,
            TextureAsset {
                format,
                dimension: TextureDimension::D2,
                width,
                height: Some(height),
                depth_or_array_slice: Some(NonZero::new(slice_count).unwrap()),
                sampler: Default::default(),
                texels: TexelData::Plain(texels.into_boxed_slice()),
            },
        );
    }
}
