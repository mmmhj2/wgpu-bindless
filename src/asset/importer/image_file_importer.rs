use std::{mem, sync::Arc};

use crate::asset::{asset_types::{Asset, AssetData::TextureAssetType, AssetMetadata, texture_asset::{TexelData, TextureAsset, TextureDimension, TextureFormat}}, importer::AssetImporter};


pub struct ImageFileImporter {
    is_srgb_format: bool,
    /// Import an array of images, instead of a single image.
    /// 
    /// When this Option exists, the path specified in the importer context will be ignored,
    /// and the asset file produced will have an `array` suffix attached to its name.
    array_slices: Option<Vec<Box<std::path::Path>>>
}

impl AssetImporter for ImageFileImporter {
    fn import(self, context: &super::ImporterContext) {
        if let Some(slices) = self.array_slices {
            todo!()
        } else {
            let asset_name = context.imported_file_path.with_extension("").file_name().expect("Imported asset should have a file name").to_str().expect("name of imported asset should contain UTF-8 chars only").to_string();
            let asset_file_path = context.imported_file_path.with_added_extension("asset").into_boxed_path();
            let asset = Arc::new(Asset::new(
                AssetMetadata{ asset_name: asset_name.clone(), asset_file_path: Some(asset_file_path) },
                TextureAssetType(Self::import_single_file(self, context).unwrap())
            ).into());
            let mut asset_db = context.asset_batch.write().expect("failed to acquire write lock for asset database");
            asset_db.insert(asset_name, asset);
        }
    }
}

impl TryInto<TextureFormat> for image::ColorType {
    type Error = ();
    fn try_into(self) -> Result<TextureFormat, Self::Error> {
        match self {
            image::ColorType::Rgba8 => Ok(TextureFormat::Rgba8Unorm),
            image::ColorType::Rgba16 => Ok(TextureFormat::Rgba16Unorm),
            image::ColorType::Rgba32F => Ok(TextureFormat::Rgba32Float),
            // These formats need to fill up to 4 components
            image::ColorType::Rgb8 => todo!(),
            image::ColorType::Rgb16 => todo!(),
            image::ColorType::Rgb32F => todo!(),
            _ => panic!("Unsupported image format"),
        }
    }
}

impl ImageFileImporter {
    pub fn new(
        is_srgb_format: bool,
        array_slices: Option<Vec<Box<std::path::Path>>>
    ) -> Self {
        Self { is_srgb_format, array_slices }
    }

    fn import_single_file(self, context: &super::ImporterContext) -> Result<TextureAsset, ()> {
        let image = image::ImageReader::open(context.imported_file_path.as_ref());
        if let Err(e) = image {
            panic!("failed to open image file");
        }
        let image = image.unwrap().decode();
        if let Err(e) = image {
            panic!("failed to decode file");
        }
        let image = image.unwrap();
        let width = image.width();
        let height = image.height();

        let format = match image.color() {
            image::ColorType::Rgb8 => if self.is_srgb_format {TextureFormat::Rgba8Srgb} else {TextureFormat::Rgba8Unorm},
            image::ColorType::Rgba8 => if self.is_srgb_format {TextureFormat::Rgba8Srgb} else {TextureFormat::Rgba8Unorm},
            image::ColorType::Rgb16 => TextureFormat::Rgba16Unorm,
            image::ColorType::Rgba16 => TextureFormat::Rgba16Unorm,
            image::ColorType::Rgb32F => TextureFormat::Rgba32Float,
            image::ColorType::Rgba32F => TextureFormat::Rgba32Float,
            _ => panic!("unsupported image format"),
        };

        let data = match image.color() {
            image::ColorType::Rgb8 | image::ColorType::Rgba8 => image.into_rgba8().into_raw().into_boxed_slice(),
            image::ColorType::Rgb16 | image::ColorType::Rgba16 =>
                bytemuck::cast_slice_box(image.into_rgba16().into_raw().into_boxed_slice()),
            image::ColorType::Rgb32F | image::ColorType::Rgba32F =>
                bytemuck::cast_slice_box(image.into_rgba32f().into_raw().into_boxed_slice()),
            _ => panic!("unsupported image format"),
        };

        return Ok(
            TextureAsset {
                format,
                dimension: TextureDimension::D2,
                size: (width, height, 1),
                sampler: Default::default(),
                texels: TexelData::Uncompressed(data)
            }
        );
    }
}
