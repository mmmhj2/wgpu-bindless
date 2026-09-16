use std::sync::{Arc, RwLock};

use crate::asset::{asset_types::{Asset, AssetMetadata, static_mesh_asset::{self, StaticMeshAsset}}, importer::AssetImporter};

pub struct GltfImporter {
}

impl AssetImporter for GltfImporter {
    fn import(&mut self, context: &super::ImporterContext) {
        assert!(context.imported_file_path.is_file());
        
        let path = context.imported_file_path.as_ref();
        let (document, buffers, textures) = gltf::import(path).expect("Failed to open gltf file.");

        for mesh in document.meshes() {
            log::info!("Found mesh {} with {} primitives, importing...", mesh.index(), mesh.primitives().len());

            for primitive in mesh.primitives() {
                let subasset_path = Self::generate_subasset_filename(path, mesh.index(), SubAssetType::Primitive);
                let subasset_name = subasset_path.file_name().unwrap().to_str().expect("Subasset contains non UTF-8 characters.").to_string();
                let imported_mesh = StaticMeshAsset::import_from_gltf_primitive(context, &primitive, &buffers);
                let imported_asset = Arc::from(
                    RwLock::from(Asset::new(
                        AssetMetadata{
                            asset_file_path: Some(subasset_path.into_boxed_path()),
                            asset_name: subasset_name.clone()
                        },
                        crate::asset::asset_types::AssetData::StaticMeshAsset(imported_mesh)
                    ))
                );
                context.asset_batch.write().expect("msg").insert(subasset_name, imported_asset);
            }
        }

        for texture in document.textures() {
            let subasset_name = Self::generate_subasset_filename(path, texture.index(), SubAssetType::Texture);
            log::info!("Found texture {}.", subasset_name.display());
        }
    }
}

#[derive(Debug)]
pub(crate) enum SubAssetType {
    Texture,
    // Note: GLTF mesh corresponds to mesh instances, which is a part of a scene.
    Mesh,
    // GLTF primitive, which corresponds to mesh.
    Primitive
}

impl GltfImporter {
    pub fn new() -> Self {
        Self {  }
    }

    pub(crate) fn generate_subasset_filename(gltf_file: &std::path::Path, index: usize, asset_type: SubAssetType) -> std::path::PathBuf {
        assert!(gltf_file.is_file());

        let new_suffix = match asset_type {
            SubAssetType::Texture => format!(".tex_{}.asset", index),
            SubAssetType::Mesh => todo!("Mesh import is not supported yet."),
            SubAssetType::Primitive => format!(".prim_{}.asset", index),
        };

        let mut new_filename = gltf_file.file_name().unwrap().to_os_string();
        new_filename.push(new_suffix);
        gltf_file.with_file_name(new_filename)
    }

    pub(crate) fn generate_subasset_filename_string(gltf_file: &std::path::Path, index: usize, asset_type: SubAssetType) -> Option<String> {
        Some(Self::generate_subasset_filename(gltf_file, index, asset_type).to_str()?.to_string())
    }
}
