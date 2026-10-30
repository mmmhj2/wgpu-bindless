//! glTF importer entry point and import orchestration.
//!
//! The importer is split by responsibility:
//! - [`super::naming`]: subasset names,
//! - [`super::primitive`]: geometry streams and material slots,
//! - [`super::scene`]: scenes, skins and their node arenas,
//! - [`super::texture`]: textures, samplers and MRAO,
//! - [`super::animation`]: baking animation curves into joint-matrix textures.
//!
//! This module owns the importer state and decides the import order.

use std::{collections::{HashMap, HashSet}, path::PathBuf, sync::{Arc, RwLock}};

use crate::asset::{
    asset_types::{Asset, AssetData, AssetMetadata},
    error::ImportError,
    importer::{AssetImporter, ImporterContext, gltf_importer::{
        animation::{bake_animation, BAKE_FPS},
        common::GltfImportError,
        naming::SubAssetType,
        texture::MraoKey,
    }},
};

/// glTF importer class.
pub struct GltfImporter {
    // Guard it by mutex if we want to parallelize it.
    pub(super) color_textures: HashSet<usize>,
    pub(super) normal_textures: HashSet<usize>,
    /// MRAO source combinations requested by material slots.
    pub(super) mrao_requests: HashSet<MraoKey>,
    /// Materials already warned about unsupported factor values.
    pub(super) warned_materials: HashSet<usize>,
    /// The glTF file this importer reads from.
    pub path: Box<std::path::Path>,
}

impl AssetImporter for GltfImporter {
    const IMPORTER_ID: &'static str = "gltf";
    /// 4: imports skins, skinned primitives and baked animations.
    /// 5: recomputes missing vertex normals from the triangle winding.
    const IMPORTER_VERSION: u32 = 5;

    fn import(mut self, context: &ImporterContext) -> Result<(), ImportError> {
        if !self.path.is_file() {
            return Err(Box::new(GltfImportError::SourceNotAFile(self.path.to_path_buf())));
        }
        context.record_source(&self.path);

        // Clone the path so borrowing it does not conflict with `&mut self` passed below.
        let source_path = self.path.clone();
        let path = source_path.as_ref();
        let (document, buffers, images) = gltf::import(path)
            .map_err(|source| GltfImportError::Read { path: path.to_path_buf(), source })?;

        for b in document.buffers() {
            if let gltf::buffer::Source::Uri(uri) = b.source() {
                // Allow embedded Base64 data in URI
                if !uri.starts_with("data") {
                    return Err(Box::new(GltfImportError::UnsupportedFeature("external resource")))
                }
            }
        }
        for i in document.images() {
            if let gltf::image::Source::Uri { uri, mime_type: _ } = i.source() {
                if !uri.starts_with("data") {
                    return Err(Box::new(GltfImportError::UnsupportedFeature("external resource")))
                }
            }
        }

        // glTF only associates a skin with *nodes*, so the skin <-> mesh mapping
        // must be resolved from the scene graph before any mesh is imported.
        let skin_usage = Self::resolve_skin_usage(&document);
        let mut mesh_skin: HashMap<usize, usize> = HashMap::new();
        for (skin, usage) in &skin_usage {
            mesh_skin.entry(usage.mesh).or_insert(*skin);
        }
        let unskinned_meshes = Self::meshes_used_unskinned(&document);

        for mesh in document.meshes() {
            log::info!("Found mesh {} with {} primitives, importing...", mesh.index(), mesh.primitives().len());

            let skin_index = mesh_skin.get(&mesh.index()).copied();
            // A mesh that is also placed on a skinless node keeps its static
            // asset, so both variants exist and neither is orphaned.
            let needs_static = skin_index.is_none() || unskinned_meshes.contains(&mesh.index());

            for primitive in mesh.primitives() {
                let skinnable = Self::has_skinning_attributes(&primitive);

                if needs_static || !skinnable {
                    let asset_type = SubAssetType::Primitive { mesh: mesh.index(), primitive: primitive.index() };
                    let subasset_path = Self::generate_subasset_filename(path, asset_type)?;
                    let subasset_name = Self::generate_subasset_name(path, asset_type)?.into_string()
                        .map_err(|_| GltfImportError::InvalidSubresourceName(path.to_path_buf()))?;
                    let imported_mesh = Self::import_from_gltf_primitive(&mut self, &primitive, &buffers)?;
                    Self::queue_asset(context, subasset_name, subasset_path, AssetData::StaticMeshAssetType(imported_mesh));
                }

                if let (Some(skin_index), true) = (skin_index, skinnable) {
                    let usage = skin_usage[&skin_index];
                    let skin_name = Self::generate_subasset_name(
                        path,
                        SubAssetType::Skin { scene: usage.scene, skin: skin_index }
                    )?.into_string().map_err(|_| GltfImportError::InvalidSubresourceName(path.to_path_buf()))?;

                    let asset_type = SubAssetType::SkinnedPrimitive { mesh: mesh.index(), primitive: primitive.index() };
                    let subasset_path = Self::generate_subasset_filename(path, asset_type)?;
                    let subasset_name = Self::generate_subasset_name(path, asset_type)?.into_string()
                        .map_err(|_| GltfImportError::InvalidSubresourceName(path.to_path_buf()))?;
                    let imported_mesh = Self::import_skinned_from_gltf_primitive(&mut self, &primitive, &buffers, skin_name)?;
                    Self::queue_asset(context, subasset_name, subasset_path, AssetData::SkinnedMeshAssetType(imported_mesh));
                }
            }
        }

        log::info!("Queuing import for color textures: {:?}", self.color_textures);
        for texture_idx in self.color_textures {
            let asset_type = SubAssetType::Texture(texture_idx);
            let subasset_path = Self::generate_subasset_filename(path, asset_type)?;
            let subasset_name = Self::generate_subasset_name(path, asset_type)?.into_string()
                .map_err(|_| GltfImportError::InvalidSubresourceName(path.to_path_buf()))?;
            log::info!("Found color texture {}.", subasset_name);
            let imported_texture = Self::import_from_gltf_texture(
                &document.textures().nth(texture_idx).expect("texture index should be valid"),
                &images, true
            )?;
            Self::queue_asset(context, subasset_name, subasset_path, AssetData::TextureAssetType(imported_texture));
        }

        log::info!("Queuing import for normal textures:{:?}", self.normal_textures);
        for texture_idx in self.normal_textures {
            let asset_type = SubAssetType::Texture(texture_idx);
            let subasset_path = Self::generate_subasset_filename(path, asset_type)?;
            let subasset_name = Self::generate_subasset_name(path, asset_type)?.into_string()
                .map_err(|_| GltfImportError::InvalidSubresourceName(path.to_path_buf()))?;
            log::info!("Found normal texture {}.", subasset_name);
            let imported_texture = Self::import_from_gltf_texture(
                &document.textures().nth(texture_idx).expect("texture index should be valid"),
                &images, false
            )?;
            Self::queue_asset(context, subasset_name, subasset_path, AssetData::TextureAssetType(imported_texture));
        }

        let mrao_requests: Vec<MraoKey> = self.mrao_requests.iter().copied().collect();
        for key in mrao_requests {
            let asset_type = SubAssetType::MraoTexture(key);
            let subasset_path = Self::generate_subasset_filename(path, asset_type)?;
            let subasset_name = Self::generate_subasset_name(path, asset_type)?.into_string()
                .map_err(|_| GltfImportError::InvalidSubresourceName(path.to_path_buf()))?;
            log::info!("Generating MRAO texture {}.", subasset_name);
            let imported_texture = Self::import_from_gltf_mrao(&document, &images, key)?;
            Self::queue_asset(context, subasset_name, subasset_path, AssetData::TextureAssetType(imported_texture));
        }

        for scene in document.scenes() {
            let asset_type = SubAssetType::Scene(scene.index());
            let subasset_path = Self::generate_subasset_filename(path, asset_type)?;
            let subasset_name = Self::generate_subasset_name(path, asset_type)?.into_string()
                .map_err(|_| GltfImportError::InvalidSubresourceName(path.to_path_buf()))?;
            log::info!("Found scene {}.", subasset_name);
            let imported_scene = Self::import_from_gltf_scene(path, &document, &scene)?;
            Self::queue_asset(context, subasset_name, subasset_path, AssetData::SceneAssetType(imported_scene));
        }

        // Skins are scoped to the scene that uses them, so that `scene_node`
        // indices can point into that scene's node arena. Importing them after
        // the scenes keeps the log order readable; the arena indices come from
        // the shared `build_scene_index_map`, not from the imported asset.
        let mut skin_inverse_bind_matrices: HashMap<usize, Vec<[[f32; 4]; 4]>> = HashMap::new();
        let mut skin_indices: Vec<usize> = skin_usage.keys().copied().collect();
        skin_indices.sort_unstable();
        for skin_index in skin_indices {
            let usage = skin_usage[&skin_index];
            let scene = document.scenes().nth(usage.scene)
                .ok_or(GltfImportError::IndexOutOfRange(usage.scene))?;
            let skin = document.skins().nth(skin_index)
                .ok_or(GltfImportError::IndexOutOfRange(skin_index))?;

            let asset_type = SubAssetType::Skin { scene: usage.scene, skin: skin_index };
            let subasset_path = Self::generate_subasset_filename(path, asset_type)?;
            let subasset_name = Self::generate_subasset_name(path, asset_type)?.into_string()
                .map_err(|_| GltfImportError::InvalidSubresourceName(path.to_path_buf()))?;
            log::info!("Found skin {} with {} joints.", subasset_name, skin.joints().count());

            let imported_skin = Self::import_from_gltf_skin(path, &document, &scene, &skin, &buffers)?;
            skin_inverse_bind_matrices.insert(
                skin_index,
                imported_skin.joints.iter().map(|joint| joint.inverse_bind_matrix).collect(),
            );
            Self::queue_asset(context, subasset_name, subasset_path, AssetData::SkinAssetType(imported_skin));
        }

        // Animations are baked right away: every frame is sampled on the CPU and
        // packed into a joint-matrix texture, so the runtime only has to pick a
        // row per draw.
        for animation in document.animations() {
            let Some(skin_index) = Self::find_animation_skin(&animation, &document) else {
                log::info!(
                    "Skipping animation '{}': no skin targets its nodes.",
                    animation.name().unwrap_or("<unnamed>")
                );
                continue;
            };
            let Some(inverse_bind_matrices) = skin_inverse_bind_matrices.get(&skin_index) else {
                log::warn!(
                    "Skipping animation '{}': skin {} was not imported.",
                    animation.name().unwrap_or("<unnamed>"),
                    skin_index
                );
                continue;
            };
            let skin = document.skins().nth(skin_index)
                .ok_or(GltfImportError::IndexOutOfRange(skin_index))?;
            let usage = skin_usage[&skin_index];

            let asset_type = SubAssetType::Animation { scene: usage.scene, animation: animation.index() };
            let subasset_path = Self::generate_subasset_filename(path, asset_type)?;
            let subasset_name = Self::generate_subasset_name(path, asset_type)?.into_string()
                .map_err(|_| GltfImportError::InvalidSubresourceName(path.to_path_buf()))?;
            log::info!(
                "Baking animation '{}' as {}.",
                animation.name().unwrap_or("<unnamed>"),
                subasset_name
            );

            let skin_name = Self::generate_subasset_name(path, SubAssetType::Skin { scene: usage.scene, skin: skin_index })?
                .into_string().map_err(|_| GltfImportError::InvalidSubresourceName(path.to_path_buf()))?;
            let baked = bake_animation(
                &document,
                &buffers,
                &animation,
                &skin,
                inverse_bind_matrices,
                usage.node,
                skin_name,
                BAKE_FPS,
            )?;
            Self::queue_asset(context, subasset_name, subasset_path, AssetData::SkinnedAnimationAssetType(baked));
        }

        Ok(())
    }
}

impl GltfImporter {
    pub fn new(path: impl AsRef<std::path::Path>) -> Self {
        Self {
            color_textures: HashSet::new(),
            normal_textures: HashSet::new(),
            mrao_requests: HashSet::new(),
            warned_materials: HashSet::new(),
            path: path.as_ref().into(),
        }
    }

    /// Register a produced asset into the importer's batch.
    fn queue_asset(context: &ImporterContext, subasset_name: String, subasset_path: PathBuf, data: AssetData) {
        let imported_asset = Arc::from(
            RwLock::from(Asset::new(
                AssetMetadata::new(
                    subasset_name.clone(),
                    Some(subasset_path.into_boxed_path())
                ),
                data
            ))
        );
        context.asset_batch.write().unwrap_or_else(std::sync::PoisonError::into_inner).insert(subasset_name, imported_asset);
    }
}
