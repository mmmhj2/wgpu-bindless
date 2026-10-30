//! Subasset naming: one deterministic name per produced asset.
//!
//! Every subasset is named `<gltf file name><suffix>`; the suffix encodes what
//! the asset is, so the name alone is enough to address it in the asset database.

use std::{ffi::OsString, path::{Path, PathBuf}};

use super::common::GltfImportError;
use super::importer::GltfImporter;
use super::texture::MraoKey;

#[derive(Debug, Clone, Copy)]
pub(super) enum SubAssetType {
    Texture(usize),
    /// A GLTF mesh, which consists of several primitives, and is placed in a scene.
    /// 
    /// Differences between meshes and primitives:
    /// - A mesh is a collection of primitives, and a primitive is a single draw call (therefore a single material).
    /// - A mesh can be instanced in a scene to have model matrices, and a primitive does not have model matrices.
    /// - Two layers of instancing are therefore possible: a mesh can be instanced in a scene (on CPU),
    /// and a primitive can be instanced in a mesh (on GPU, with instanced drawing).
    /// 
    /// Currently unused, but we may want to support importing a mesh as a single subasset in the future, if we have submeshes set up.
    #[allow(dead_code)]
    Mesh(usize),
    /// GLTF primitive, which corresponds to mesh for our renderer.
    /// TODO: support multiple primitives per mesh. Maybe we should use `submesh` and map primitives to submeshes.
    Primitive { mesh: usize, primitive: usize },
    /// GLTF primitive bound to a skin, i.e. carrying `JOINTS_0` / `WEIGHTS_0`.
    ///
    /// Kept separate from [`SubAssetType::Primitive`] because both variants may
    /// exist for the same primitive when the mesh is also placed on a skinless
    /// node.
    SkinnedPrimitive { mesh: usize, primitive: usize },
    /// A GLTF skin, scoped to the scene that uses it.
    Skin { scene: usize, skin: usize },
    /// A GLTF animation baked into joint-matrix texture, scoped to its scene.
    Animation { scene: usize, animation: usize },
    /// A GLTF scene, preserved as a hierarchy of nodes with local transforms.
    Scene(usize),
    /// A renderer-layout MRAO texture derived from optional MR and AO textures.
    MraoTexture(MraoKey),
}

impl GltfImporter {
    // For non-public interfaces, concrete type `&Path` might actually be
    // better than `AsRef<Path>`, as it reduces instantiation bloat. See
    // https://internals.rust-lang.org/t/reducing-generics-bloat/6337
    pub(super) fn generate_subasset_filename(gltf_file: impl AsRef<Path>, asset_type: SubAssetType) -> Result<PathBuf, GltfImportError> {
        Ok(gltf_file.as_ref().with_file_name(Self::generate_subasset_name(&gltf_file, asset_type)?).with_added_extension("asset"))
    }

    pub(super) fn generate_subasset_name(gltf_file: impl AsRef<Path>, asset_type: SubAssetType) -> Result<OsString, GltfImportError> {
        let new_suffix = match asset_type {
            SubAssetType::Texture(index) => format!(".tex_{}", index),
            SubAssetType::Mesh(index) => format!(".mesh_{}", index),
            SubAssetType::Primitive { mesh, primitive } => format!(".prim_{}_{}", mesh, primitive),
            SubAssetType::SkinnedPrimitive { mesh, primitive } => format!(".skpr_{}_{}", mesh, primitive),
            SubAssetType::Skin { scene, skin } => format!(".skin_{}_{}", scene, skin),
            SubAssetType::Animation { scene, animation } => format!(".anim_{}_{}", scene, animation),
            SubAssetType::Scene(index) => format!(".scene_{}", index),
            SubAssetType::MraoTexture(key) => {
                fn source_label(prefix: &str, index: Option<usize>) -> String {
                    index
                        .map(|index| format!("{prefix}{index}"))
                        .unwrap_or_else(|| String::from("none"))
                }

                format!(
                    ".mrao_{}_{}",
                    source_label("mr", key.mr),
                    source_label("ao", key.ao)
                )
            }
        };

        let mut new_filename = gltf_file.as_ref().file_name()
            .ok_or_else(|| GltfImportError::InvalidSubresourceName(gltf_file.as_ref().to_path_buf()))?
            .to_os_string();
        new_filename.push(new_suffix);
        Ok(new_filename)
    }

    /// Resolve the subasset name of a single glTF scene.
    pub fn scene_subasset_name(gltf_file: impl AsRef<Path>, scene_index: usize) -> Result<String, GltfImportError> {
        Self::generate_subasset_name(&gltf_file, SubAssetType::Scene(scene_index))?
            .into_string()
            .map_err(|_| GltfImportError::InvalidSubresourceName(gltf_file.as_ref().to_path_buf()))
    }

    /// Resolve the subasset name of a skinned primitive.
    pub fn skinned_primitive_subasset_name(gltf_file: impl AsRef<Path>, mesh_index: usize, primitive_index: usize) -> Result<String, GltfImportError> {
        Self::generate_subasset_name(&gltf_file, SubAssetType::SkinnedPrimitive { mesh: mesh_index, primitive: primitive_index })?
            .into_string()
            .map_err(|_| GltfImportError::InvalidSubresourceName(gltf_file.as_ref().to_path_buf()))
    }

    /// Resolve the subasset name of a skin, scoped to the scene that uses it.
    pub fn skin_subasset_name(gltf_file: impl AsRef<Path>, scene_index: usize, skin_index: usize) -> Result<String, GltfImportError> {
        Self::generate_subasset_name(&gltf_file, SubAssetType::Skin { scene: scene_index, skin: skin_index })?
            .into_string()
            .map_err(|_| GltfImportError::InvalidSubresourceName(gltf_file.as_ref().to_path_buf()))
    }

    /// Resolve the subasset name of a baked animation, scoped to its scene.
    pub fn animation_subasset_name(gltf_file: impl AsRef<Path>, scene_index: usize, animation_index: usize) -> Result<String, GltfImportError> {
        Self::generate_subasset_name(&gltf_file, SubAssetType::Animation { scene: scene_index, animation: animation_index })?
            .into_string()
            .map_err(|_| GltfImportError::InvalidSubresourceName(gltf_file.as_ref().to_path_buf()))
    }
}
