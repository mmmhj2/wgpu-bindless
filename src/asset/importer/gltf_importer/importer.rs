use std::{collections::{HashMap, HashSet, VecDeque}, ffi::OsString, num::NonZeroU32, path::{Path, PathBuf}, sync::{Arc, RwLock}};

use crate::asset::{asset_types::{Asset, AssetData, AssetMetadata, scene_asset::{SceneAsset, SceneNodeAsset}, skin_asset::{SkinAsset, SkinJoint}, skinned_mesh_asset::SkinnedMeshAsset, static_mesh_asset::{StaticMeshAsset, StaticMeshMaterial}, texture_asset::{AddressMode, FilterMode, Sampler, TexelData, TextureAsset, TextureDimension::D2, TextureFormat}, vertex::{VertexBufferOthers, VertexBufferPosition, VertexBufferSkinning}}, error::ImportError, importer::{AssetImporter, ImporterContext, gltf_importer::{animation::{bake_animation, BAKE_FPS}, common::{GltfImageView, GltfImportError}, mrao::build_mrao_texture_asset}, vertex_reconditioner::{ReconditionerError, TangentRecalculator}}};

/// glTF importer class.
/// 
/// TODO: Asset import impl is too large. Try splitting it into smaller trunks.
pub struct GltfImporter {
    // Guard it by mutex if we want to parallelize it.
    color_textures: HashSet<usize>,
    normal_textures: HashSet<usize>,
    /// MRAO source combinations requested by material slots.
    mrao_requests: HashSet<MraoKey>,
    /// Materials already warned about unsupported factor values.
    warned_materials: HashSet<usize>,
    /// The glTF file this importer reads from.
    pub path: Box<std::path::Path>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct MraoKey {
    mr: Option<usize>,
    ao: Option<usize>,
}

impl AssetImporter for GltfImporter {
    const IMPORTER_ID: &'static str = "gltf";
    /// 4: imports skins and skinned primitives (`.skpr_*` / `.skin_*`).
    const IMPORTER_VERSION: u32 = 4;

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
            if let gltf::buffer::Source::Uri(_) = b.source() {
                return Err(Box::new(GltfImportError::UnsupportedFeature("external resource")))
            }
        }
        for i in document.images() {
            if let gltf::image::Source::Uri { uri: _, mime_type: _ } = i.source() {
                return Err(Box::new(GltfImportError::UnsupportedFeature("external resource")))
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
            let imported_asset = Arc::from(
                RwLock::from(Asset::new(
                    AssetMetadata::new(
                        subasset_name.clone(),
                        Some(subasset_path.into_boxed_path())
                    ),
                    crate::asset::asset_types::AssetData::TextureAssetType(imported_texture)
                ))
            );
            context.asset_batch.write().unwrap_or_else(std::sync::PoisonError::into_inner).insert(subasset_name, imported_asset);
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
            let imported_asset = Arc::from(
                RwLock::from(Asset::new(
                    AssetMetadata::new(
                        subasset_name.clone(),
                        Some(subasset_path.into_boxed_path())
                    ),
                    crate::asset::asset_types::AssetData::TextureAssetType(imported_texture)
                ))
            );
            context.asset_batch.write().unwrap_or_else(std::sync::PoisonError::into_inner).insert(subasset_name, imported_asset);
        }

        let mrao_requests: Vec<MraoKey> = self.mrao_requests.iter().copied().collect();
        for key in mrao_requests {
            let asset_type = SubAssetType::MraoTexture(key);
            let subasset_path = Self::generate_subasset_filename(path, asset_type)?;
            let subasset_name = Self::generate_subasset_name(path, asset_type)?.into_string()
                .map_err(|_| GltfImportError::InvalidSubresourceName(path.to_path_buf()))?;
            log::info!("Generating MRAO texture {}.", subasset_name);
            let imported_texture = Self::import_from_gltf_mrao(&document, &images, key)?;
            let imported_asset = Arc::from(
                RwLock::from(Asset::new(
                    AssetMetadata::new(
                        subasset_name.clone(),
                        Some(subasset_path.into_boxed_path())
                    ),
                    crate::asset::asset_types::AssetData::TextureAssetType(imported_texture)
                ))
            );
            context.asset_batch.write().unwrap_or_else(std::sync::PoisonError::into_inner).insert(subasset_name, imported_asset);
        }

        for scene in document.scenes() {
            let asset_type = SubAssetType::Scene(scene.index());
            let subasset_path = Self::generate_subasset_filename(path, asset_type)?;
            let subasset_name = Self::generate_subasset_name(path, asset_type)?.into_string()
                .map_err(|_| GltfImportError::InvalidSubresourceName(path.to_path_buf()))?;
            log::info!("Found scene {}.", subasset_name);
            let imported_scene = Self::import_from_gltf_scene(path, &document, &scene)?;
            let imported_asset = Arc::from(
                RwLock::from(Asset::new(
                    AssetMetadata::new(
                        subasset_name.clone(),
                        Some(subasset_path.into_boxed_path())
                    ),
                    crate::asset::asset_types::AssetData::SceneAssetType(imported_scene)
                ))
            );
            context.asset_batch.write().unwrap_or_else(std::sync::PoisonError::into_inner).insert(subasset_name, imported_asset);
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

/// Where a skin is used: the scene that references it, the node carrying it and
/// the mesh it deforms.
#[derive(Debug, Clone, Copy)]
pub(super) struct SkinUsage {
    pub scene: usize,
    pub node: usize,
    pub mesh: usize,
}

/// Column-major identity matrix, used when a skin omits its inverse bind
/// matrices.
const IDENTITY_MATRIX: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

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
    pub fn new(path: impl AsRef<std::path::Path>) -> Self {
        Self {
            color_textures: HashSet::new(),
            normal_textures: HashSet::new(),
            mrao_requests: HashSet::new(),
            warned_materials: HashSet::new(),
            path: path.as_ref().into(),
        }
    }

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

    pub(super) fn generate_subasset_filename_string(gltf_file: impl AsRef<Path>, asset_type: SubAssetType) -> Option<String> {
        Some(Self::generate_subasset_filename(gltf_file, asset_type).ok()?.to_str()?.to_string())
    }

    /// Resolve the subasset name of a single glTF primitive.
    ///
    /// One mesh may contain several primitives, so both the mesh index and the
    /// primitive index are required to address a primitive uniquely.
    pub(super) fn primitive_subasset_name(gltf_file: impl AsRef<Path>, mesh_index: usize, primitive_index: usize) -> Result<String, GltfImportError> {
        Self::generate_subasset_name(&gltf_file, SubAssetType::Primitive { mesh: mesh_index, primitive: primitive_index })?
            .into_string()
            .map_err(|_| GltfImportError::InvalidSubresourceName(gltf_file.as_ref().to_path_buf()))
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

    /// Walk every scene and resolve each skin to the first node that uses it.
    ///
    /// glTF skins are referenced from *nodes*, so the association only exists in
    /// the scene graph. First occurrence wins, which keeps the mapping
    /// deterministic for skins shared by several nodes.
    pub(super) fn resolve_skin_usage(document: &gltf::Document) -> HashMap<usize, SkinUsage> {
        let mut usage = HashMap::new();
        for scene in document.scenes() {
            let mut stack: Vec<gltf::Node> = scene.nodes().collect();
            while let Some(node) = stack.pop() {
                stack.extend(node.children());
                let (Some(skin), Some(mesh)) = (node.skin(), node.mesh()) else { continue };
                usage.entry(skin.index()).or_insert(SkinUsage {
                    scene: scene.index(),
                    node: node.index(),
                    mesh: mesh.index(),
                });
            }
        }
        usage
    }

    /// Find the skin an animation belongs to.
    ///
    /// glTF animations target nodes rather than skins, so the association is
    /// derived from the skin's joint set: the first skin that owns at least one
    /// animated node wins.
    pub(super) fn find_animation_skin(animation: &gltf::Animation, document: &gltf::Document) -> Option<usize> {
        let targets: HashSet<usize> = animation
            .channels()
            .map(|channel| channel.target().node().index())
            .collect();
        if targets.is_empty() {
            return None;
        }

        document
            .skins()
            .find(|skin| skin.joints().any(|joint| targets.contains(&joint.index())))
            .map(|skin| skin.index())
    }

    /// Meshes that are placed on at least one node without a skin.
    ///
    /// Such meshes need a static asset in addition to (or instead of) a skinned
    /// one, otherwise the skinless placement would render nothing.
    pub(super) fn meshes_used_unskinned(document: &gltf::Document) -> HashSet<usize> {
        let mut meshes = HashSet::new();
        for scene in document.scenes() {
            let mut stack: Vec<gltf::Node> = scene.nodes().collect();
            while let Some(node) = stack.pop() {
                stack.extend(node.children());
                if let Some(mesh) = node.mesh() {
                    if node.skin().is_none() {
                        meshes.insert(mesh.index());
                    }
                }
            }
        }
        meshes
    }

    /// Whether a primitive carries the skinning attributes the compute shader
    /// consumes.
    pub(super) fn has_skinning_attributes(primitive: &gltf::Primitive) -> bool {
        let mut joints = false;
        let mut weights = false;
        for (semantic, _) in primitive.attributes() {
            match semantic {
                gltf::Semantic::Joints(0) => joints = true,
                gltf::Semantic::Weights(0) => weights = true,
                _ => {}
            }
        }
        joints && weights
    }

    // Primitive importer
    fn construct_position_buffer (
        primitive: &gltf::Primitive,
        buffers: &[gltf::buffer::Data]
    ) -> Vec<VertexBufferPosition> {
        let mut position_buffer: Vec<VertexBufferPosition> = Vec::new();

        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
        if let Some(iter) = reader.read_positions() {
           for vertex_position in iter {
               position_buffer.push(VertexBufferPosition { position: vertex_position });
           }
       }

        position_buffer
    }

    fn construct_attribute_buffer (
        primitive: &gltf::Primitive,
        buffers: &[gltf::buffer::Data],
        vertices: usize
    ) -> (Vec<VertexBufferOthers>, bool) {
        let mut attribute_buffer = Vec::new();
        attribute_buffer.resize(
            vertices,
            VertexBufferOthers { color: [1.0, 1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0], tangent: [0.0, 0.0, 0.0, 1.0], uv0: [0.0, 0.0] }
        );

        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
        if let Some(iter) = reader.read_colors(0) {
            for (i, c) in iter.into_rgba_f32().enumerate() {
                attribute_buffer[i].color = c;
            }
        } else {
            log::warn!("Imported mesh does not have vertex color, or the vertex color is not located at set 0.")
        }

        if let Some(iter) = reader.read_normals() {
            for (i, n) in iter.enumerate() {
                attribute_buffer[i].normal = n;
            }
        } else {
            log::warn!("Imported mesh does not have vertex normal.")
        }

        if let Some(iter) = reader.read_tex_coords(0) {
            for (i, uv) in iter.into_f32().enumerate() {
                attribute_buffer[i].uv0 = uv;
            }
        } else {
            log::warn!("Imported mesh does not have texture coordinate, or the texcoord is not located at set 0.")
        }

        if let Some(iter) = reader.read_tangents() {
            for (i, n) in iter.enumerate() {
                attribute_buffer[i].tangent = n;
            }

            (attribute_buffer, false)
        } else {
            log::info!("Imported mesh does not vertex have tangent. It will be automatically calculated on upload.");
            (attribute_buffer, true)
        }
    }

    fn construct_index_buffer (
        primitive: &gltf::Primitive,
        buffers: &[gltf::buffer::Data]
    ) -> Option<Vec<u32>> {
        let mut indices = Vec::new();

        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
        let iter = reader.read_indices()?;
        for i in iter.into_u32() {
            indices.push(i);
        }

        Some(indices)
    }

    // Scene importer
    /// Assign an arena index to every node reachable from the scene roots.
    ///
    /// BFS keeps the order deterministic and roots first. Shared by the scene
    /// importer (which emits the nodes) and the skin importer (which needs the
    /// arena indices of the joint nodes).
    pub(super) fn build_scene_index_map(
        document: &gltf::Document,
        scene: &gltf::Scene,
    ) -> Result<(Vec<usize>, HashMap<usize, u32>), GltfImportError> {
        let mut order = Vec::<usize>::new();
        let mut index_of = HashMap::<usize, u32>::new();
        let mut queue = VecDeque::<usize>::new();

        for root_node in scene.nodes() {
            queue.push_back(root_node.index());
        }

        while let Some(node_index) = queue.pop_front() {
            if index_of.contains_key(&node_index) {
                continue;
            }

            let node = document.nodes().nth(node_index)
                .ok_or(GltfImportError::IndexOutOfRange(node_index))?;
            index_of.insert(node_index, order.len() as u32);
            order.push(node_index);

            for child in node.children() {
                queue.push_back(child.index());
            }
        }

        Ok((order, index_of))
    }

    /// Import a glTF skin as a [`SkinAsset`].
    ///
    /// Joints are expressed as indices into the owning scene's node arena, so
    /// the skin does not duplicate the hierarchy: local rest transforms and the
    /// parent links stay in [`SceneAsset`], and the runtime can derive joint
    /// world transforms from `SceneGraph`.
    pub(crate) fn import_from_gltf_skin(
        gltf_file: &Path,
        document: &gltf::Document,
        scene: &gltf::Scene,
        skin: &gltf::Skin,
        buffers: &[gltf::buffer::Data],
    ) -> Result<SkinAsset, GltfImportError> {
        let (_, index_of) = Self::build_scene_index_map(document, scene)?;
        let scene_name = Self::generate_subasset_name(gltf_file, SubAssetType::Scene(scene.index()))?
            .into_string()
            .map_err(|_| GltfImportError::InvalidSubresourceName(gltf_file.to_path_buf()))?;

        let joint_nodes: Vec<gltf::Node> = skin.joints().collect();
        let inverse_bind_matrices: Vec<[[f32; 4]; 4]> =
            match skin.reader(|buffer| Some(&buffers[buffer.index()])).read_inverse_bind_matrices() {
                Some(iter) => iter.collect(),
                // glTF allows omitting the accessor, in which case the identity
                // matrix is implied for every joint.
                None => vec![IDENTITY_MATRIX; joint_nodes.len()],
            };

        if inverse_bind_matrices.len() != joint_nodes.len() {
            return Err(GltfImportError::SkinJointCountMismatch {
                joints: joint_nodes.len(),
                matrices: inverse_bind_matrices.len(),
            });
        }

        let joints = joint_nodes
            .iter()
            .zip(inverse_bind_matrices)
            .map(|(node, inverse_bind_matrix)| {
                let scene_node = index_of.get(&node.index()).copied()
                    .ok_or(GltfImportError::IndexOutOfRange(node.index()))?;
                Ok(SkinJoint { scene_node: scene_node as usize, inverse_bind_matrix })
            })
            .collect::<Result<Vec<_>, GltfImportError>>()?;

        Ok(SkinAsset { scene: scene_name, joints })
    }

    /// Import a glTF scene as a hierarchy of nodes with local transforms.
    ///
    /// Every reachable glTF node becomes a [`SceneNodeAsset`] carrying its
    /// *local* transform and children. Primitive subasset names are attached to
    /// the node that owns the mesh. No world matrix is baked here: the renderer
    /// derives world transforms from the hierarchy at runtime.
    pub(crate) fn import_from_gltf_scene(
        gltf_file: &Path,
        document: &gltf::Document,
        scene: &gltf::Scene
    ) -> Result<SceneAsset, GltfImportError> {
        // First pass: assign an arena index to every node reachable from the
        // scene roots. BFS keeps the order deterministic and roots first.
        let (order, index_of) = Self::build_scene_index_map(document, scene)?;

        // Second pass: build the arena now that every child index is known.
        let mut nodes = Vec::with_capacity(order.len());
        for gltf_index in order {
            let node = document.nodes().nth(gltf_index)
                .ok_or(GltfImportError::IndexOutOfRange(gltf_index))?;

            // A node carrying a skin references the skinned variant of each
            // primitive, so the renderer can tell both apart by asset kind.
            let skinned = node.skin().is_some();
            let mesh_subasset_names = match node.mesh() {
                Some(mesh) => mesh.primitives()
                    .map(|primitive| {
                        let asset_type = if skinned && Self::has_skinning_attributes(&primitive) {
                            SubAssetType::SkinnedPrimitive { mesh: mesh.index(), primitive: primitive.index() }
                        } else {
                            SubAssetType::Primitive { mesh: mesh.index(), primitive: primitive.index() }
                        };
                        Self::generate_subasset_name(gltf_file, asset_type)?.into_string()
                            .map_err(|_| GltfImportError::InvalidSubresourceName(gltf_file.to_path_buf()))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                None => Vec::new(),
            };

            let children = node.children()
                .map(|child| index_of.get(&child.index()).copied()
                    .ok_or(GltfImportError::IndexOutOfRange(child.index())))
                .collect::<Result<Vec<_>, _>>()?;

            nodes.push(SceneNodeAsset {
                name: node.name().map(str::to_owned),
                local_matrix: node.transform().matrix(),
                mesh_subasset_names,
                children,
            });
        }

        let roots = scene.nodes()
            .map(|node| index_of.get(&node.index()).copied()
                .ok_or(GltfImportError::IndexOutOfRange(node.index())))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(SceneAsset { nodes, roots })
    }

    pub(crate) fn import_from_gltf_primitive(
        importer: &mut GltfImporter,
        primitive: &gltf::Primitive,
        buffers: &[gltf::buffer::Data]
    ) -> Result<StaticMeshAsset, GltfImportError> {
        let position_buffer = Self::construct_position_buffer(primitive, buffers);
        let (attribute_buffer, calculate_tangent) = Self::construct_attribute_buffer(primitive, buffers, position_buffer.len());
        let index_buffer = Self::construct_index_buffer(primitive, buffers);
        let vertex_draw_count = match index_buffer.as_ref() {
            Some(b) => b.len(),
            None => position_buffer.len()
        } as u32;

        let mut ret = StaticMeshAsset {
            vp: position_buffer,
            va: attribute_buffer,
            vi: index_buffer,
            material: StaticMeshMaterial {
                diffuse_texture_name: None,
                normal_texture_name: None,
                mrao_texture_name: None
            },
            vertex_draw_count,
            vertex_color_scale: [1.0; 4]
        };

        // Process the materials
        let material = primitive.material();
        Self::warn_unsupported_material_factors(importer, &material);
        let pbr_material = material.pbr_metallic_roughness();

        ret.material.diffuse_texture_name = if let Some(base_color_texture) = pbr_material.base_color_texture() {
            let texture_name = GltfImporter::generate_subasset_name(
                &importer.path,
                SubAssetType::Texture(base_color_texture.texture().index())
            )?.into_string().map_err(|_| GltfImportError::InvalidSubresourceName(importer.path.to_path_buf()))?;
            importer.color_textures.insert(base_color_texture.texture().index());
            Some(texture_name)
        } else {
            None
        };

        ret.material.normal_texture_name = if let Some(normal_texture) = material.normal_texture() {
            let texture_name = GltfImporter::generate_subasset_name(
                &importer.path,
                SubAssetType::Texture(normal_texture.texture().index())
            )?.into_string().map_err(|_| GltfImportError::InvalidSubresourceName(importer.path.to_path_buf()))?;
            importer.normal_textures.insert(normal_texture.texture().index());
            Some(texture_name)
        } else {
            None
        };

        let mr_texture = pbr_material.metallic_roughness_texture()
            .map(|info| (info.texture().index(), info.tex_coord()));
        let ao_texture = material.occlusion_texture()
            .map(|info| (info.texture().index(), info.tex_coord()));

        for (label, tex_coord) in [
            ("metallic-roughness", mr_texture.map(|(_, tex_coord)| tex_coord)),
            ("occlusion", ao_texture.map(|(_, tex_coord)| tex_coord)),
        ] {
            if let Some(tex_coord) = tex_coord {
                if tex_coord != 0 {
                    log::warn!(
                        "{} texture uses TEXCOORD_{}; the renderer only supports TEXCOORD_0",
                        label,
                        tex_coord
                    );
                }
            }
        }

        let (mr_index, ao_index) = match (mr_texture, ao_texture) {
            (Some((mr_index, mr_tex_coord)), Some((ao_index, ao_tex_coord))) => {
                if mr_tex_coord == ao_tex_coord {
                    (Some(mr_index), Some(ao_index))
                } else {
                    log::warn!(
                        "metallic-roughness and occlusion textures use different TEXCOORD sets; occlusion will be ignored"
                    );
                    (Some(mr_index), None)
                }
            }
            (Some((mr_index, _)), None) => (Some(mr_index), None),
            (None, Some((ao_index, _))) => (None, Some(ao_index)),
            (None, None) => (None, None),
        };

        if mr_index.is_some() || ao_index.is_some() {
            let key = MraoKey { mr: mr_index, ao: ao_index };
            importer.mrao_requests.insert(key);
            ret.material.mrao_texture_name = Some(
                GltfImporter::generate_subasset_name(
                    &importer.path,
                    SubAssetType::MraoTexture(key)
                )?.into_string().map_err(|_| GltfImportError::InvalidSubresourceName(importer.path.to_path_buf()))?
            );
        }

        if calculate_tangent {
            ret.recalculate_tangents()?;
        }

        // Finalize vertex data (e.g. vertex color scaling) for zero-copy upload.
        ret.recondition();

        Ok(ret)
    }

    /// Import a primitive as a [`SkinnedMeshAsset`].
    ///
    /// The rest pose is imported exactly like a static primitive.
    pub(crate) fn import_skinned_from_gltf_primitive(
        importer: &mut GltfImporter,
        primitive: &gltf::Primitive,
        buffers: &[gltf::buffer::Data],
        skin_name: String,
    ) -> Result<SkinnedMeshAsset, GltfImportError> {
        let restpose = Self::import_from_gltf_primitive(importer, primitive, buffers)?;

        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
        let joints: Vec<[u16; 4]> = reader.read_joints(0)
            .ok_or(GltfImportError::MissingSkinningAttributes)?
            .into_u16()
            .collect();
        let weights: Vec<[f32; 4]> = reader.read_weights(0)
            .ok_or(GltfImportError::MissingSkinningAttributes)?
            .into_f32()
            .collect();

        assert_eq!(
            joints.len(),
            weights.len(),
            "JOINTS_0 and WEIGHTS_0 must have the same element count."
        );
        assert_eq!(
            joints.len(),
            restpose.vp.len(),
            "skinning attributes must have one entry per vertex."
        );

        let animation = joints
            .iter()
            .zip(weights.iter())
            .map(|(indices, weights)| Self::normalize_skinning_weights(indices, weights))
            .collect();

        Ok(SkinnedMeshAsset { skin: skin_name, restpose, animation })
    }

    /// Keep the four strongest influences and renormalize their weights.
    ///
    /// glTF allows more than four influences per vertex, but the compute shader
    /// only reads four. Renormalizing after the cut keeps the deformation from
    /// shrinking the mesh.
    fn normalize_skinning_weights(indices: &[u16; 4], weights: &[f32; 4]) -> VertexBufferSkinning {
        let mut influences: [(u32, f32); 4] = [
            (indices[0] as u32, weights[0].max(0.0)),
            (indices[1] as u32, weights[1].max(0.0)),
            (indices[2] as u32, weights[2].max(0.0)),
            (indices[3] as u32, weights[3].max(0.0)),
        ];
        influences.sort_by(|a, b| b.1.total_cmp(&a.1));

        let sum: f32 = influences.iter().map(|(_, weight)| weight).sum();
        if sum <= f32::EPSILON {
            // Degenerate input: bind the vertex to the first joint instead of
            // letting it collapse to the origin.
            return VertexBufferSkinning {
                bone_indices: [0; 4],
                bone_weights: [1.0, 0.0, 0.0, 0.0],
            };
        }

        let mut skinning = VertexBufferSkinning {
            bone_indices: [0; 4],
            bone_weights: [0.0; 4],
        };
        for (slot, (index, weight)) in influences.iter().enumerate() {
            skinning.bone_indices[slot] = *index;
            skinning.bone_weights[slot] = weight / sum;
        }
        skinning
    }

    fn warn_unsupported_material_factors(importer: &mut GltfImporter, material: &gltf::Material) {
        const EPSILON: f32 = 1e-6;

        let material_key = material.index().unwrap_or(usize::MAX);
        if !importer.warned_materials.insert(material_key) {
            return;
        }

        let material_index = material.index()
            .map(|index| index.to_string())
            .unwrap_or_else(|| String::from("<default>"));
        let material_name = material.name().unwrap_or("<unnamed>");
        let pbr = material.pbr_metallic_roughness();

        let metallic_factor = pbr.metallic_factor();
        if (metallic_factor - 1.0).abs() > EPSILON {
            log::warn!(
                "glTF material {} (name: {}) uses unsupported metallicFactor {}; the value will be ignored",
                material_index,
                material_name,
                metallic_factor
            );
        }

        let roughness_factor = pbr.roughness_factor();
        if (roughness_factor - 1.0).abs() > EPSILON {
            log::warn!(
                "glTF material {} (name: {}) uses unsupported roughnessFactor {}; the value will be ignored",
                material_index,
                material_name,
                roughness_factor
            );
        }

        if let Some(occlusion_texture) = material.occlusion_texture() {
            let occlusion_strength = occlusion_texture.strength();
            if (occlusion_strength - 1.0).abs() > EPSILON {
                log::warn!(
                    "glTF material {} (name: {}) uses unsupported occlusion strength {}; the value will be ignored",
                    material_index,
                    material_name,
                    occlusion_strength
                );
            }
        }
    }

    fn import_from_gltf_mrao(
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

    // Texture importer
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
