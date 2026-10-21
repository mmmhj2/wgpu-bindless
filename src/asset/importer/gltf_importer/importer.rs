use std::{collections::{HashSet, VecDeque}, ffi::OsString, num::NonZeroU32, path::{Path, PathBuf}, sync::{Arc, RwLock}};

use crate::asset::{asset_types::{Asset, AssetMetadata, scene_asset::{SceneAsset, SceneMeshInstanceAsset}, static_mesh_asset::{StaticMeshAsset, StaticMeshMaterial}, texture_asset::{AddressMode, FilterMode, Sampler, TexelData, TextureAsset, TextureDimension::D2, TextureFormat}, vertex::{VertexBufferOthers, VertexBufferPosition}}, error::ImportError, importer::{AssetImporter, ImporterContext, gltf_importer::{common::{GltfImageView, GltfImportError}, mrao::build_mrao_texture_asset}, vertex_reconditioner::{ReconditionerError, TangentRecalculator}}};

pub struct GltfImporter {
    // Guard it by mutex if we want to parallelize it.
    pub color_textures: HashSet<String>,
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
    const IMPORTER_VERSION: u32 = 2;

    fn import(mut self, context: &ImporterContext) -> Result<(), ImportError> {
        if !self.path.is_file() {
            return Err(Box::new(GltfImportError::SourceNotAFile(self.path.to_path_buf())));
        }
        context.record_source(&self.path);

        // Clone the path so borrowing it does not conflict with `&mut self` passed below.
        let source_path = self.path.clone();
        let path = source_path.as_ref();
        let (document, buffers, textures) = gltf::import(path)
            .map_err(|source| GltfImportError::Read { path: path.to_path_buf(), source })?;

        for mesh in document.meshes() {
            log::info!("Found mesh {} with {} primitives, importing...", mesh.index(), mesh.primitives().len());

            for primitive in mesh.primitives() {
                let asset_type = SubAssetType::Primitive { mesh: mesh.index(), primitive: primitive.index() };
                let subasset_path = Self::generate_subasset_filename(path, asset_type)?;
                let subasset_name = Self::generate_subasset_name(path, asset_type)?.into_string()
                    .map_err(|_| GltfImportError::InvalidSubresourceName(path.to_path_buf()))?;
                let imported_mesh = Self::import_from_gltf_primitive(&mut self, &primitive, &buffers)?;
                let imported_asset = Arc::from(
                    RwLock::from(Asset::new(
                        AssetMetadata::new(
                            subasset_name.clone(),
                            Some(subasset_path.into_boxed_path())
                        ),
                        crate::asset::asset_types::AssetData::StaticMeshAssetType(imported_mesh)
                    ))
                );
                context.asset_batch.write().unwrap_or_else(std::sync::PoisonError::into_inner).insert(subasset_name, imported_asset);
            }
        }

        log::info!("Following textures are marked as color texture: {:?}", self.color_textures);

        for texture in document.textures() {
            let asset_type = SubAssetType::Texture(texture.index());
            let subasset_path = Self::generate_subasset_filename(path, asset_type)?;
            let subasset_name = Self::generate_subasset_name(path, asset_type)?.into_string()
                .map_err(|_| GltfImportError::InvalidSubresourceName(path.to_path_buf()))?;
            log::info!("Found texture {}.", subasset_name);
            let imported_texture = Self::import_from_gltf_texture(&texture, &textures, self.color_textures.contains(&subasset_name))?;
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
            let imported_texture = Self::import_from_gltf_mrao(&document, &textures, key)?;
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
    /// A GLTF scene, flattened into a list of mesh instances with baked transforms.
    Scene(usize),
    /// A renderer-layout MRAO texture derived from optional MR and AO textures.
    MraoTexture(MraoKey),
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
    pub fn new(path: impl AsRef<std::path::Path>) -> Self {
        Self {
            color_textures: HashSet::new(),
            mrao_requests: HashSet::new(),
            warned_materials: HashSet::new(),
            path: path.as_ref().into(),
        }
    }

    pub(crate) fn generate_subasset_filename(gltf_file: &Path, asset_type: SubAssetType) -> Result<PathBuf, GltfImportError> {
        Ok(gltf_file.with_file_name(Self::generate_subasset_name(gltf_file, asset_type)?).with_added_extension("asset"))
    }

    pub(crate) fn generate_subasset_name(gltf_file: &Path, asset_type: SubAssetType) -> Result<OsString, GltfImportError> {
        let new_suffix = match asset_type {
            SubAssetType::Texture(index) => format!(".tex_{}", index),
            SubAssetType::Mesh(index) => format!(".mesh_{}", index),
            SubAssetType::Primitive { mesh, primitive } => format!(".prim_{}_{}", mesh, primitive),
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

        let mut new_filename = gltf_file.file_name()
            .ok_or_else(|| GltfImportError::InvalidSubresourceName(gltf_file.to_path_buf()))?
            .to_os_string();
        new_filename.push(new_suffix);
        Ok(new_filename)
    }

    pub(crate) fn generate_subasset_filename_string(gltf_file: &Path, asset_type: SubAssetType) -> Option<String> {
        Some(Self::generate_subasset_filename(gltf_file, asset_type).ok()?.to_str()?.to_string())
    }

    /// Resolve the subasset name of a single glTF primitive.
    ///
    /// One mesh may contain several primitives, so both the mesh index and the
    /// primitive index are required to address a primitive uniquely.
    pub(crate) fn primitive_subasset_name(gltf_file: &Path, mesh_index: usize, primitive_index: usize) -> Result<String, GltfImportError> {
        Self::generate_subasset_name(gltf_file, SubAssetType::Primitive { mesh: mesh_index, primitive: primitive_index })?
            .into_string()
            .map_err(|_| GltfImportError::InvalidSubresourceName(gltf_file.to_path_buf()))
    }

    /// Resolve the subasset name of a single glTF scene.
    pub fn scene_subasset_name(gltf_file: &Path, scene_index: usize) -> Result<String, GltfImportError> {
        Self::generate_subasset_name(gltf_file, SubAssetType::Scene(scene_index))?
            .into_string()
            .map_err(|_| GltfImportError::InvalidSubresourceName(gltf_file.to_path_buf()))
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
            log::warn!("Imported mesh does not vertex have normal.")
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
    /// Flatten a glTF scene into a list of mesh instances with baked world
    /// transforms.
    ///
    /// The scene graph is traversed breadth-first from the scene roots. Each
    /// primitive of every encountered mesh produces one instance, so the
    /// renderer never needs to re-parse the source glTF to walk the hierarchy.
    pub(crate) fn import_from_gltf_scene(
        gltf_file: &Path,
        document: &gltf::Document,
        scene: &gltf::Scene
    ) -> Result<SceneAsset, GltfImportError> {
        let mut instances = Vec::new();

        let mut transform_queue = VecDeque::<cgmath::Matrix4<f32>>::new();
        let mut node_queue = VecDeque::<usize>::new();

        for root_node in scene.nodes() {
            transform_queue.push_back(root_node.transform().matrix().into());
            node_queue.push_back(root_node.index());

            while let (Some(node_index), Some(current_transform)) = (node_queue.pop_front(), transform_queue.pop_front()) {
                let current_node = document.nodes().nth(node_index)
                    .ok_or(GltfImportError::IndexOutOfRange(node_index))?;

                if let Some(mesh) = current_node.mesh() {
                    for primitive in mesh.primitives() {
                        instances.push(SceneMeshInstanceAsset {
                            world_matrix: *AsRef::<[f32; 16]>::as_ref(&current_transform),
                            mesh_subasset_name: Self::primitive_subasset_name(gltf_file, mesh.index(), primitive.index())?,
                        });
                    }
                }

                for child in current_node.children() {
                    transform_queue.push_back(current_transform * cgmath::Matrix4::from(child.transform().matrix()));
                    node_queue.push_back(child.index());
                }
            }
        }

        Ok(SceneAsset { instances })
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
            importer.color_textures.insert(texture_name.clone());
            Some(texture_name)
        } else {
            None
        };

        ret.material.normal_texture_name = if let Some(normal_texture) = material.normal_texture() {
            Some(
                GltfImporter::generate_subasset_name(
                    &importer.path,
                    SubAssetType::Texture(normal_texture.texture().index())
                )?.into_string().map_err(|_| GltfImportError::InvalidSubresourceName(importer.path.to_path_buf()))?
            )
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
