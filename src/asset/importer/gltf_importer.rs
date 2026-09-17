use std::{collections::HashSet, ffi::OsString, sync::{Arc, RwLock}};

use crate::{asset::{asset_types::{Asset, AssetMetadata, static_mesh_asset::{StaticMeshAsset, StaticMeshMaterial}, texture_asset::{AddressMode, FilterMode, Sampler, TexelData, TextureAsset, TextureDimension::D2, TextureFormat}}, importer::{AssetImporter, ImporterContext, vertex_reconditioner::TangentRecalculator}}, renderer::mesh::vertex_types::{VertexBufferOthers, VertexBufferPosition}};

pub struct GltfImporter {
    // Guard it by mutex if we want to parallelize it.
    pub color_textures: HashSet<String>
}

impl AssetImporter for GltfImporter {
    fn import(mut self, context: &super::ImporterContext) {
        assert!(context.imported_file_path.is_file());
        
        let path = context.imported_file_path.as_ref();
        let (document, buffers, textures) = gltf::import(path).expect("Failed to open gltf file.");

        for mesh in document.meshes() {
            log::info!("Found mesh {} with {} primitives, importing...", mesh.index(), mesh.primitives().len());

            for primitive in mesh.primitives() {
                let subasset_path = Self::generate_subasset_filename(path, mesh.index(), SubAssetType::Primitive);
                let subasset_name = Self::generate_subasset_name(path, mesh.index(), SubAssetType::Primitive).into_string().expect("asset name should contain UTF-8 chars only.");
                let imported_mesh = Self::import_from_gltf_primitive(&mut self, context, &primitive, &buffers);
                let imported_asset = Arc::from(
                    RwLock::from(Asset::new(
                        AssetMetadata{
                            asset_file_path: Some(subasset_path.into_boxed_path()),
                            asset_name: subasset_name.clone()
                        },
                        crate::asset::asset_types::AssetData::StaticMeshAssetType(imported_mesh)
                    ))
                );
                context.asset_batch.write().expect("cannot acquire write lock for asset database.").insert(subasset_name, imported_asset);
            }
        }

        log::info!("Following textures are marked as color texture: {:?}", self.color_textures);

        for texture in document.textures() {
            let subasset_path = Self::generate_subasset_filename(path, texture.index(), SubAssetType::Texture);
            let subasset_name = Self::generate_subasset_name(path, texture.index(), SubAssetType::Texture).into_string().expect("asset name should contain UTF-8 chars only.");
            log::info!("Found texture {}.", subasset_name);
            let imported_texture = Self::import_from_gltf_texture(&texture, &textures, self.color_textures.contains(&subasset_name)).expect("failed to import texture asset");
            let imported_asset = Arc::from(
                RwLock::from(Asset::new(
                    AssetMetadata {
                        asset_file_path: Some(subasset_path.into_boxed_path()),
                        asset_name: subasset_name.clone()
                    },
                    crate::asset::asset_types::AssetData::TextureAssetType(imported_texture)
                ))
            );
            context.asset_batch.write().expect("cannot acquire write lock for asset database.").insert(subasset_name, imported_asset);
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
    pub fn new() -> Self {
        Self { color_textures: HashSet::new() }
    }

    pub(crate) fn generate_subasset_filename(gltf_file: &std::path::Path, index: usize, asset_type: SubAssetType) -> std::path::PathBuf {
        gltf_file.with_file_name(Self::generate_subasset_name(gltf_file, index, asset_type)).with_added_extension("asset")
    }

    pub(crate) fn generate_subasset_name(gltf_file: &std::path::Path, index: usize, asset_type: SubAssetType) -> OsString {
        assert!(gltf_file.is_file());

        let new_suffix = match asset_type {
            SubAssetType::Texture => format!(".tex_{}", index),
            SubAssetType::Mesh => todo!("Mesh import is not supported yet."),
            SubAssetType::Primitive => format!(".prim_{}", index),
        };

        let mut new_filename = gltf_file.file_name().unwrap().to_os_string();
        new_filename.push(new_suffix);
        return new_filename;
    }

    pub(crate) fn generate_subasset_filename_string(gltf_file: &std::path::Path, index: usize, asset_type: SubAssetType) -> Option<String> {
        Some(Self::generate_subasset_filename(gltf_file, index, asset_type).to_str()?.to_string())
    }

    // Primitive importer
    fn construct_position_buffer (
        primitive: &gltf::Primitive,
        buffers: &Vec<gltf::buffer::Data>
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
        buffers: &Vec<gltf::buffer::Data>,
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
        buffers: &Vec<gltf::buffer::Data>
    ) -> Option<Vec<u32>> {
        let mut indices = Vec::new();

        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
        let iter = reader.read_indices()?;
        for i in iter.into_u32() {
            indices.push(i);
        }

        Some(indices)
    }


    pub(crate) fn import_from_gltf_primitive(
        importer: &mut GltfImporter,
        context: &ImporterContext,
        primitive: &gltf::Primitive,
        buffers: &Vec<gltf::buffer::Data>
    ) -> StaticMeshAsset {
        let position_buffer = Self::construct_position_buffer(primitive, buffers);
        let (attribute_buffer, calculate_tangent) = Self::construct_attribute_buffer(primitive, buffers, position_buffer.len());
        let index_buffer = Self::construct_index_buffer(&primitive, buffers);
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
        let pbr_material = primitive.material().pbr_metallic_roughness();

        ret.material.diffuse_texture_name = if let Some(base_color_texture) = pbr_material.base_color_texture() {
            let texture_name = GltfImporter::generate_subasset_name(
                &context.imported_file_path,
                base_color_texture.texture().index(),
                crate::asset::importer::gltf_importer::SubAssetType::Texture
            ).into_string().expect("diffuse texture name should contain UTF-8 chars only");
            importer.color_textures.insert(texture_name.clone());
            Some(texture_name)
        } else {
            None
        };

        ret.material.normal_texture_name = if let Some(normal_texture) = primitive.material().normal_texture() {
            Some(
                GltfImporter::generate_subasset_name(
                    &context.imported_file_path,
                    normal_texture.texture().index(),
                    crate::asset::importer::gltf_importer::SubAssetType::Texture
                ).into_string().expect("normal texture name should contain UTF-8 chars only")
            )
        } else {
            None
        };

        let should_have_mrao_texture = primitive.material().pbr_metallic_roughness().metallic_roughness_texture().is_some() || primitive.material().occlusion_texture().is_some();
        if should_have_mrao_texture {
            todo!("MRAO texture is not implemented.")
        };
        
        if calculate_tangent {
            ret.recalculate_tangents();
        }

        return ret;
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

    fn match_texture_format(f: gltf::image::Format, is_srgb: bool) -> TextureFormat {
        match f {
            gltf::image::Format::R8G8B8A8 => if is_srgb {TextureFormat::Rgba8Srgb} else {TextureFormat::Rgba8Unorm},
            gltf::image::Format::R16G16B16A16 => TextureFormat::Rgba16Unorm,
            gltf::image::Format::R32G32B32A32FLOAT => TextureFormat::Rgba32Float,
            _ => panic!("Unsupported GLTF texture format")
        }
    }

    pub fn import_from_gltf_texture(
        texture: &gltf::Texture,
        images: &Vec<gltf::image::Data>,
        is_srgb: bool
    ) -> Result<TextureAsset, ()> {
        let image = &images[texture.index()];

        let converted_format = Self::match_texture_format(image.format, is_srgb);
        let image = &images[texture.source().index()];

        let asset = TextureAsset {
            format: converted_format,
            dimension: D2,
            width: image.width.try_into().expect("width of the texture should be at least 1"),
            height: Some(image.height.try_into().expect("height of the texture should be at least 1")),
            depth_or_array_slice: None,
            sampler: Self::import_from_gltf_sampler(&texture.sampler()),
            // One gltf image can correspond to multiple textures with different sampler.
            // So we must clone its data for safety.
            texels: TexelData::Plain(image.pixels.clone().into_boxed_slice()),
        };

        Ok(asset)
    }
}
