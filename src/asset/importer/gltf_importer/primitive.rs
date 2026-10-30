//! Primitive import: geometry streams, skinning attributes and material slots.

use crate::asset::asset_types::{
    skinned_mesh_asset::SkinnedMeshAsset,
    static_mesh_asset::{StaticMeshAsset, StaticMeshMaterial},
    vertex::{VertexBufferOthers, VertexBufferPosition, VertexBufferSkinning},
};
use crate::asset::importer::vertex_reconditioner::{NormalRecalculator, TangentRecalculator};

use super::common::GltfImportError;
use super::importer::GltfImporter;
use super::naming::SubAssetType;
use super::texture::MraoKey;

impl GltfImporter {
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

    /// Build the non-position vertex stream.
    ///
    /// Returns `(attributes, needs_normals, needs_tangents)`: missing normals and
    /// tangents are recomputed by the importer afterwards, normals first.
    fn construct_attribute_buffer (
        primitive: &gltf::Primitive,
        buffers: &[gltf::buffer::Data],
        vertices: usize
    ) -> (Vec<VertexBufferOthers>, bool, bool) {
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

        let needs_normals = match reader.read_normals() {
            Some(iter) => {
                for (i, n) in iter.enumerate() {
                    attribute_buffer[i].normal = n;
                }
                false
            }
            None => {
                log::warn!("Imported mesh does not have vertex normal. It will be recalculated from the triangle winding.");
                true
            }
        };

        if let Some(iter) = reader.read_tex_coords(0) {
            for (i, uv) in iter.into_f32().enumerate() {
                attribute_buffer[i].uv0 = uv;
            }
        } else {
            log::warn!("Imported mesh does not have texture coordinate, or the texcoord is not located at set 0.")
        }

        let needs_tangents = match reader.read_tangents() {
            Some(iter) => {
                for (i, n) in iter.enumerate() {
                    attribute_buffer[i].tangent = n;
                }
                false
            }
            None => {
                log::info!("Imported mesh does not vertex have tangent. It will be automatically calculated on upload.");
                true
            }
        };

        (attribute_buffer, needs_normals, needs_tangents)
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

    pub(crate) fn import_from_gltf_primitive(
        importer: &mut GltfImporter,
        primitive: &gltf::Primitive,
        buffers: &[gltf::buffer::Data]
    ) -> Result<StaticMeshAsset, GltfImportError> {
        let position_buffer = Self::construct_position_buffer(primitive, buffers);
        let (attribute_buffer, calculate_normal, calculate_tangent) = Self::construct_attribute_buffer(primitive, buffers, position_buffer.len());
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

        // Normals must exist before tangents: the tangent solver orthogonalizes
        // against the vertex normal.
        if calculate_normal {
            ret.recalculate_normals()?;
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
}
