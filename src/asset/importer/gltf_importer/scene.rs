//! Scene and skin import.
//!
//! A glTF scene becomes a node arena with *local* transforms; the renderer
//! derives world transforms at runtime. A skin references those nodes by arena
//! index, so the hierarchy is never duplicated.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

use crate::asset::asset_types::{
    scene_asset::{SceneAsset, SceneNodeAsset},
    skin_asset::{SkinAsset, SkinJoint},
};

use super::common::GltfImportError;
use super::importer::GltfImporter;
use super::naming::SubAssetType;

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

impl GltfImporter {
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
}
