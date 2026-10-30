//! Read-only runtime view of a [`SceneAsset`] hierarchy.
//!
//! The asset stores local transforms only. This module validates the tree,
//! computes world transforms once during construction, and exposes read-only
//! accessors. Any scene change requires rebuilding the graph (and the renderer
//! instances derived from it).

use std::collections::{HashMap, HashSet};

use cgmath::Matrix4;

use crate::{asset::asset_types::{AssetData, AssetKind, scene_asset::SceneAsset}, renderer::{drawable::{draw_list::DrawList, skinned_mesh::{SkinnedAnimation, SkinnedMesh}, static_mesh_batch::StaticMeshBatch}, resources::asset_cache::MeshUploadContext}};

/// One node of the runtime scene graph.
#[derive(Debug)]
pub struct SceneNode {
    name: Option<String>,
    local_transform: Matrix4<f32>,
    world_transform: Matrix4<f32>,
    parent: Option<usize>,
    children: Vec<usize>,
    mesh_subasset_names: Vec<String>,
}

impl SceneNode {
    pub fn name(&self) -> Option<&str> { self.name.as_deref() }
    pub fn local_transform(&self) -> &Matrix4<f32> { &self.local_transform }
    pub fn world_transform(&self) -> &Matrix4<f32> { &self.world_transform }
    pub fn parent(&self) -> Option<usize> { self.parent }
    pub fn children(&self) -> &[usize] { &self.children }
    pub fn mesh_subasset_names(&self) -> &[String] { &self.mesh_subasset_names }
}

/// A mesh primitive referenced by a scene node.
#[derive(Debug, Clone, Copy)]
pub struct MeshInstanceRef<'a> {
    /// Index of the owning node in [`SceneGraph::nodes`].
    pub node_index: usize,
    /// Static-mesh subasset name (one per primitive).
    pub mesh_subasset_name: &'a str,
    /// World matrix of the owning node, column-major.
    pub world_matrix: [[f32; 4]; 4],
}

/// Read-only runtime hierarchy built from a [`SceneAsset`].
///
/// World transforms are computed exactly once in [`SceneGraph::from_asset`].
/// There is intentionally no mutation API: rebuilding the graph is the only
/// way to reflect a scene change.
#[derive(Debug)]
pub struct SceneGraph {
    nodes: Vec<SceneNode>,
    roots: Vec<usize>,
}

/// One planned batch before GPU resources are created.
#[derive(Debug)]
struct BatchPlan<'a> {
    mesh_subasset_name: &'a str,
    world_matrices: Vec<[[f32; 4]; 4]>,
}

/// Group mesh instances by primitive name, preserving the extraction order.
///
/// `can_batch` predicate determines whether a given asset can be batched.
fn plan_batches<'a>(
    graph: &'a SceneGraph,
    can_batch: impl Fn(&str) -> bool,
) -> Vec<BatchPlan<'a>> {
    let mut plans: Vec<BatchPlan<'a>> = Vec::new();
    let mut index_of: HashMap<&'a str, usize> = HashMap::new();

    for instance in graph.mesh_instances() {
        if !can_batch(instance.mesh_subasset_name) {
            continue;
        }
        let index = match index_of.get(instance.mesh_subasset_name) {
            Some(&index) => index,
            None => {
                let index = plans.len();
                index_of.insert(instance.mesh_subasset_name, index);
                plans.push(BatchPlan {
                    mesh_subasset_name: instance.mesh_subasset_name,
                    world_matrices: Vec::new(),
                });
                index
            }
        };
        plans[index].world_matrices.push(instance.world_matrix);
    }

    plans
}

impl SceneGraph {
    /// Build the runtime graph and compute every world transform.
    ///
    /// Panics if the asset hierarchy is malformed: an index is out of range, a
    /// node has multiple parents, a root is also a child, roots are duplicated,
    /// or a node is not reachable from any root.
    pub fn from_asset(asset: &SceneAsset) -> Self {
        let node_count = asset.nodes.len();

        let mut nodes: Vec<SceneNode> = asset
            .nodes
            .iter()
            .map(|node| SceneNode {
                name: node.name.clone(),
                local_transform: Matrix4::from(node.local_matrix),
                // Placeholder; replaced by the traversal below.
                world_transform: Matrix4::from(node.local_matrix),
                parent: None,
                children: node.children.iter().map(|&child| child as usize).collect(),
                mesh_subasset_names: node.mesh_subasset_names.clone(),
            })
            .collect();

        // Validate child indices and assign the unique parent of each node.
        for parent_index in 0..node_count {
            let children = nodes[parent_index].children.clone();
            for child in children {
                assert!(
                    child < node_count,
                    "scene node {parent_index} references child {child}, but the scene only has {node_count} nodes"
                );

                let existing_parent = nodes[child].parent;
                assert!(
                    existing_parent.is_none(),
                    "scene node {child} has multiple parents ({} and {parent_index})",
                    existing_parent.expect("parent was checked"),
                );
                nodes[child].parent = Some(parent_index);
            }
        }

        // Validate roots.
        let mut roots = Vec::with_capacity(asset.roots.len());
        let mut root_seen = vec![false; node_count];
        for &root in &asset.roots {
            let root = root as usize;
            assert!(
                root < node_count,
                "scene root {root} is out of range (node count {node_count})"
            );
            assert!(!root_seen[root], "scene root {root} is listed more than once");

            let existing_parent = nodes[root].parent;
            assert!(
                existing_parent.is_none(),
                "scene root {root} is also a child of node {}",
                existing_parent.expect("parent was checked"),
            );

            root_seen[root] = true;
            roots.push(root);
        }

        // Compute world transforms, rejecting cycles and orphan nodes at the
        // same time. Each node must be reached exactly once from the roots.
        let mut visited = vec![false; node_count];
        let mut stack: Vec<(usize, Option<Matrix4<f32>>)> = roots
            .iter()
            .rev()
            .map(|&root| (root, None))
            .collect();

        while let Some((index, parent_world)) = stack.pop() {
            assert!(
                !visited[index],
                "scene node {index} is reachable more than once; the hierarchy is not a tree"
            );
            visited[index] = true;

            let local = nodes[index].local_transform;
            let world = parent_world.map_or(local, |parent_world| parent_world * local);
            nodes[index].world_transform = world;

            let children = nodes[index].children.clone();
            for child in children.into_iter().rev() {
                stack.push((child, Some(world)));
            }
        }

        if let Some(unreachable) = visited.iter().position(|visited| !visited) {
            panic!("scene node {unreachable} is not reachable from any root");
        }

        Self { nodes, roots }
    }

    pub fn node_count(&self) -> usize { self.nodes.len() }
    pub fn roots(&self) -> &[usize] { &self.roots }
    pub fn nodes(&self) -> &[SceneNode] { &self.nodes }

    /// Panics if `index` is out of range.
    pub fn node(&self, index: usize) -> &SceneNode {
        self.nodes.get(index).unwrap_or_else(|| {
            panic!(
                "scene node index {index} is out of range (node count {})",
                self.nodes.len()
            )
        })
    }

    /// Total number of mesh primitives referenced by the hierarchy.
    pub fn mesh_instance_count(&self) -> usize {
        self.nodes
            .iter()
            .map(|node| node.mesh_subasset_names.len())
            .sum()
    }

    /// Iterate every mesh primitive referenced by the hierarchy.
    ///
    /// Order is deterministic: node order first, then the node's primitive
    /// order. This is the single extraction point for renderer data.
    pub fn mesh_instances(&self) -> impl Iterator<Item = MeshInstanceRef<'_>> + '_ {
        self.nodes.iter().enumerate().flat_map(|(node_index, node)| {
            node.mesh_subasset_names
                .iter()
                .map(move |mesh_subasset_name| MeshInstanceRef {
                    node_index,
                    mesh_subasset_name: mesh_subasset_name.as_str(),
                    world_matrix: *AsRef::<[[f32; 4]; 4]>::as_ref(&node.world_transform),
                })
        })
    }

    /// Consume the scene graph into a draw list for drawing
    pub fn into_draw_list(
        self,
        ctx: &mut MeshUploadContext<'_>,
    ) -> DrawList {
        // Skinned meshes are never instanced and bypass `plan_batches` entirely.
        let mut skinned_instances: Vec<(String, [[f32; 4]; 4])> = Vec::new();
        let mut skinned_names: HashSet<&str> = HashSet::new();
        for instance in self.mesh_instances() {
            if ctx.asset_kind(instance.mesh_subasset_name) == Some(AssetKind::SkinnedMesh) {
                skinned_names.insert(instance.mesh_subasset_name);
                skinned_instances.push((instance.mesh_subasset_name.to_owned(), instance.world_matrix));
            }
        }

        let plans = plan_batches(&self, |name| !skinned_names.contains(name));
        log::info!(
            "Building {} static mesh batches and {} skinned meshes from {} mesh instances",
            plans.len(),
            skinned_instances.len(),
            self.mesh_instance_count()
        );

        let static_opaque_batch = plans
            .into_iter()
            .map(|plan| {
                log::debug!(
                    "Static mesh batch '{}' covers {} instances",
                    plan.mesh_subasset_name,
                    plan.world_matrices.len()
                );
                StaticMeshBatch::create_from_matrices(ctx, plan.mesh_subasset_name, &plan.world_matrices)
            })
            .collect();

        // Resolve which baked animation drives each skin once, before any GPU
        // resource is created.
        let animations = ctx.animations_by_skin();

        let mut skinned_opaque = Vec::with_capacity(skinned_instances.len());
        for (name, world_matrix) in skinned_instances {
            let asset = ctx.assets.get_asset(&name)
                .unwrap_or_else(|| panic!("skinned mesh asset '{name}' not found."));
            let asset = asset.read().expect("failed to acquire read lock for skinned mesh asset.");
            let AssetData::SkinnedMeshAssetType(mesh_asset) = asset.get_data() else {
                panic!("asset '{name}' is not a skinned mesh asset.");
            };

            let geometry = ctx.skinned_mesh_from_asset(&name, mesh_asset);
            let material = ctx.material_from_asset(&mesh_asset.restpose.material);
            let animation = animations
                .get(&mesh_asset.skin)
                .and_then(|(animation_name, frames, fps)| {
                    ctx.skinned_animation_texture(animation_name).map(|texture| SkinnedAnimation {
                        texture,
                        frames: *frames,
                        fps: *fps,
                    })
                });
            skinned_opaque.push(SkinnedMesh::new(ctx, geometry, material, world_matrix, animation));
        }

        DrawList::new(static_opaque_batch, skinned_opaque)
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use crate::asset::asset_types::scene_asset::{SceneAsset, SceneNodeAsset};
    use cgmath::Matrix4;

    fn identity() -> [[f32; 4]; 4] {
        *AsRef::<[[f32; 4]; 4]>::as_ref(&Matrix4::<f32>::from_scale(1.0))
    }

    fn translation(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
        *AsRef::<[[f32; 4]; 4]>::as_ref(&Matrix4::from_translation(cgmath::vec3(x, y, z)))
    }

    fn translation_of(matrix: &[[f32; 4]; 4]) -> [f32; 3] {
        [matrix[3][0], matrix[3][1], matrix[3][2]]
    }

    fn assert_translation(matrix: &[[f32; 4]; 4], expected: [f32; 3]) {
        let actual = translation_of(matrix);
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "expected translation {expected:?}, got {actual:?}"
            );
        }
    }

    fn node(
        name: &str,
        local_matrix: [[f32; 4]; 4],
        meshes: &[&str],
        children: &[u32],
    ) -> SceneNodeAsset {
        SceneNodeAsset {
            name: Some(name.to_string()),
            local_matrix,
            mesh_subasset_names: meshes.iter().map(|name| name.to_string()).collect(),
            children: children.to_vec(),
        }
    }

    #[test]
    fn groups_shared_mesh_instances() {
        let asset = SceneAsset {
            nodes: vec![
                node("root", translation(10.0, 0.0, 0.0), &["mesh.a"], &[1]),
                node("child", translation(0.0, 2.0, 0.0), &["mesh.a"], &[]),
                node("other", translation(-3.0, 0.0, 0.0), &["mesh.b"], &[]),
            ],
            roots: vec![0, 2],
        };
        let graph = SceneGraph::from_asset(&asset);

        let plans = plan_batches(&graph, |_| true);
        assert_eq!(plans.len(), 2);

        assert_eq!(plans[0].mesh_subasset_name, "mesh.a");
        assert_eq!(plans[0].world_matrices.len(), 2);
        assert_translation(&plans[0].world_matrices[0], [10.0, 0.0, 0.0]);
        assert_translation(&plans[0].world_matrices[1], [10.0, 2.0, 0.0]);

        assert_eq!(plans[1].mesh_subasset_name, "mesh.b");
        assert_eq!(plans[1].world_matrices.len(), 1);
        assert_translation(&plans[1].world_matrices[0], [-3.0, 0.0, 0.0]);
    }

    #[test]
    fn preserves_first_seen_order() {
        let asset = SceneAsset {
            nodes: vec![
                node("z-first", identity(), &["mesh.z"], &[]),
                node("a-second", identity(), &["mesh.a"], &[]),
            ],
            roots: vec![0, 1],
        };
        let graph = SceneGraph::from_asset(&asset);

        let plans = plan_batches(&graph, |_| true);
        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].mesh_subasset_name, "mesh.z");
        assert_eq!(plans[1].mesh_subasset_name, "mesh.a");
    }

    #[test]
    fn repeated_primitive_on_same_node_merges() {
        let asset = SceneAsset {
            nodes: vec![node(
                "multi",
                identity(),
                &["mesh.a", "mesh.a"],
                &[],
            )],
            roots: vec![0],
        };
        let graph = SceneGraph::from_asset(&asset);

        let plans = plan_batches(&graph, |_| true);
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].world_matrices.len(), 2);
        assert_eq!(plans[0].world_matrices[0], plans[0].world_matrices[1]);
    }

    #[test]
    fn empty_scene_produces_no_batches() {
        let asset = SceneAsset {
            nodes: Vec::new(),
            roots: Vec::new(),
        };
        let graph = SceneGraph::from_asset(&asset);

        assert!(plan_batches(&graph, |_| true).is_empty());
    }

    #[test]
    fn per_mesh_matrix_order_matches_extraction_order() {
        let asset = SceneAsset {
            nodes: vec![
                node("a0", translation(1.0, 0.0, 0.0), &["mesh.a"], &[1]),
                node("a1", translation(0.0, 2.0, 0.0), &["mesh.a"], &[]),
                node("b0", translation(0.0, 0.0, 3.0), &["mesh.b"], &[]),
                node("a2", translation(4.0, 0.0, 0.0), &["mesh.a"], &[]),
            ],
            roots: vec![0, 2, 3],
        };
        let graph = SceneGraph::from_asset(&asset);

        let plans = plan_batches(&graph, |_| true);
        assert_eq!(
            plans
                .iter()
                .map(|plan| plan.mesh_subasset_name)
                .collect::<Vec<_>>(),
            vec!["mesh.a", "mesh.b"]
        );

        for plan in &plans {
            let expected: Vec<_> = graph
                .mesh_instances()
                .filter(|instance| instance.mesh_subasset_name == plan.mesh_subasset_name)
                .map(|instance| instance.world_matrix)
                .collect();
            assert_eq!(plan.world_matrices, expected);
        }
    }
}
