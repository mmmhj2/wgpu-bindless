//! Read-only runtime view of a [`SceneAsset`] hierarchy.
//!
//! The asset stores local transforms only. This module validates the tree,
//! computes world transforms once during construction, and exposes read-only
//! accessors. Any scene change requires rebuilding the graph (and the renderer
//! instances derived from it).

use cgmath::Matrix4;

use crate::asset::asset_types::scene_asset::SceneAsset;

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
}
