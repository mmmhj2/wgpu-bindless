use std::{collections::HashMap, num::NonZero, sync::Arc};

use crate::{
    asset::asset_types::{AssetData, scene_asset::SceneAsset}, renderer::{
        drawable::{drawable_mesh_traits::{Mesh, DrawableMesh}, static_mesh::StaticMesh}, pipeline::pbr_material::PBRMaterial, resources::{asset_cache::MeshUploadContext, mesh_layout::MeshLayout}, scene::graph::SceneGraph,
    },
};

/// Size of one column-major `mat4x4<f32>` entry in the model-matrix buffer.
const MODEL_MATRIX_SIZE: u64 = std::mem::size_of::<[[f32; 4]; 4]>() as u64;

/// A single draw call covering one or more instances of the same static-mesh
/// primitive.
///
/// All instances share the geometry and material baked into the primitive's
/// [`StaticMesh`] asset. Their model matrices live in one read-only storage
/// buffer and are indexed by `@builtin(instance_index)`.
///
/// Only opaque and alpha-mask primitives may use this type. Alpha-blended
/// transparent instances must not be grouped into a batch: they require a
/// global back-to-front draw order (see `docs/instancing_design.md`).
#[derive(Clone)]
pub struct StaticMeshBatch {
    mesh: Arc<StaticMesh>,
    // Owned explicitly so the buffer outlives its bind group. The bind group
    // also keeps the resource alive, but keeping the handle makes ownership
    // and future per-frame updates straightforward.
    #[allow(dead_code)]
    model_matrix_buffer: wgpu::Buffer,
    model_matrix_bind_group: wgpu::BindGroup,
    instance_count: u32,
}

/// One planned batch before GPU resources are created.
struct BatchPlan<'a> {
    mesh_subasset_name: &'a str,
    world_matrices: Vec<[[f32; 4]; 4]>,
}

impl StaticMeshBatch {
    fn generate_bind_group(
        device: &wgpu::Device,
        mmgr: &MeshLayout,
        buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model matrix batch bind group"),
            layout: mmgr.get_bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: 0,
                        size: NonZero::new(buffer.size()),
                    }),
                },
            ],
        })
    }

    /// Create a batch from explicit world matrices.
    ///
    /// Matrices are column-major, matching both `cgmath` and WGSL. They must
    /// not be empty. Callers are responsible for only passing opaque or
    /// alpha-mask instances; alpha-blended instances must use a separate
    /// transparent path.
    ///
    /// Panics if `world_matrices` is empty, if the instance count or buffer size
    /// exceeds device limits, or if the mesh asset is missing / not a static
    /// mesh asset.
    pub fn create_from_matrices(
        ctx: &mut MeshUploadContext<'_>,
        mesh_subasset_name: &str,
        world_matrices: &[[[f32; 4]; 4]],
    ) -> Self {
        assert!(
            !world_matrices.is_empty(),
            "cannot create a static mesh batch with zero instances for '{mesh_subasset_name}'"
        );

        let instance_count = u32::try_from(world_matrices.len())
            .expect("static mesh batch instance count exceeds u32::MAX");
        let buffer_size = MODEL_MATRIX_SIZE * world_matrices.len() as u64;
        let max_binding_size = u64::from(ctx.device.limits().max_storage_buffer_binding_size);
        assert!(
            buffer_size <= max_binding_size,
            "static mesh batch for '{mesh_subasset_name}' requires {buffer_size} bytes, \
             exceeding max_storage_buffer_binding_size ({max_binding_size})"
        );

        let asset = ctx
            .assets
            .get_asset(mesh_subasset_name)
            .unwrap_or_else(|| panic!("static mesh asset '{mesh_subasset_name}' not found."));
        let asset = asset
            .read()
            .expect("failed to acquire read lock for static mesh asset.");
        let mesh_asset = match asset.get_data() {
            AssetData::StaticMeshAssetType(m) => m,
            _ => panic!("asset '{mesh_subasset_name}' is not a static mesh asset."),
        };

        let model_matrix_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("model matrix batch buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let model_matrix_bind_group =
            Self::generate_bind_group(ctx.device, ctx.mesh_layout, &model_matrix_buffer);
        ctx.queue.write_buffer(
            &model_matrix_buffer,
            0,
            bytemuck::cast_slice(world_matrices),
        );

        Self {
            mesh: ctx.static_mesh_from_asset(mesh_subasset_name, mesh_asset),
            model_matrix_buffer,
            model_matrix_bind_group,
            instance_count,
        }
    }

    /// Group every mesh instance in `graph` by primitive and build one opaque
    /// batch per unique primitive.
    ///
    /// Batch order follows the first occurrence in [`SceneGraph::mesh_instances`],
    /// so the result is deterministic. Only opaque / alpha-mask instances are
    /// expected; transparent instances must not go through this path.
    pub fn create_opaque_from_scene_graph(
        ctx: &mut MeshUploadContext<'_>,
        graph: &SceneGraph,
    ) -> Vec<Self> {
        let plans = plan_batches(graph);
        log::info!(
            "Building {} static mesh batches from {} mesh instances",
            plans.len(),
            graph.mesh_instance_count()
        );

        plans
            .into_iter()
            .map(|plan| {
                log::debug!(
                    "Static mesh batch '{}' covers {} instances",
                    plan.mesh_subasset_name,
                    plan.world_matrices.len()
                );
                Self::create_from_matrices(ctx, plan.mesh_subasset_name, &plan.world_matrices)
            })
            .collect()
    }

    /// Convenience wrapper around [`Self::create_opaque_from_scene_graph`] for
    /// callers that only have a [`SceneAsset`].
    pub fn create_opaque_from_scene_asset(
        ctx: &mut MeshUploadContext<'_>,
        scene: &SceneAsset,
    ) -> Vec<Self> {
        let graph = SceneGraph::from_asset(scene);
        Self::create_opaque_from_scene_graph(ctx, &graph)
    }

    /// Number of instances covered by this batch. Always at least one.
    pub fn instance_count(&self) -> u32 {
        self.instance_count
    }
}

/// Group mesh instances by primitive name, preserving the extraction order.
fn plan_batches<'a>(graph: &'a SceneGraph) -> Vec<BatchPlan<'a>> {
    let mut plans: Vec<BatchPlan<'a>> = Vec::new();
    let mut index_of: HashMap<&'a str, usize> = HashMap::new();

    for instance in graph.mesh_instances() {
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

impl Mesh for StaticMeshBatch {
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer] {
        self.mesh.get_vertex_buffer()
    }
    fn get_vertex_draw_count(&self) -> u32 {
        self.mesh.get_vertex_draw_count()
    }
    fn get_index_buffer(&self) -> Option<&wgpu::Buffer> {
        self.mesh.get_index_buffer()
    }
    fn get_vertex_type(&self) -> super::vertex_types::VertexType {
        self.mesh.get_vertex_type()
    }
}

impl DrawableMesh for StaticMeshBatch {
    fn get_material(&self) -> &PBRMaterial {
        self.mesh.get_material()
    }
    fn get_model_matrix_bind_group(&self) -> &wgpu::BindGroup {
        &self.model_matrix_bind_group
    }
    fn get_instance_range(&self) -> std::ops::Range<u32> {
        0..self.instance_count
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

        let plans = plan_batches(&graph);
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

        let plans = plan_batches(&graph);
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

        let plans = plan_batches(&graph);
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

        assert!(plan_batches(&graph).is_empty());
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

        let plans = plan_batches(&graph);
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
