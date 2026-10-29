#[cfg(test)]
mod test {
    use rust_renderer::{
        asset::asset_types::scene_asset::{SceneAsset, SceneNodeAsset},
        renderer::scene::graph::SceneGraph,
    };

    fn identity() -> [[f32; 4]; 4] {
        *AsRef::<[[f32; 4]; 4]>::as_ref(&cgmath::Matrix4::<f32>::from_scale(1.0))
    }

    fn translation(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
        *AsRef::<[[f32; 4]; 4]>::as_ref(&cgmath::Matrix4::from_translation(
            cgmath::vec3(x, y, z),
        ))
    }

    fn translation_of(matrix: &cgmath::Matrix4<f32>) -> [f32; 3] {
        let columns = AsRef::<[[f32; 4]; 4]>::as_ref(matrix);
        [columns[3][0], columns[3][1], columns[3][2]]
    }

    fn assert_translation(matrix: &cgmath::Matrix4<f32>, expected: [f32; 3]) {
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
    fn parent_child_world_transform() {
        let asset = SceneAsset {
            nodes: vec![
                node("root", translation(10.0, 0.0, 0.0), &["mesh.a"], &[1]),
                node("child", translation(0.0, 2.0, 0.0), &["mesh.a"], &[]),
            ],
            roots: vec![0],
        };

        let graph = SceneGraph::from_asset(&asset);

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.roots().len(), 1);
        assert_eq!(graph.roots()[0], 0);
        assert_eq!(graph.mesh_instance_count(), 2);

        assert_translation(graph.node(0).world_transform(), [10.0, 0.0, 0.0]);
        assert_translation(graph.node(1).local_transform(), [0.0, 2.0, 0.0]);
        assert_translation(graph.node(1).world_transform(), [10.0, 2.0, 0.0]);

        let instances: Vec<_> = graph.mesh_instances().collect();
        assert_eq!(instances.len(), 2);
        assert_eq!(instances[0].node_index, 0);
        assert_eq!(instances[0].mesh_subasset_name, "mesh.a");
        assert_eq!(instances[1].node_index, 1);
        assert_eq!(instances[1].mesh_subasset_name, "mesh.a");
    }

    #[test]
    fn multiple_roots() {
        let asset = SceneAsset {
            nodes: vec![
                node("left", translation(-3.0, 0.0, 0.0), &["mesh.a"], &[]),
                node("right", translation(3.0, 0.0, 0.0), &["mesh.b"], &[]),
            ],
            roots: vec![0, 1],
        };

        let graph = SceneGraph::from_asset(&asset);

        assert_eq!(graph.roots().len(), 2);
        assert_translation(graph.node(0).world_transform(), [-3.0, 0.0, 0.0]);
        assert_translation(graph.node(1).world_transform(), [3.0, 0.0, 0.0]);

        let instances: Vec<_> = graph.mesh_instances().collect();
        assert_eq!(instances.len(), 2);
        assert_eq!(instances[0].mesh_subasset_name, "mesh.a");
        assert_eq!(instances[1].mesh_subasset_name, "mesh.b");
    }

    #[test]
    fn multi_primitive_node() {
        let asset = SceneAsset {
            nodes: vec![node("multi", identity(), &["mesh.a", "mesh.b"], &[])],
            roots: vec![0],
        };

        let graph = SceneGraph::from_asset(&asset);

        assert_eq!(graph.mesh_instance_count(), 2);
        let instances: Vec<_> = graph.mesh_instances().collect();
        assert_eq!(instances.len(), 2);
        assert_eq!(instances[0].node_index, 0);
        assert_eq!(instances[0].mesh_subasset_name, "mesh.a");
        assert_eq!(instances[1].node_index, 0);
        assert_eq!(instances[1].mesh_subasset_name, "mesh.b");
        assert_eq!(instances[0].world_matrix, instances[1].world_matrix);
    }

    #[test]
    fn transform_only_node_does_not_emit_instance() {
        let asset = SceneAsset {
            nodes: vec![
                node("empty-parent", translation(1.0, 0.0, 0.0), &[], &[1]),
                node("mesh-child", translation(0.0, 4.0, 0.0), &["mesh.a"], &[]),
            ],
            roots: vec![0],
        };

        let graph = SceneGraph::from_asset(&asset);

        assert_eq!(graph.mesh_instance_count(), 1);
        assert_translation(graph.node(1).world_transform(), [1.0, 4.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn root_out_of_range_panics() {
        let asset = SceneAsset {
            nodes: vec![node("only", identity(), &["mesh.a"], &[])],
            roots: vec![3],
        };

        let _ = SceneGraph::from_asset(&asset);
    }

    #[test]
    #[should_panic(expected = "references child")]
    fn child_out_of_range_panics() {
        let asset = SceneAsset {
            nodes: vec![node("only", identity(), &["mesh.a"], &[3])],
            roots: vec![0],
        };

        let _ = SceneGraph::from_asset(&asset);
    }

    #[test]
    #[should_panic(expected = "multiple parents")]
    fn multiple_parents_panic() {
        let asset = SceneAsset {
            nodes: vec![
                node("a", identity(), &[], &[2]),
                node("b", identity(), &[], &[2]),
                node("shared", identity(), &["mesh.a"], &[]),
            ],
            roots: vec![0, 1],
        };

        let _ = SceneGraph::from_asset(&asset);
    }

    #[test]
    #[should_panic(expected = "is also a child")]
    fn root_that_is_a_child_panics() {
        let asset = SceneAsset {
            nodes: vec![
                node("parent", identity(), &[], &[1]),
                node("child", identity(), &["mesh.a"], &[]),
            ],
            roots: vec![0, 1],
        };

        let _ = SceneGraph::from_asset(&asset);
    }

    #[test]
    #[should_panic(expected = "listed more than once")]
    fn duplicate_root_panics() {
        let asset = SceneAsset {
            nodes: vec![node("only", identity(), &["mesh.a"], &[])],
            roots: vec![0, 0],
        };

        let _ = SceneGraph::from_asset(&asset);
    }

    #[test]
    #[should_panic(expected = "not reachable from any root")]
    fn unreachable_node_panics() {
        let asset = SceneAsset {
            nodes: vec![
                node("root", identity(), &["mesh.a"], &[]),
                node("orphan", identity(), &["mesh.b"], &[]),
            ],
            roots: vec![0],
        };

        let _ = SceneGraph::from_asset(&asset);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn node_access_out_of_range_panics() {
        let asset = SceneAsset {
            nodes: vec![node("only", identity(), &["mesh.a"], &[])],
            roots: vec![0],
        };
        let graph = SceneGraph::from_asset(&asset);

        let _ = graph.node(1);
    }
}
