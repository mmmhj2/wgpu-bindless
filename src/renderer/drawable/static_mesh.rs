use crate::{
    asset::asset_types::static_mesh_asset::GpuStaticMesh, renderer::{
        drawable::{drawable_mesh_traits::Mesh, vertex_types::VertexType}, pipeline::pbr_material::PBRMaterial,
    }
};

/// Actual instanced mesh, whose data have already been pushed onto GPU.
#[derive(Debug)]
pub(crate) struct StaticMesh {
    geometry: GpuStaticMesh,
    material: PBRMaterial
}

impl StaticMesh {
    pub(crate) fn new(geometry: GpuStaticMesh, material: PBRMaterial) -> Self {
        Self { geometry, material }
    }

    pub fn get_material(&self) -> &PBRMaterial { &self.material }
}

impl Mesh for StaticMesh {
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer] { &self.geometry.vertex_buffers }
    fn get_vertex_draw_count(&self) -> u32 { self.geometry.vertex_draw_count }
    fn get_index_buffer(&self) -> Option<&wgpu::Buffer> { self.geometry.index_buffer.as_ref() }
    fn get_vertex_type(&self) -> VertexType { VertexType::Basic }
}

#[cfg(test)]
mod test {
    use crate::asset::importer::gltf_importer::GltfImporter;

    #[test]
    fn test_buffer_import() {
        let (document, buffers, _) = gltf::import("resource/test_two_cubes.glb").expect("Failed to import GLB file.");

        let mesh = document.meshes()
            .find(|x| x.name().unwrap_or_default() == "cube_textured")
            .expect("Cannot find cube_textured.");

        let primitive = mesh.primitives().next().expect("cube_textured has not primitives.");
        assert_eq!(primitive.mode(), gltf::mesh::Mode::Triangles);

        let mut importer = GltfImporter::new("resource/test_two_cubes.glb");
        let asset = GltfImporter::import_from_gltf_primitive(&mut importer, &primitive, &buffers).expect("primitive import should succeed");

        // Four vertices for each face.
        assert_eq!(asset.vp.len(), 4 * 6);
        assert_eq!(asset.vp.len(), asset.va.len());
        // Every face has two triangles and therefore six vertices.
        assert_eq!(asset.vi.as_ref().expect("Cannot find index buffer.").len(), 2 * 6 * 3);
        println!("Vertex positions: {:?}", &asset.vp);
        println!("Vertex attributes: {:?}", &asset.va);
        println!("Indices: {:?}", &asset.vi);
    }
}
