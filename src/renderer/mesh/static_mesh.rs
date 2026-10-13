use crate::{
    asset::{
        asset_manager::AssetManager,
        asset_types::{
            AssetData,
            GpuAssetType,
            static_mesh_asset::StaticMeshAsset,
            vertex::{VertexBufferOthers, VertexBufferPosition},
        },
        importer::vertex_reconditioner::{
            TangentRecalculator, VertexColorApplyScale, VertexReconditionable, VertexReconditionableAttributeWrite
        },
    },
    renderer::{
    device_interface::DeviceInterface,
    mesh::Mesh,
    pipeline::{
        bindless_resource_manager::BindlessResourceManager,
        pbr_material::PBRMaterial,
    }
}};

/// Actual instanced mesh, whose data have already been pushed onto GPU.
pub(crate) struct StaticMesh {
    vertex_attribute_buffers    : [wgpu::Buffer; 2],
    index_buffer                : Option<wgpu::Buffer>,
    vertex_draw_count           : u32,
    material                    : PBRMaterial
}
impl StaticMesh {
    pub fn get_material(&self) -> &PBRMaterial { &self.material }
}
impl Mesh for StaticMesh {
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer] { &self.vertex_attribute_buffers }
    fn get_vertex_draw_count(&self) -> u32 { self.vertex_draw_count }
    fn get_index_buffer(&self) -> Option<&wgpu::Buffer> { self.index_buffer.as_ref() }
    fn get_vertex_type(&self) -> super::vertex_types::VertexType { super::vertex_types::VertexType::Basic }
}

/// Upload a texture asset referenced by name and register it with the bindless manager.
///
/// Returns the (texture, sampler) bindless indices.
/// 
/// **TODO:** Manipulating asset within renderer seems to be a bad design for
/// coupling. We should consider a better design in the future, maybe by having
/// a separate asset upload stage before the renderer is initialized.
fn push_texture_asset(
    di: &DeviceInterface,
    bindless_manager: &mut BindlessResourceManager,
    assets: &AssetManager,
    name: &str
) -> (usize, usize) {
    let asset = assets.get_asset(name).unwrap_or_else(|| panic!("texture asset '{name}' not found."));
    let asset = asset.read().expect("failed to acquire read lock for texture asset.");
    let texture = match asset.get_data() {
        AssetData::TextureAssetType(t) => t,
        _ => panic!("asset '{name}' is not a texture asset."),
    };

    let uploaded = texture.upload(name, di.get_device(), di.get_queue());
    let tx = bindless_manager.push_texture(uploaded.create_view(&Default::default()))
        .expect("failed to push texture to bindless manager.");
    let sp = bindless_manager.push_sampler(di.get_device(), texture.sampler.clone().into())
        .expect("failed to push sampler to bindless manager.");
    (tx, sp)
}

/// Builder for instanced mesh.
/// It contains all data of an instanced mesh, but is stored on CPU for further reconditioning.
/// It must be committed to GPU before using. 
pub struct StaticMeshBuilder {
    vp  : Vec<VertexBufferPosition>,
    va  : Vec<VertexBufferOthers>,
    vi  : Option<Vec<u32>>,
    material : PBRMaterial,
    vertex_draw_count : u32,
    vertex_color_scale: [f32; 4],
    need_tangent : bool
}

impl StaticMeshBuilder {
    /// Build a mesh builder from an imported static mesh asset.
    ///
    /// Textures are resolved by name through `assets`; missing slots fall back
    /// to the bindless manager's default textures.
    /// 
    /// **Note**: The builder copies whole vertex buffer of the mesh on CPU. Some
    /// optimization may be needed in the future to avoid this.
    pub(crate) fn new(
        di: &DeviceInterface,
        bindless_manager: &mut BindlessResourceManager,
        mesh: &StaticMeshAsset,
        assets: &AssetManager
    ) -> Self {
        let diffuse = match &mesh.material.diffuse_texture_name {
            Some(name) => push_texture_asset(di, bindless_manager, assets, name),
            None => (bindless_manager.get_white_texture(), bindless_manager.get_default_sampler()),
        };

        let normal = match &mesh.material.normal_texture_name {
            Some(name) => push_texture_asset(di, bindless_manager, assets, name),
            None => (bindless_manager.get_default_bump_texture(), bindless_manager.get_default_sampler()),
        };

        let mrao = match &mesh.material.mrao_texture_name {
            Some(name) => push_texture_asset(di, bindless_manager, assets, name),
            None => (bindless_manager.get_default_mrao_texture(), bindless_manager.get_default_sampler()),
        };

        let material = PBRMaterial::new(
            diffuse.0,
            normal.0,
            mrao.0,
            diffuse.1,
            normal.1,
            mrao.1
        );

        Self {
            vp: mesh.vp.clone(),
            va: mesh.va.clone(),
            vi: mesh.vi.clone(),
            material,
            vertex_draw_count: mesh.vertex_draw_count,
            vertex_color_scale: mesh.vertex_color_scale,
            // The importer has already recalculated tangents where they were missing.
            need_tangent: false,
        }
    }

    fn push_buffers(
        &self,
        di: &DeviceInterface
    ) -> ([wgpu::Buffer; 2], Option<wgpu::Buffer>) {
        assert_eq!(self.va.len(), self.vp.len());

        let position_buffer = di.get_device().create_buffer(
            &wgpu::BufferDescriptor{
                label: None,
                size: (self.vp.len() as u64 * std::mem::size_of::<VertexBufferPosition>() as u64) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
                mapped_at_creation: false,
            }
        );

        let attribute_buffer = di.get_device().create_buffer(
            &wgpu::BufferDescriptor{
                label: None,
                size: (self.va.len() as u64 * std::mem::size_of::<VertexBufferOthers>() as u64) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
                mapped_at_creation: false,
            }
        );

        di.get_queue().write_buffer(&position_buffer, 0, bytemuck::cast_slice(&self.vp));
        di.get_queue().write_buffer(&attribute_buffer, 0, bytemuck::cast_slice(&self.va));

        if let Some(index) = self.vi.as_ref() {
            let index_buffer = di.get_device().create_buffer(
                &wgpu::BufferDescriptor{
                    label: None,
                    size: (index.len() as u64 * std::mem::size_of::<u32>() as u64) as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::INDEX,
                    mapped_at_creation: false,
                }
            );
            di.get_queue().write_buffer(&index_buffer, 0, bytemuck::cast_slice(&index));
            ([position_buffer, attribute_buffer], Some(index_buffer))
        } else {
            ([position_buffer, attribute_buffer], None)
        }
    }

    /// Recondition the vertex attributes, and commit the builder onto the GPU.
    pub(crate) fn recondition_and_commit(mut self, di: &DeviceInterface) -> StaticMesh {
        log::debug!("Applying vertex color scale: {:?}.", self.vertex_color_scale);
        self.rescale_vertex_color(self.vertex_color_scale);

        if self.need_tangent  { 
            log::debug!("Recalculating tangent vectors.");
            self.recalculate_tangents();
        }
        let (vp_va, vi) = self.push_buffers(di);
        StaticMesh { vertex_attribute_buffers: vp_va, index_buffer: vi, vertex_draw_count: self.vertex_draw_count, material: self.material }
    }
}
impl VertexReconditionable for StaticMeshBuilder {
    fn get_position_buffer(&self) -> &Vec<VertexBufferPosition> { &self.vp }
    fn get_attribute_buffer(&self) -> &Vec<VertexBufferOthers> { &self.va }
    fn get_index_buffer(&self) -> Option<&Vec<u32>> { self.vi.as_ref() }
}
impl VertexReconditionableAttributeWrite for StaticMeshBuilder {
    fn get_attribute_buffer_mut(&mut self) -> &mut Vec<VertexBufferOthers> { &mut self.va }
}
impl TangentRecalculator for StaticMeshBuilder {}
impl VertexColorApplyScale for StaticMeshBuilder {}

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
        let asset = GltfImporter::import_from_gltf_primitive(&mut importer, &primitive, &buffers);

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
