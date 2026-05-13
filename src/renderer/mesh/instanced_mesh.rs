use std::{collections::VecDeque, sync::Arc};

use cgmath::SquareMatrix;

use crate::renderer::{device_interface::DeviceInterface, mesh::{DrawableMesh, Mesh, vertex_reconditioner::{TangentRecalculator, VertexReconditionable, VertexReconditionableAttributeWrite}, vertex_types::{VertexBufferOthers, VertexBufferPosition}}, pipeline::{bindless_resource_manager::BindlessResourceManager, pbr_material::PBRMaterial, sampler::SamplerDescription, texture::{Texture, TextureType}}};

/// Actual instanced mesh, whose data have already been pushed onto GPU.
struct InstancedMesh {
    vertex_attribute_buffers    : [wgpu::Buffer; 2],
    index_buffer                : Option<wgpu::Buffer>,
    vertex_draw_count           : u32,
    material                    : PBRMaterial
}
impl InstancedMesh {
    fn get_material(&self) -> &PBRMaterial { &self.material }
}
impl Mesh for InstancedMesh {
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer] { &self.vertex_attribute_buffers }
    fn get_vertex_draw_count(&self) -> u32 { self.vertex_draw_count }
    fn get_index_buffer(&self) -> Option<&wgpu::Buffer> { self.index_buffer.as_ref() }
    fn get_vertex_type(&self) -> super::vertex_types::VertexType { super::vertex_types::VertexType::Basic }
}

/// Builder for instanced mesh.
/// It contains all data of an instanced mesh, but is stored on CPU for further reconditioning.
/// It must be committed to GPU before using. 
pub struct InstancedMeshBuilder {
    vp  : Vec<VertexBufferPosition>,
    va  : Vec<VertexBufferOthers>,
    vi  : Option<Vec<u32>>,
    material : PBRMaterial,
    vertex_draw_count : u32,
    need_tangent : bool
}

impl InstancedMeshBuilder {
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

    fn new(
        di: &DeviceInterface,
        bindless_manager: &mut BindlessResourceManager,
        primitive: &gltf::Primitive,
        buffers: &Vec<gltf::buffer::Data>,
        images: &Vec<gltf::image::Data>
    ) -> Self {
        let vp = Self::construct_position_buffer(
            &primitive,
            buffers
        );
        let (va, need_tangent) = Self::construct_attribute_buffer(&primitive, buffers, vp.len());
        let vi = Self::construct_index_buffer(&primitive, buffers);
        let vertex_draw_count = match vi.as_ref() {
            Some(b) => b.len(),
            None => vp.len()
        } as u32;

        // Construct textures for the primitive.
        let pbr_material = primitive.material().pbr_metallic_roughness();

        let diffuse_id = if let Some(base_color_texture) = pbr_material.base_color_texture() {
            let tx: wgpu::Texture = Texture::create_from_gltf(
                di,
                TextureType::ColorSrgb,
                &base_color_texture.texture(),
                images,
                None
            ).expect("Failed to import diffuse texture").into();
            
            let txv = tx.create_view(&Default::default());
            let tx = bindless_manager.push_texture(txv).expect("Failed to push diffuse texture.");

            let spd = SamplerDescription::from(&base_color_texture.texture().sampler());
            let sp = bindless_manager.push_sampler(di.get_device(), spd).expect("");

            (tx, sp)
        } else {
            (bindless_manager.get_white_texture(), bindless_manager.get_default_sampler())
        };

        let normal_id = if let Some(normal_texture) = primitive.material().normal_texture() {
            let tx: wgpu::Texture = Texture::create_from_gltf(
                di,
                TextureType::Normal,
                &normal_texture.texture(),
                images,
                None
            ).expect("Failed to import normal texture").into();
            
            let txv = tx.create_view(&Default::default());
            let tx = bindless_manager.push_texture(txv).expect("Failed to push normal texture.");

            let spd = SamplerDescription::from(&normal_texture.texture().sampler());
            let sp = bindless_manager.push_sampler(di.get_device(), spd).expect("");

            (tx, sp)
        } else {
            (bindless_manager.get_default_bump_texture(), bindless_manager.get_default_sampler())
        };

        let material = PBRMaterial::new(
            diffuse_id.0,
            normal_id.0,
            bindless_manager.get_default_sampler(),
            diffuse_id.1,
            normal_id.1,
            bindless_manager.get_default_sampler()
        );

        Self { vp, va, vi, vertex_draw_count, need_tangent, material }
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
    fn recondition_and_commit(mut self, di: &DeviceInterface) -> InstancedMesh {
        if self.need_tangent  { self.recalculate_tangents(); }
        let (vp_va, vi) = self.push_buffers(di);
        InstancedMesh { vertex_attribute_buffers: vp_va, index_buffer: vi, vertex_draw_count: self.vertex_draw_count, material: self.material }
    }
}
impl VertexReconditionable for InstancedMeshBuilder {
    fn get_position_buffer(&self) -> &Vec<VertexBufferPosition> { &self.vp }
    fn get_attribute_buffer(&self) -> &Vec<VertexBufferOthers> { &self.va }
    fn get_index_buffer(&self) -> Option<&Vec<u32>> { self.vi.as_ref() }
}
impl VertexReconditionableAttributeWrite for InstancedMeshBuilder {
    fn get_attribute_buffer_mut(&mut self) -> &mut Vec<VertexBufferOthers> { &mut self.va }
}
impl TangentRecalculator for InstancedMeshBuilder {}

/// Instances of instanced mesh.
/// Holds unique data for each instance such as model matrix, and a reference to the underlying mesh.
#[derive(Clone)]
pub struct InstancedMeshInstance {
    mesh            : Arc<InstancedMesh>,
    model_matrix    : cgmath::Matrix4<f32>
}

impl InstancedMeshInstance {
    /// Create instances from a GLTF scene.
    /// 
    /// Only nodes that contains meshes are processed.
    /// Transforms of the nodes will be preserved in the model matrices of the
    /// instances produced.
    pub fn create_from_gltf_scene(
        di: &DeviceInterface,
        bindless_manager: &mut BindlessResourceManager,
        scene: &gltf::Scene,
        buffers: &Vec<gltf::buffer::Data>,
        images: &Vec<gltf::image::Data>
    ) -> Vec<Self> {
        let mut ret = Vec::new();

        // Do a BFS to collect all meshes.
        // DFS, recursion and trees in Rust are simply PITA.
        let mut transform_queue = VecDeque::<cgmath::Matrix4<f32>>::new();
        let mut node_queue = VecDeque::<usize>::new();

        for root_node in scene.nodes() {
            transform_queue.push_back(root_node.transform().matrix().into());
            node_queue.push_back(root_node.index());

            while !node_queue.is_empty() {
                let current_node = scene.nodes().nth(node_queue.pop_front().unwrap()).expect("valid node index");
                let current_transform = transform_queue.pop_front().unwrap();
                if let Some(m) = current_node.mesh() {
                    let mut instances = Self::create_from_gltf_mesh(di, bindless_manager, &m, buffers, images);
                    for inst in &mut instances {
                        inst.model_matrix = current_transform;
                    }
                    ret.extend(instances);
                }

                for ch in current_node.children() {
                    transform_queue.push_back(current_transform * cgmath::Matrix4::from(ch.transform().matrix()));
                    node_queue.push_back(ch.index());
                }
            }
        }

        return ret;
    }

    /// Create instances from a GLTF mesh.
    /// 
    /// All primitives will be processed, each one corresponding to an new instance.
    pub fn create_from_gltf_mesh(
        di: &DeviceInterface,
        bindless_manager: &mut BindlessResourceManager,
        mesh: &gltf::Mesh,
        buffers: &Vec<gltf::buffer::Data>,
        images: &Vec<gltf::image::Data>
    ) -> Vec<Self> {
        let mut ret = Vec::new();

        for primitive in mesh.primitives() {
            ret.push(
                Self{
                    mesh: InstancedMeshBuilder::new(di, bindless_manager, &primitive, buffers, images).recondition_and_commit(di).into(),
                    model_matrix: cgmath::Matrix4::identity().into()
                }
            )
        }

        return ret;
    }
}

impl Mesh for InstancedMeshInstance {
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer] { self.mesh.get_vertex_buffer() }
    fn get_vertex_draw_count(&self) -> u32 { self.mesh.get_vertex_draw_count() }
    fn get_index_buffer(&self) -> Option<&wgpu::Buffer> { self.mesh.get_index_buffer() }
    fn get_vertex_type(&self) -> super::vertex_types::VertexType { self.mesh.get_vertex_type() }
}

impl DrawableMesh for InstancedMeshInstance {
    fn get_model_matrix(&self) -> &[[f32; 4]; 4] { self.model_matrix.as_ref() }
    fn get_material(&self) -> &PBRMaterial { self.mesh.get_material() }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_buffer_import() {
        let (document, buffers, _) = gltf::import("resource/test_two_cubes.glb").expect("Failed to import GLB file.");

        let mesh = document.meshes()
            .find(|x| x.name().unwrap_or_default() == "cube_textured")
            .expect("Cannot find cube_textured.");

        let primitive = mesh.primitives().next().expect("cube_textured has not primitives.");
        assert_eq!(primitive.mode(), gltf::mesh::Mode::Triangles);

        let vp = InstancedMeshBuilder::construct_position_buffer(&primitive, &buffers);
        let va = InstancedMeshBuilder::construct_attribute_buffer(&primitive, &buffers, vp.len());
        let vi = InstancedMeshBuilder::construct_index_buffer(&primitive, &buffers).expect("Cannot find index buffer.");

        // Four vertices for each face.
        assert_eq!(vp.len(), 4 * 6);
        assert_eq!(vp.len(), va.0.len());
        // Every face has two triangles and therefore six vertices.
        assert_eq!(vi.len(), 2 * 6 * 3);
        println!("Vertex positions: {:?}", &vp);
        println!("Vertex attributes: {:?}", &va);
        println!("Indices: {:?}", &vi);
    }
}
