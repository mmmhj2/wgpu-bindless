use std::{num::NonZero, ops::Range};

use crate::renderer::{drawable::vertex_types::VertexType, pipeline::pbr_material::PBRMaterial, resources::mesh_layout::MeshLayout};

pub trait Mesh {
    /// Get a slice of references to all vertex attribute buffers.
    /// 
    /// The binding of these buffers into the render pass is specified
    /// by the `VertexType` of the current mesh.
    fn get_vertex_buffer(&self) -> &[wgpu::Buffer];

    /// Query how many vertices should be drawn for the mesh.
    /// 
    /// If the mesh is indexed, it should be equal to the index count.
    /// If it is not indexed, it should be equal to the vertex count.
    fn get_vertex_draw_count(&self) -> u32;

    /// Get the index buffer of the mesh.
    /// 
    /// For non-indexed meshes, None is returned.
    /// In which case the draw call should not be indexed either.
    fn get_index_buffer(&self) -> Option<&wgpu::Buffer>;

    /// Get the vertex type of the mesh.
    /// Currently unused.
    #[allow(unused)]
    fn get_vertex_type(&self) -> VertexType;
}

pub trait DrawableMesh : Mesh {
    /// Get the bind group that contains the storage buffer for model matrices.
    /// 
    /// Note that both cgmath and WGSL uses *column-major* matrices.
    /// While Rust has enforces row-major order, so long as you don't manipulate
    /// the matrix directly with Rust array, it will be fine.
    fn get_model_matrix_bind_group(&self) -> &wgpu::BindGroup;

    /// Range of instances passed to the draw call.
    ///
    /// The range start is the first instance index, and each draw's
    /// `@builtin(instance_index)` indexes the bound model-matrix storage
    /// buffer. A single-instance mesh returns `0..1`; a batch of N instances
    /// returns `0..N`.
    fn get_instance_range(&self) -> Range<u32>;

    /// Get the material description of the mesh.
    fn get_material(&self) -> &PBRMaterial;

    fn draw(&self, rp: &mut wgpu::RenderPass) -> () {
        rp.set_bind_group(3, Some(self.get_model_matrix_bind_group()), &[]);
        rp.set_immediates(0, &self.get_material().as_u8_arr());

        let vbs = self.get_vertex_buffer();
        for (i, b) in vbs.iter().enumerate() {
            rp.set_vertex_buffer(i as u32, b.slice(..));
        }

        if let Some(ib) = self.get_index_buffer() {
            rp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.get_vertex_draw_count(), 0, self.get_instance_range());
        } else {
            rp.draw(0..self.get_vertex_draw_count(), self.get_instance_range());
        }
    }
} 

pub trait ImmediateDrawableMesh {
    /// Get the model matrix of the mesh.
    fn get_model_matrix(&self) -> &[[f32; 4]; 4];

    fn build_immediate_model_matrix(&self, device: &wgpu::Device, queue: &wgpu::Queue, mmgr: &MeshLayout) -> wgpu::BindGroup {
        let immediate_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("immediate mesh model matrix buffer"),
            size: (std::mem::size_of::<f32>() * 4 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        queue.write_buffer(&immediate_buffer, 0, bytemuck::cast_slice(self.get_model_matrix()));

        let immediate_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("immediate mesh bind group"),
            layout: mmgr.get_bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding{
                        buffer: &immediate_buffer,
                        offset: 0,
                        size: NonZero::new(immediate_buffer.size())
                    })
                }
            ]
        });

        return immediate_bind_group;
    }
}
