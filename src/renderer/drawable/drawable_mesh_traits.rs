use std::ops::Range;

use crate::renderer::{drawable::vertex_types::VertexType, pipeline::pbr_material::PBRMaterial, resources::mesh_layout::ModelMatrixBindingSource, upload::buffer::upload_buffer};

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

    /// Record draw calls into the given render pass.
    /// 
    /// Set up per-object bind groups and immediates, bind vertex and index
    /// buffer, and record the draw call.
    /// 
    /// To actually get the draw call correct, you must bind correct pipelines
    /// to the render pass before drawing the mesh proper.
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

    /// Build the model-matrix bind group of an immediate mesh.
    fn build_immediate_model_matrix(&self, device: &wgpu::Device, queue: &wgpu::Queue, bindings: &dyn ModelMatrixBindingSource) -> wgpu::BindGroup {
        let immediate_buffer = upload_buffer(
            device,
            queue,
            "immediate mesh model matrix buffer",
            wgpu::BufferUsages::STORAGE,
            self.get_model_matrix().as_slice(),
        )
        .expect("failed to create the immediate mesh model matrix buffer");

        bindings.create_model_matrix_bind_group(
            device,
            "immediate mesh bind group",
            &immediate_buffer,
        )
    }
}
