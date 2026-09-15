use std::{num::NonZero, ops::Range};

use crate::renderer::{device_interface::DeviceInterface, mesh::{Mesh, mesh_manager::MeshManager}, pipeline::pbr_material::PBRMaterial};

pub trait DrawableMesh : Mesh {
    /// Get the bind group that contains the storage buffer for model matrices.
    /// 
    /// Note that both cgmath and WGSL uses *column-major* matrices.
    /// While Rust has enforces row-major order, so long as you don't manipulate
    /// the matrix directly with Rust array, it will be fine.
    fn get_model_matrix_bind_group(&self) -> &wgpu::BindGroup;

    fn get_instance_range(&self) -> Range<u32>;

    /// Get the material description of the mesh.
    fn get_material(&self) -> &PBRMaterial;

    fn draw(&self, rp: &mut wgpu::RenderPass) -> () {
        rp.set_bind_group(2, Some(self.get_model_matrix_bind_group()), &[]);
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

pub(crate) struct ImmediateContext<'a> {
    pub device: &'a DeviceInterface,
    pub mesh_manager: &'a MeshManager
}

pub trait ImmediateDrawableMesh {
    /// Get the model matrix of the mesh.
    fn get_model_matrix(&self) -> &[[f32; 4]; 4];

    fn build_immediate_model_matrix(&self, di: &DeviceInterface, mmgr: &MeshManager) -> wgpu::BindGroup {
        let device = di.get_device();

        let immediate_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("immediate mesh model matrix buffer"),
            size: (std::mem::size_of::<f32>() * 4 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        di.get_queue().write_buffer(&immediate_buffer, 0, bytemuck::cast_slice(self.get_model_matrix()));

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
