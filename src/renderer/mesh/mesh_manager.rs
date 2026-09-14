use std::num::NonZero;

use crate::renderer::device_interface::DeviceInterface;

pub struct MeshManager {
    bind_group_layout: wgpu::BindGroupLayout
}

impl MeshManager {
    const BLG_MODEL_MATRICES : [wgpu::BindGroupLayoutEntry; 1] = [
        wgpu::BindGroupLayoutEntry{
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                // At least one model matrix is needed.
                min_binding_size: Some(NonZero::new(std::mem::size_of::<[[f32; 4]; 4]>() as u64).unwrap())
            },
            count: None,
        }
    ];

    const BGLD_MODEL_MATRICES: wgpu::BindGroupLayoutDescriptor<'static> = wgpu::BindGroupLayoutDescriptor{
        label: Some("Model matrices descriptor set layout"),
        entries: &Self::BLG_MODEL_MATRICES,
    };

    pub fn new(di: &DeviceInterface) -> Self {
        Self {
            bind_group_layout: di.get_device().create_bind_group_layout(&Self::BGLD_MODEL_MATRICES)
        }
    }

    pub fn get_bind_group_layout(&self) -> &wgpu::BindGroupLayout { &self.bind_group_layout }
}
