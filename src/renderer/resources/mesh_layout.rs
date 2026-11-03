use std::num::NonZero;

/// A small helper struct handling per-mesh binding groups (i.e. model matrices)
pub struct MeshLayout {
    bind_group_layout: wgpu::BindGroupLayout
}

impl MeshLayout {
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

    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            bind_group_layout: device.create_bind_group_layout(&Self::BGLD_MODEL_MATRICES)
        }
    }

    pub fn get_bind_group_layout(&self) -> &wgpu::BindGroupLayout { &self.bind_group_layout }
}

/// Source of the per-mesh (group 3) bind groups.
pub trait ModelMatrixBindingSource {
    /// Create a group-3 bind group covering the whole of `buffer`.
    fn create_model_matrix_bind_group(
        &self,
        device: &wgpu::Device,
        label: &str,
        buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup;
}

impl ModelMatrixBindingSource for MeshLayout {
    fn create_model_matrix_bind_group(
        &self,
        device: &wgpu::Device,
        label: &str,
        buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.bind_group_layout,
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
}
