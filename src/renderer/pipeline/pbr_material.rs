use crate::renderer::pipeline::{bindless_resource_manager::BindlessResourceManager, resource_allocator::LinearResourceAllocatorError, sampler::SamplerDescription};

/// Buffer entry for a PBR material.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct PBRMaterialBuffer {
    data : [u16; 6]
}

impl From<&PBRMaterial> for PBRMaterialBuffer {
    fn from(value: &PBRMaterial) -> Self {
        Self {
            data: [value.diffuse_tx as u16,
            value.diffuse_sp as u16,
            value.normal_tx as u16,
            value.normal_sp as u16,
            value.mrao_tx as u16,
            value.mrao_sp as u16]
        }
    }
}

impl PBRMaterialBuffer {
    pub fn get_slice(&self) -> &[u16] { &self.data }
}

pub struct PBRMaterial {
    diffuse_tx  : usize,
    normal_tx   : usize,
    mrao_tx     : usize,
    diffuse_sp  : usize,
    normal_sp   : usize,
    mrao_sp     : usize
}

impl PBRMaterial {
    pub fn new(
        d: &wgpu::Device,
        manager: &mut BindlessResourceManager,
        diffuse: Option<(wgpu::TextureView, Option<SamplerDescription>)>,
        normal: Option<(wgpu::TextureView, Option<SamplerDescription>)>,
        mrao: Option<(wgpu::TextureView, Option<SamplerDescription>)>
    ) -> Result<Self, LinearResourceAllocatorError> {
        let mut ret = Self {
            diffuse_tx: manager.get_white_texture(),
            normal_tx: manager.get_default_bump_texture(),
            mrao_tx: manager.get_white_texture(),
            diffuse_sp: manager.get_default_sampler(),
            normal_sp: manager.get_default_sampler(),
            mrao_sp: manager.get_default_sampler()
        };

        if let Some(df) = diffuse {
            ret.diffuse_tx = manager.push_texture(df.0)?;
            if let Some(s) = df.1 { ret.diffuse_sp = manager.push_sampler(d, s)? }
        }

        if let Some(n) = normal {
            ret.normal_tx = manager.push_texture(n.0)?;
            if let Some(s) = n.1 { ret.normal_sp = manager.push_sampler(d, s)? }
        }
        if let Some(m) = mrao {
            ret.mrao_tx = manager.push_texture(m.0)?;
            if let Some(s) = m.1 { ret.mrao_sp = manager.push_sampler(d, s)? }
        }

        Ok(ret)
    }
}
