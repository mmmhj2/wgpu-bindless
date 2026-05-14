use crate::renderer::pipeline::{bindless_resource_manager::BindlessResourceManager, resource_allocator::LinearResourceAllocatorError, sampler::SamplerDescription};

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
        diffuse_tx: usize,
        normal_tx: usize,
        mrao_tx: usize,
        diffuse_sp: usize,
        normal_sp: usize,
        mrao_sp: usize
    ) -> Self {
        Self { diffuse_tx, normal_tx, mrao_tx, diffuse_sp, normal_sp, mrao_sp }
    }

    pub fn create_from_views(
        d: &wgpu::Device,
        manager: &mut BindlessResourceManager,
        diffuse: Option<(wgpu::TextureView, Option<SamplerDescription>)>,
        normal: Option<(wgpu::TextureView, Option<SamplerDescription>)>,
        mrao: Option<(wgpu::TextureView, Option<SamplerDescription>)>
    ) -> Result<Self, LinearResourceAllocatorError> {
        let mut ret = Self {
            diffuse_tx: manager.get_white_texture(),
            normal_tx: manager.get_default_bump_texture(),
            mrao_tx: manager.get_default_mrao_texture(),
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

    /// Convert this material into a u8 slice for uploading to GPU.
    /// 
    /// The buffer is composed of 6 16-bit integers:
    /// 1. Texture ID of Diffuse Map
    /// 1. Sampler ID of Diffuse Map
    /// 1. Texture ID of Normal Map
    /// 1. Sampler ID of Normal Map
    /// 1. Texture ID of MRAO Map
    /// 1. Sampler ID of MRAO Map
    /// 
    /// WGSL does not support 16-bit integers, so they are packed into 3 32-bit integers on the shader side.
    /// As WGSL enforces little-endianness, lower 16 bits of each integers are therefore the index of texture.
    pub fn as_u8_arr(&self) -> [u8; 12] {
        let bytes = [self.diffuse_tx as u16,
            self.diffuse_sp as u16,
            self.normal_tx as u16,
            self.normal_sp as u16,
            self.mrao_tx as u16,
            self.mrao_sp as u16];
        bytemuck::cast(bytes)
    }
}
