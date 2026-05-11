
/// Wrapper around wgpu::SamplerDescriptor.
/// 
/// This wrapper does not contain any floating point members, allowing
/// it to be stored within a hash map.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerDescription {
    pub address_mode    : [wgpu::AddressMode; 3],
    pub mag_filter      : wgpu::FilterMode,
    pub min_filter      : wgpu::FilterMode,
    pub mipmap_filter   : wgpu::MipmapFilterMode,
    pub compare         : Option<wgpu::CompareFunction>,
    pub anisotropy_clamp: u16,
    pub border_color    : Option<wgpu::SamplerBorderColor>  
}

impl Default for SamplerDescription {
    fn default() -> Self {
        Self {
            address_mode: Default::default(),
            mag_filter: Default::default(),
            min_filter: Default::default(),
            mipmap_filter: Default::default(),
            compare: None,
            anisotropy_clamp: 1,
            border_color: None
        }
    }
}

impl From<&SamplerDescription> for wgpu::SamplerDescriptor<'_> {
    fn from(value: &SamplerDescription) -> Self {
        Self { 
            label: Some("Hashed sampler"),
            address_mode_u: value.address_mode[0],
            address_mode_v: value.address_mode[1],
            address_mode_w: value.address_mode[2],
            mag_filter: value.mag_filter,
            min_filter: value.min_filter,
            mipmap_filter: value.mipmap_filter,
            lod_min_clamp: 0.0f32,
            lod_max_clamp: 32.0f32,
            compare: value.compare,
            anisotropy_clamp: value.anisotropy_clamp,
            border_color: value.border_color
        }
    }
}
