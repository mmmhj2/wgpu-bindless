
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

impl From<&gltf::texture::Sampler<'_>> for SamplerDescription {

    fn from(value: &gltf::texture::Sampler) -> Self {
        fn match_address_mode(m: gltf::texture::WrappingMode) -> wgpu::AddressMode {
            match m {
                gltf::texture::WrappingMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
                gltf::texture::WrappingMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
                gltf::texture::WrappingMode::Repeat => wgpu::AddressMode::Repeat,
            }
        }

        fn match_mag_filter(f: Option<gltf::texture::MagFilter>) -> wgpu::FilterMode {
            match f {
                Some(gltf::texture::MagFilter::Nearest) | None => wgpu::FilterMode::Nearest,
                Some(gltf::texture::MagFilter::Linear) => wgpu::FilterMode::Linear,
            }
        }

        fn match_min_filter(f: Option<gltf::texture::MinFilter>) -> wgpu::FilterMode {
            match f {
                Some(gltf::texture::MinFilter::Nearest)
                | Some(gltf::texture::MinFilter::NearestMipmapNearest)
                | Some(gltf::texture::MinFilter::NearestMipmapLinear)
                | None => wgpu::FilterMode::Nearest,
                Some(gltf::texture::MinFilter::Linear)
                | Some(gltf::texture::MinFilter::LinearMipmapNearest)
                | Some(gltf::texture::MinFilter::LinearMipmapLinear) => wgpu::FilterMode::Linear,
            }
        }

        fn match_mipmap_filter(f: Option<gltf::texture::MinFilter>) -> wgpu::MipmapFilterMode {
            match f {
                Some(gltf::texture::MinFilter::Nearest)
                | Some(gltf::texture::MinFilter::Linear)
                | Some(gltf::texture::MinFilter::NearestMipmapNearest)
                | Some(gltf::texture::MinFilter::LinearMipmapNearest)
                | None => wgpu::MipmapFilterMode::Nearest,
                Some(gltf::texture::MinFilter::NearestMipmapLinear)
                | Some(gltf::texture::MinFilter::LinearMipmapLinear) => wgpu::MipmapFilterMode::Linear,
            }
        }

        Self {
            address_mode: [ match_address_mode(value.wrap_s()), match_address_mode(value.wrap_t()), wgpu::AddressMode::ClampToEdge ],
            mag_filter: match_mag_filter(value.mag_filter()),
            min_filter: match_min_filter(value.min_filter()),
            mipmap_filter: match_mipmap_filter(value.min_filter()),
            ..Default::default()
        }
    }
}
