struct Camera {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    vp_matrix: mat4x4<f32>
};
@group(1) @binding(0)
var<uniform> camera: Camera;

struct GpuLight {
    // rgb = radiance (colour * intensity), a = kind (0 = directional, 1 = point, 2 = spot)
    color_kind      : vec4<f32>,
    // xyz = world position, a = range
    position_range  : vec4<f32>,
    // xyz = world direction, a = reserved
    direction       : vec4<f32>,
    // x = cos(inner cone), y = cos(outer cone), zw = reserved for shadow parameters
    cones           : vec4<f32>
};

struct Lighting {
    ambient     : vec3<f32>,
    light_count : u32,
    lights      : array<GpuLight, 64>
};
@group(2) @binding(0)
var<uniform> lighting: Lighting;

@group(3) @binding(0)
var<storage, read> model_matrices: array<mat4x4<f32>>;

struct Immediates {
    diffuse_tx_sp   : u32,
    normal_tx_sp    : u32,
    mrao_tx_sp      : u32
};
var<immediate> immediates: Immediates;

/// Unpack a combined index into texture index and sampler index, respectively.
fn unpack_tx(idx: u32) -> vec2<u32> {
    return vec2<u32>(
        (idx & 0x0000FFFF),
        (idx & 0xFFFF0000) >> 16
    );
}

fn get_model_matrix(instance: u32) -> mat4x4<f32> {
    return model_matrices[instance];
}

fn sample_diffuse(uv: vec2<f32>) -> vec4<f32> {
    var indices = unpack_tx(immediates.diffuse_tx_sp);
    return textureSample(
        bindless_textures[indices[0]],
        bindless_samplers[indices[1]],
        uv
    );
}

fn sample_normal(uv: vec2<f32>) -> vec4<f32> {
    var indices = unpack_tx(immediates.normal_tx_sp);
    return textureSample(
        bindless_textures[indices[0]],
        bindless_samplers[indices[1]],
        uv
    );
}

fn sample_mrao(uv: vec2<f32>) -> vec4<f32> {
    var indices = unpack_tx(immediates.mrao_tx_sp);
    return textureSample(
        bindless_textures[indices[0]],
        bindless_samplers[indices[1]],
        uv
    );
}
