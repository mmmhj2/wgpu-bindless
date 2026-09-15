@group(0) @binding(0)
var bindless_textures : binding_array<texture_2d<f32>, 512>;
@group(0) @binding(1)
var bindless_samplers : binding_array<sampler, 128>;

struct Camera {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    vp_matrix: mat4x4<f32>
};
@group(1) @binding(0)
var<uniform> camera: Camera;

@group(2) @binding(0)
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
