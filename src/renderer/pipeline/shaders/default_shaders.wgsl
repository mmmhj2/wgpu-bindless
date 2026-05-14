
// Note: Use `binding_array` instead of `array``.
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

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) uv0: vec2<f32>
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv0: vec2<f32>
};

struct Immediates {
    model_matrix    : mat4x4<f32>,
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

@vertex
fn vs_main(
    model: VertexInput
) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = camera.vp_matrix * immediates.model_matrix * vec4<f32>(model.position.xyz, 1.0);

    out.color = model.color;
    out.uv0 = model.uv0;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    var diffuse_indices = unpack_tx(immediates.diffuse_tx_sp);
    var sampled_diffuse_color = textureSample(
        bindless_textures[diffuse_indices[0]],
        bindless_samplers[diffuse_indices[1]],
        in.uv0
    );

    return in.color * sampled_diffuse_color;
}
