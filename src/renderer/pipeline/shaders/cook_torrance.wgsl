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

fn unpack_tx(idx: u32) -> vec2<u32> {
    return vec2<u32>(
        (idx & 0x0000FFFF),
        (idx & 0xFFFF0000) >> 16
    );
}

const PI: f32 = 3.14159265359;
const EPSILON: f32 = 0.0001;

struct PBRVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) bitangent: vec3<f32>,
    @location(4) uv0: vec2<f32>,
    @location(5) color: vec4<f32>
};

fn get_normal_matrix(model_matrix: mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        model_matrix[0].xyz,
        model_matrix[1].xyz,
        model_matrix[2].xyz
    );
}

@vertex
fn vs_main(model: VertexInput) -> PBRVertexOutput {
    var out: PBRVertexOutput;

    out.clip_position = camera.vp_matrix * immediates.model_matrix * vec4<f32>(model.position.xyz, 1.0);

    out.world_position = (immediates.model_matrix * vec4<f32>(model.position.xyz, 1.0)).xyz;

    var normal_matrix = get_normal_matrix(immediates.model_matrix);
    out.normal = normalize(normal_matrix * model.normal);

    out.tangent = normalize(normal_matrix * model.tangent.xyz);
    out.bitangent = cross(out.normal, out.tangent) * model.tangent.w;

    out.uv0 = model.uv0;
    out.color = model.color;

    return out;
}

fn D_GGX(ndotH: f32, roughness: f32) -> f32 {
    var alpha = roughness * roughness;
    var alpha_sq = alpha * alpha;
    var ndotH_sq = ndotH * ndotH;
    var denom = (ndotH_sq * (alpha_sq - 1.0) + 1.0);
    return alpha_sq / (PI * denom * denom);
}

fn G_Smith(ndotV: f32, ndotL: f32, roughness: f32) -> f32 {
    var r = roughness + 1.0;
    var k = (r * r) / 8.0;
    var ggx1 = ndotV / (ndotV * (1.0 - k) + k);
    var ggx2 = ndotL / (ndotL * (1.0 - k) + k);
    return ggx1 * ggx2;
}

fn F_Schlick(dotV: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(1.0 - dotV, 5.0);
}

@fragment
fn fs_main(in: PBRVertexOutput) -> @location(0) vec4<f32> {
    var diffuse_indices = unpack_tx(immediates.diffuse_tx_sp);
    var diffuse_texture = textureSample(
        bindless_textures[diffuse_indices[0]],
        bindless_samplers[diffuse_indices[1]],
        in.uv0
    );
    var baseColor = in.color.rgb * diffuse_texture.rgb;

    var normal_indices = unpack_tx(immediates.normal_tx_sp);
    var normal_texture = textureSample(
        bindless_textures[normal_indices[0]],
        bindless_samplers[normal_indices[1]],
        in.uv0
    );

    var mrao_indices = unpack_tx(immediates.mrao_tx_sp);
    var mrao_texture = textureSample(
        bindless_textures[mrao_indices[0]],
        bindless_samplers[mrao_indices[1]],
        in.uv0
    );

    var ao = mrao_texture.r;
    var metallic = mrao_texture.g;
    var roughness = mrao_texture.b;

    var TBN = mat3x3<f32>(in.tangent, in.bitangent, in.normal);
    var tangent_normal = normal_texture.rgb * 2.0 - 1.0;
    // Since WGSL does not have matrix inverse function,
    // we will have to do everything in the view space.
    var N = normalize((camera.view_matrix * vec4<f32>(TBN * tangent_normal, 0.0)).xyz);
    var V = -normalize((camera.view_matrix * vec4<f32>(in.world_position, 1.0)).xyz);
    var light_dir = vec3<f32>(-1.0, 1.0, 0.0);
    var L = normalize((camera.view_matrix * vec4<f32>(light_dir, 0.0)).xyz);
    var H = normalize(V + L);

    var ndotV = max(dot(N, V), 0.0);
    var ndotL = max(dot(N, L), 0.0);
    var ndotH = max(dot(N, H), 0.0);
    var dotV = max(dot(H, V), 0.0);

    var F0 = mix(vec3<f32>(0.04), baseColor, metallic);
    var D = D_GGX(ndotH, roughness);
    var G = G_Smith(ndotV, ndotL, roughness);
    var F = F_Schlick(dotV, F0);

    var specular_brdf = (D * G * F) / max(4.0 * ndotV * ndotL, EPSILON);

    var kd = mix(vec3<f32>(1.0) - F, vec3<f32>(0.0), metallic);
    var diffuse_brdf = baseColor * kd;

    var Lo = (diffuse_brdf + specular_brdf) * ndotL;
    return vec4<f32>(Lo, 1.0);
}
