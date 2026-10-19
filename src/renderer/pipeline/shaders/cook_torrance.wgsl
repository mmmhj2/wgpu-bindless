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

const PI: f32 = 3.14159265359;
const EPSILON: f32 = 0.0001;

const LIGHT_DIRECTIONAL: f32 = 0.0;
const LIGHT_SPOT: f32 = 2.0;

/// Smoothly fade a point/spot light out towards its range boundary.
/// A non-positive range disables the window entirely.
fn range_window(distance: f32, range: f32) -> f32 {
    if (range <= 0.0) {
        return 1.0;
    }
    let ratio = clamp(distance / range, 0.0, 1.0);
    let ratio_sq = ratio * ratio;
    return max(1.0 - ratio_sq * ratio_sq, 0.0);
}

/// Cone falloff for spot lights, using precomputed cosines of the cone angles.
fn spot_attenuation(cos_theta: f32, cos_inner: f32, cos_outer: f32) -> f32 {
    return clamp((cos_theta - cos_outer) / max(cos_inner - cos_outer, EPSILON), 0.0, 1.0);
}

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
fn vs_main(
    model: VertexInput,
    @builtin(instance_index) instance_index: u32
) -> PBRVertexOutput {
    var out: PBRVertexOutput;
    var model_mat = get_model_matrix(instance_index);
    out.clip_position = camera.vp_matrix * model_mat * vec4<f32>(model.position.xyz, 1.0);

    out.world_position = (model_mat * vec4<f32>(model.position.xyz, 1.0)).xyz;

    var normal_matrix = get_normal_matrix(model_mat);
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
    var diffuse_texture = sample_diffuse(in.uv0);
    var baseColor = in.color.rgb * diffuse_texture.rgb;

    var normal_texture = sample_normal(in.uv0);
    var mrao_texture = sample_mrao(in.uv0);

    var ao = mrao_texture.r;
    var metallic = mrao_texture.g;
    var roughness = mrao_texture.b;

    var TBN = mat3x3<f32>(in.tangent, in.bitangent, in.normal);
    var tangent_normal = normal_texture.rgb * 2.0 - 1.0;
    // Since WGSL does not have matrix inverse function,
    // we will have to do everything in the view space.
    // Or maybe we should precompute the inverse and pass it to GPU?
    var P = (camera.view_matrix * vec4<f32>(in.world_position, 1.0)).xyz;
    var N = normalize((camera.view_matrix * vec4<f32>(TBN * tangent_normal, 0.0)).xyz);
    var V = normalize(-P);

    var ndotV = max(dot(N, V), 0.0);
    var F0 = mix(vec3<f32>(0.04), baseColor, metallic);

    var Lo = vec3<f32>(0.0);
    for (var i = 0u; i < lighting.light_count; i = i + 1u) {
        let light = lighting.lights[i];
        let kind = light.color_kind.a;
        let radiance = light.color_kind.rgb;

        var L: vec3<f32>;
        var attenuation = 1.0;

        if (kind == LIGHT_DIRECTIONAL) {
            // Directional: constant direction, no distance falloff.
            L = normalize((camera.view_matrix * vec4<f32>(light.direction.xyz, 0.0)).xyz);
        } else {
            let light_position = (camera.view_matrix * vec4<f32>(light.position_range.xyz, 1.0)).xyz;
            let to_light = light_position - P;
            let distance = length(to_light);
            L = to_light / max(distance, EPSILON);
            attenuation = range_window(distance, light.position_range.a) / max(distance * distance, EPSILON);

            if (kind == LIGHT_SPOT) {
                // Spot: additionally modulate by the cone around the light direction.
                let light_dir = normalize((camera.view_matrix * vec4<f32>(light.direction.xyz, 0.0)).xyz);
                let cos_theta = dot(-L, light_dir);
                attenuation = attenuation * spot_attenuation(cos_theta, light.cones.x, light.cones.y);
            }
        }

        let ndotL = max(dot(N, L), 0.0);
        if (ndotL <= 0.0 || attenuation <= 0.0) {
            continue;
        }

        let H = normalize(V + L);
        let ndotH = max(dot(N, H), 0.0);
        let dotV = max(dot(H, V), 0.0);

        let D = D_GGX(ndotH, roughness);
        let G = G_Smith(ndotV, ndotL, roughness);
        let F = F_Schlick(dotV, F0);

        let specular_brdf = (D * G * F) / max(4.0 * ndotV * ndotL, EPSILON);
        let kd = (vec3<f32>(1.0) - F) * (1.0 - metallic);
        let diffuse_brdf = baseColor * kd;

        Lo = Lo + (diffuse_brdf + specular_brdf) * ndotL * radiance * attenuation;
    }

    Lo = Lo + baseColor * lighting.ambient * ao;
    return vec4<f32>(Lo, 1.0);
}
