
// NOTE: The following interface definitions are preliminary and subject to incompatible changes.

struct Position {
    // We cannot use `vec3` due to possible alignment problem.
    position: array<f32, 3>
}

struct Attribute {
    _color: array<f32, 4>,
    normal: array<f32, 3>,
    tangent: array<f32, 4>,
    _uv0: array<f32, 2>
}

@group(1) @binding(0)
var<storage, read> position_in: array<Position>;
@group(1) @binding(1)
var<storage, read> attribute_in: array<Attribute>;
@group(1) @binding(2)
var<storage, read_write> position_out: array<Position>;
@group(1) @binding(3)
var<storage, read_write> attribute_out: array<Attribute>;

struct Bones {
    indices: vec4<u32>,
    weights: vec4<f32>
};
@group(1) @binding(4)
var<storage, read> bones: array<Bones>;

struct Immediate {
    // Packed texture and sampler index.
    anim_tex: u32,
    // frame id of the animation
    frame: f32
};
var<immediate> imm : Immediate;

/// Maximum number of joint influences read per vertex.
const MAX_INFLUENCES: u32 = 4u;

/// Rebuild one joint matrix from the baked animation texture.
///
/// The texture is laid out as `width = joint_count * 4`, `height = frames`.
/// Each joint owns four consecutive texels holding the four *columns* of its
/// object-space skinning matrix, so no filtering is involved: this is a plain
/// texel fetch, not a sample.
fn load_joint_matrix(joint: u32) -> mat4x4<f32> {
    // The immediate packs the texture index in the low half.
    let texture = bindless_textures[imm.anim_tex & 0xFFFFu];
    let x = i32(joint * 4u);
    let y1 = i32(floor(imm.frame));
    // Wrapping around should be handled by animation asset importer
    let y2 = y1 + 1;
    let alpha = fract(imm.frame);
    // At 30 FPS the distortion from matrix lerp are most likely negligeable.
    return mat4x4<f32>(
        mix(textureLoad(texture, vec2<i32>(x + 0, y1), 0), textureLoad(texture, vec2<i32>(x + 0, y2), 0), alpha),
        mix(textureLoad(texture, vec2<i32>(x + 1, y1), 0), textureLoad(texture, vec2<i32>(x + 1, y2), 0), alpha),
        mix(textureLoad(texture, vec2<i32>(x + 2, y1), 0), textureLoad(texture, vec2<i32>(x + 2, y2), 0), alpha),
        mix(textureLoad(texture, vec2<i32>(x + 3, y1), 0), textureLoad(texture, vec2<i32>(x + 3, y2), 0), alpha)
    );
}

fn unpack_vec3(v: array<f32, 3>) -> vec3<f32> {
    return vec3<f32>(v[0], v[1], v[2]);
}

/// Drop the handedness component of a packed tangent.
fn unpack_tangent(v: array<f32, 4>) -> vec3<f32> {
    return vec3<f32>(v[0], v[1], v[2]);
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&position_out)) {
        return;
    }

    let bone = bones[index];
    let rest_attribute = attribute_in[index];
    let rest_position = unpack_vec3(position_in[index].position);

    var skinned_position = vec3<f32>(0.0);
    var skinned_normal = vec3<f32>(0.0);
    var skinned_tangent = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var i = 0u; i < MAX_INFLUENCES; i = i + 1u) {
        let weight = bone.weights[i];
        if (weight <= 0.0) {
            continue;
        }

        let matrix = load_joint_matrix(bone.indices[i]);
        // Normals and tangents only take the linear (rotation/scale) part; the
        // translation must not be applied to directions.
        let linear = mat3x3<f32>(matrix[0].xyz, matrix[1].xyz, matrix[2].xyz);

        skinned_position += weight * (matrix * vec4<f32>(rest_position, 1.0)).xyz;
        skinned_normal += weight * (linear * unpack_vec3(rest_attribute.normal));
        skinned_tangent += weight * (linear * unpack_tangent(rest_attribute.tangent));
        total_weight += weight;
    }

    // Guard against degenerate vectors
    if (total_weight <= 0.0) {
        skinned_position = rest_position;
        skinned_normal = unpack_vec3(rest_attribute.normal);
        skinned_tangent = unpack_tangent(rest_attribute.tangent);
    }
    let normal_length = length(skinned_normal);
    if (normal_length > 0.0) {
        skinned_normal = skinned_normal / normal_length;
    }
    let tangent_length = length(skinned_tangent);
    if (tangent_length > 0.0) {
        skinned_tangent = skinned_tangent / tangent_length;
    }

    position_out[index].position = array<f32, 3>(skinned_position.x, skinned_position.y, skinned_position.z);

    attribute_out[index]._color = rest_attribute._color;
    attribute_out[index].normal = array<f32, 3>(skinned_normal.x, skinned_normal.y, skinned_normal.z);
    // The w component carries the tangent handedness and must survive unchanged.
    attribute_out[index].tangent = array<f32, 4>(skinned_tangent.x, skinned_tangent.y, skinned_tangent.z, rest_attribute.tangent[3]);
    attribute_out[index]._uv0 = rest_attribute._uv0;
}
