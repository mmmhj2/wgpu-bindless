// Vertex morphing compute shader.
//
// One invocation deforms one vertex. The shader reads the rest pose, walks the
// morph target channels described by the data-driven layout buffer, and writes
// the deformed streams. Missing attributes carry a zero mask bit and are never
// read.

const MAX_TARGETS: u32 = 8u;
const CHANNEL_COUNT: u32 = 5u;

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

struct AttributeOffset {
    offsets: array<u32, 5>
}
struct MorphLayout {
    channel_count: u32,
    attribute_masks: array<u32, 8>,
    attribute_offsets: array<AttributeOffset, 8>
}

// Uniform buffer can cause some alignment problems.
@group(2) @binding(0)
var<storage, read> morph_layout: MorphLayout;
@group(2) @binding(1)
var<storage, read> deltas: array<f32>;

struct Immediate {
    // Bindless index of the baked weight texture or the base-weight texture.
    tex: u32,
    // Fractional row the weights are sampled at.
    frame: f32,
    // Number of weights to consume. Zero means "copy the rest pose".
    channels: u32,
};
var<immediate> imm: Immediate;

fn unpack_vec3(v: array<f32, 3>) -> vec3<f32> {
    return vec3<f32>(v[0], v[1], v[2]);
}

fn unpack_tangent(v: array<f32, 4>) -> vec3<f32> {
    return vec3<f32>(v[0], v[1], v[2]);
}

fn unpack_vec4(v: array<f32, 4>) -> vec4<f32> {
    return vec4<f32>(v[0], v[1], v[2], v[3]);
}

fn unpack_vec2(v: array<f32, 2>) -> vec2<f32> {
    return vec2<f32>(v[0], v[1]);
}

fn load_delta_vec3(offset: u32) -> vec3<f32> {
    return vec3<f32>(deltas[offset], deltas[offset + 1u], deltas[offset + 2u]);
}

fn load_delta_vec2(offset: u32) -> vec2<f32> {
    return vec2<f32>(deltas[offset], deltas[offset + 1u]);
}

fn load_delta_vec4(offset: u32) -> vec4<f32> {
    return vec4<f32>(deltas[offset], deltas[offset + 1u], deltas[offset + 2u], deltas[offset + 3u]);
}

/// Weights of up to four targets share one texel, so a row of
/// `ceil(MAX_TARGETS / 4)` texels covers every target of one frame.
const WEIGHTS_PER_TEXEL: u32 = 4u;
const WEIGHT_ROW_TEXELS: u32 = (MAX_TARGETS + WEIGHTS_PER_TEXEL - 1u) / WEIGHTS_PER_TEXEL;

/// Read the whole weight row at `frame`, interpolating between the two adjacent
/// baked frames. Out-of-range texels read as zero.
fn load_weight_row(frame: f32) -> array<vec4<f32>, WEIGHT_ROW_TEXELS> {
    let weight_tex = bindless_textures[imm.tex & 0xFFFFu];
    let y1 = i32(floor(frame));
    let alpha = fract(frame);

    var row: array<vec4<f32>, WEIGHT_ROW_TEXELS>;
    for (var x = 0u; x < WEIGHT_ROW_TEXELS; x = x + 1u) {
        let here = textureLoad(weight_tex, vec2<i32>(i32(x), y1), 0);
        if (alpha > 0.0) {
            let next = textureLoad(weight_tex, vec2<i32>(i32(x), y1 + 1), 0);
            row[x] = mix(here, next, alpha);
        } else {
            row[x] = here;
        }
    }
    return row;
}

/// Weight of one target, read out of a pre-loaded row.
fn weight_of(row: array<vec4<f32>, WEIGHT_ROW_TEXELS>, channel: u32) -> f32 {
    let texel = row[channel / WEIGHTS_PER_TEXEL];
    let lane = channel % WEIGHTS_PER_TEXEL;
    if (lane == 0u) {
        return texel.x;
    }
    if (lane == 1u) {
        return texel.y;
    }
    if (lane == 2u) {
        return texel.z;
    }
    return texel.w;
}

fn copy_rest_pose(index: u32, rest_attribute: Attribute) {
    position_out[index].position = position_in[index].position;
    attribute_out[index]._color = rest_attribute._color;
    attribute_out[index].normal = rest_attribute.normal;
    attribute_out[index].tangent = rest_attribute.tangent;
    attribute_out[index]._uv0 = rest_attribute._uv0;
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&position_out)) {
        return;
    }

    let rest_attribute = attribute_in[index];
    // The layout has `MAX_TARGETS` mask and offset slots; never walk past it
    // even if a malformed dispatch supplies a larger count.
    let channel_count = min(imm.channels, MAX_TARGETS);
    if (channel_count == 0u) {
        copy_rest_pose(index, rest_attribute);
        return;
    }

    var position = unpack_vec3(position_in[index].position);
    var normal = unpack_vec3(rest_attribute.normal);
    var tangent = unpack_tangent(rest_attribute.tangent);
    var color = unpack_vec4(rest_attribute._color);
    var uv0 = unpack_vec2(rest_attribute._uv0);

    let weights = load_weight_row(imm.frame);

    for (var channel = 0u; channel < channel_count; channel = channel + 1u) {
        let weight = weight_of(weights, channel);
        if (weight == 0.0) {
            continue;
        }

        let mask = morph_layout.attribute_masks[channel];
        if ((mask & 1u) != 0u) {
            position += weight * load_delta_vec3(morph_layout.attribute_offsets[channel].offsets[0u] + index * 3u);
        }
        if ((mask & 2u) != 0u) {
            normal += weight * load_delta_vec3(morph_layout.attribute_offsets[channel].offsets[1u] + index * 3u);
        }
        if ((mask & 4u) != 0u) {
            tangent += weight * load_delta_vec3(morph_layout.attribute_offsets[channel].offsets[2u] + index * 3u);
        }
        if ((mask & 8u) != 0u) {
            uv0 += weight * load_delta_vec2(morph_layout.attribute_offsets[channel].offsets[3u] + index * 2u);
        }
        if ((mask & 16u) != 0u) {
            color += weight * load_delta_vec4(morph_layout.attribute_offsets[channel].offsets[4u] + index * 4u);
        }
    }

    let normal_length = length(normal);
    if (normal_length > 0.0) {
        normal = normal / normal_length;
    }
    let tangent_length = length(tangent);
    if (tangent_length > 0.0) {
        tangent = tangent / tangent_length;
    }

    position_out[index].position = array<f32, 3>(position.x, position.y, position.z);
    attribute_out[index]._color = array<f32, 4>(color.x, color.y, color.z, color.w);
    attribute_out[index].normal = array<f32, 3>(normal.x, normal.y, normal.z);
    // The w component carries the tangent handedness and must survive unchanged.
    attribute_out[index].tangent = array<f32, 4>(
        tangent.x,
        tangent.y,
        tangent.z,
        rest_attribute.tangent[3],
    );
    attribute_out[index]._uv0 = array<f32, 2>(uv0.x, uv0.y);
}
