// Vertex morphing compute shader.
//
// One invocation deforms one vertex. The shader reads the rest pose, walks the
// morph target channels described by the data-driven layout buffer, and writes
// the deformed streams. Missing attributes carry a zero mask bit and are never
// read.

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

// The morphing pipeline shares the skinning group-1 layout, so binding 4 is
// declared to match it. Vertex morphing never reads joint data.
struct Bones {
    indices: vec4<u32>,
    weights: vec4<f32>
};
@group(1) @binding(4)
var<storage, read> bones: array<Bones>;

// `[0]` channel count, `[1..9]` per-channel attribute masks,
// `[9..49]` per-channel, per-attribute offsets into `deltas`.
@group(2) @binding(0)
var<storage, read> morph_layout: array<u32, 49>;
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

const MAX_TARGETS: u32 = 8u;
const CHANNEL_COUNT: u32 = 5u;
const MASK_WORD_OFFSET: u32 = 1u;
const OFFSET_WORD_OFFSET: u32 = 1u + MAX_TARGETS;

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

/// Read one channel's weight at `frame` from the bindless weight texture.
fn load_weight(frame: f32, channel: u32) -> f32 {
    let weight_tex = bindless_textures[imm.tex & 0xFFFFu];
    let x = i32(channel / 4u);
    let y1 = i32(floor(frame));
    let alpha = fract(frame);

    var texel = textureLoad(weight_tex, vec2<i32>(x, y1), 0);
    if (alpha > 0.0) {
        texel = mix(texel, textureLoad(weight_tex, vec2<i32>(x, y1 + 1), 0), alpha);
    }

    let lane = channel % 4u;
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

    for (var channel = 0u; channel < channel_count; channel = channel + 1u) {
        let weight = load_weight(imm.frame, channel);
        if (weight == 0.0) {
            continue;
        }

        let mask = morph_layout[MASK_WORD_OFFSET + channel];
        let offset_base = OFFSET_WORD_OFFSET + channel * CHANNEL_COUNT;

        if ((mask & 1u) != 0u) {
            position += weight * load_delta_vec3(morph_layout[offset_base + 0u]);
        }
        if ((mask & 2u) != 0u) {
            normal += weight * load_delta_vec3(morph_layout[offset_base + 1u]);
        }
        if ((mask & 4u) != 0u) {
            tangent += weight * load_delta_vec3(morph_layout[offset_base + 2u]);
        }
        if ((mask & 8u) != 0u) {
            uv0 += weight * load_delta_vec2(morph_layout[offset_base + 3u]);
        }
        if ((mask & 16u) != 0u) {
            color += weight * load_delta_vec4(morph_layout[offset_base + 4u]);
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
