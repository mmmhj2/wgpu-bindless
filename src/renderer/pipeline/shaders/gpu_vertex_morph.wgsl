
struct Position {
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

// TODO: 4 arrays for testing is more than enough, but 8 are required by the spec.
// TODO: maybe use binding_array
@group(2) @binding(0)
var<storage, read> morph_source_0: array<f32>;
@group(2) @binding(1)
var<storage, read> morph_source_1: array<f32>;
@group(2) @binding(2)
var<storage, read> morph_source_2: array<f32>;
@group(2) @binding(3)
var<storage, read> morph_source_3: array<f32>;


struct Immediate {
    // Packed texture and sampler index.
    anim_tex: u32,
    // frame id of the animation
    frame: f32,

    // Destination of each channels, packed into 32 bit integer.
    // Each 4-bits specifies one channel from low to high:
    // - 0: ignored (Guarantees no read from the morph source);
    // - 1: position;
    // - 2: normal;
    // - 3: tangent;
    // - higher: unused, reserved for texcoord and color.
    //
    // Note: zero weight from the morph target texture may not guarantee no read
    // due to possible non-uniform control flow restriction. 
    channel_destination: u32,
};
var<immediate> imm : Immediate;
