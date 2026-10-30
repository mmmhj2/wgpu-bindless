
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
    frame: u32
};
var<immediate> imm : Immediate;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // TODO...
}
