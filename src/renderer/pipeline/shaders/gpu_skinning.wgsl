
// NOTE: The following interface definitions are preliminary and subject to incompatible changes.
@group(1) @binding(0)
var<storage, read> position_in: array<f32>;
@group(1) @binding(1)
var<storage, read_write> position_out: array<f32>;

struct Bones {
    indices: vec4<u32>,
    weights: vec4<f32>
};
@group(1) @binding(2)
var<storage, read> bones: array<Bones>;

struct Immediate {
    // Packed texture and sampler index.
    anim_tex: u32
};
var<immediate> imm : Immediate;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // TODO...
}
