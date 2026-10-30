
// NOTE: The following interface definitions are preliminary and subject to incompatible changes.
@group(0) @binding(0)
var<storage, read> position_in: array<vec4f>;
@group(0) @binding(1)
var<storage, read_write> position_out: array<vec4f>;

struct Immediate {
    anim_tex: u32
};
var<immediate> imm : Immediate;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // TODO...
}
