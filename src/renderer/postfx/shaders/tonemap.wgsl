// HDR color produced by the scene pass. It is a pass-private framebuffer, so
// it stays in the pass-local bind group (group 1) rather than the bindless
// table. The sampler comes from the global bindless sampler table; its index
// is supplied through the immediate block.
@group(1) @binding(0)
var hdr_input: texture_2d<f32>;
// Average luminance of the scene, typically a 1x1 texture.
// Unused for ACES tonemapping.
@group(1) @binding(1)
var luminance_tx: texture_2d<f32>;

struct Immediates {
    sampler_index: u32
};
var<immediate> immediates: Immediates;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(1) uv: vec2<f32>
};

fn luminance(col : vec3f) -> f32 {
    return 0.2125 * col.r + 0.7154 * col.g + 0.0721 * col.b;
}

@vertex
fn vs_main(
    @builtin(vertex_index) gl_VertexIndex: u32
) -> VertexOutput {
    var out: VertexOutput;
    out.uv = vec2f(f32((gl_VertexIndex << 1) & 2), f32(gl_VertexIndex & 2));
    out.pos = vec4f(out.uv * 2.0f + -1.0f, 0.0f, 1.0f);
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

@fragment
fn fs_aces(in: VertexOutput) -> @location(0) vec4<f32> {
    const A = 2.51f;
    const B = 0.03f;
    const C = 2.43f;
    const D = 0.59f;
    const E = 0.14f;

    var c = textureSample(hdr_input, bindless_samplers[immediates.sampler_index], in.uv) * 0.6;
    var out = (c * (A * c + B)) / (c * (C * c + D) + E);
    return clamp(out, vec4<f32>(0.0), vec4<f32>(1.0)); 
}
