@group(0) @binding(0)
var hdr_input: texture_2d<f32>;
@group(0) @binding(1)
var luminance_tx: texture_2d<f32>;
@group(0) @binding(2)
var hdr_input_sp: sampler;

const A = 2.51f;
const B = 0.03f;
const C = 2.43f;
const D = 0.59f;
const E = 0.14f;

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
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var out = textureSample(hdr_input, hdr_input_sp, in.uv);
    return (out * (A * out + B)) / (out * (C * out + D) + E); 
}
