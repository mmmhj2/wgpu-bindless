@group(0) @binding(0)
var hdr_input: texture_2d<f32>;
@group(0) @binding(1)
var hdr_input_sp: sampler;


struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(1) uv: vec2<f32>
};

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
    return out / (out + 1.0f);
}
