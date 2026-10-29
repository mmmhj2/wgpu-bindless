
// Bloom shader adapted from https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom
// which in turn is based on the ACM Siggraph 2014 presentation of Call of Duty.

// The source texture is a pass-private framebuffer (it can be a render target),
// so it stays in the pass-local bind group (group 1) instead of the global
// bindless table. The sampler, however, comes from the bindless sampler table;
// its index is supplied through the immediate block.
@group(1) @binding(0)
var source_tx: texture_2d<f32>;

struct Immediates {
    blur_radius: f32,
    sampler_index: u32
};
var<immediate> immediates: Immediates;

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(1) uv: vec2f
};

fn sample_source(uv: vec2f) -> vec4f {
    return textureSample(source_tx, bindless_samplers[immediates.sampler_index], uv);
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
fn fs_downsample(
    in: VertexOutput
) -> @location(0) vec4f {
    var source_texels = vec2f(textureDimensions(source_tx, 0));
    var x = 1.0 / source_texels.x;
    var y = 1.0 / source_texels.y;

    var a = sample_source(vec2(in.uv.x - 2*x, in.uv.y + 2*y)).rgb;
    var b = sample_source(vec2(in.uv.x,       in.uv.y + 2*y)).rgb;
    var c = sample_source(vec2(in.uv.x + 2*x, in.uv.y + 2*y)).rgb;

    var d = sample_source(vec2(in.uv.x - 2*x, in.uv.y)).rgb;
    var e = sample_source(vec2(in.uv.x,       in.uv.y)).rgb;
    var f = sample_source(vec2(in.uv.x + 2*x, in.uv.y)).rgb;

    var g = sample_source(vec2(in.uv.x - 2*x, in.uv.y - 2*y)).rgb;
    var h = sample_source(vec2(in.uv.x,       in.uv.y - 2*y)).rgb;
    var i = sample_source(vec2(in.uv.x + 2*x, in.uv.y - 2*y)).rgb;

    var j = sample_source(vec2(in.uv.x - x, in.uv.y + y)).rgb;
    var k = sample_source(vec2(in.uv.x + x, in.uv.y + y)).rgb;
    var l = sample_source(vec2(in.uv.x - x, in.uv.y - y)).rgb;
    var m = sample_source(vec2(in.uv.x + x, in.uv.y - y)).rgb;

    var downsample = e*0.125;
    downsample += (a+c+g+i)*0.03125;
    downsample += (b+d+f+h)*0.0625;
    downsample += (j+k+l+m)*0.125;
    return vec4f(downsample, 0.0);
}

@fragment
fn fs_upsample(
    in: VertexOutput
) -> @location(0) vec4f {
    var x = immediates.blur_radius;
    var y = immediates.blur_radius;

    var a = sample_source(vec2(in.uv.x - x, in.uv.y + y)).rgb;
    var b = sample_source(vec2(in.uv.x,     in.uv.y + y)).rgb;
    var c = sample_source(vec2(in.uv.x + x, in.uv.y + y)).rgb;

    var d = sample_source(vec2(in.uv.x - x, in.uv.y)).rgb;
    var e = sample_source(vec2(in.uv.x,     in.uv.y)).rgb;
    var f = sample_source(vec2(in.uv.x + x, in.uv.y)).rgb;

    var g = sample_source(vec2(in.uv.x - x, in.uv.y - y)).rgb;
    var h = sample_source(vec2(in.uv.x,     in.uv.y - y)).rgb;
    var i = sample_source(vec2(in.uv.x + x, in.uv.y - y)).rgb;

    var upsample = e*4.0;
    upsample += (b+d+f+h)*2.0;
    upsample += (a+c+g+i);
    upsample *= 1.0 / 16.0;
    return vec4f(upsample, 0.0);
}

// Final composite pass:
// no blur, mixing is handled by blending, so sample the texture only.
@fragment
fn fs_composite(
    in: VertexOutput
) -> @location(0) vec4f {
    return vec4f(sample_source(vec2(in.uv.x, in.uv.y)).rgb, 0.0);
}
