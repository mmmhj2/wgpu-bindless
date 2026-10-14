
// Bloom shader adapted from https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom
// which in turn is based on the ACM Siggraph 2014 presentation of Call of Duty.

// Maybe we should not use binding array as it might confuse synchronization
// because we need to use the source textures as render targets.
@group(0) @binding(0)
var source_tx: texture_2d<f32>;

// This sampler is expected to be a bilinear, clamp-to-edge one.
@group(0) @binding(1)
var source_sp: sampler;

struct Immediates {
    blur_radius: f32
};
var<immediate> immediates: Immediates;

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(1) uv: vec2f
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
fn fs_downsample(
    in: VertexOutput
) -> @location(0) vec4f {
    var source_texels = vec2f(textureDimensions(source_tx, 0));
    var x = 1.0 / source_texels.x;
    var y = 1.0 / source_texels.y;

    var a = textureSample(source_tx, source_sp, vec2(in.uv.x - 2*x, in.uv.y + 2*y)).rgb;
    var b = textureSample(source_tx, source_sp, vec2(in.uv.x,       in.uv.y + 2*y)).rgb;
    var c = textureSample(source_tx, source_sp, vec2(in.uv.x + 2*x, in.uv.y + 2*y)).rgb;

    var d = textureSample(source_tx, source_sp, vec2(in.uv.x - 2*x, in.uv.y)).rgb;
    var e = textureSample(source_tx, source_sp, vec2(in.uv.x,       in.uv.y)).rgb;
    var f = textureSample(source_tx, source_sp, vec2(in.uv.x + 2*x, in.uv.y)).rgb;

    var g = textureSample(source_tx, source_sp, vec2(in.uv.x - 2*x, in.uv.y - 2*y)).rgb;
    var h = textureSample(source_tx, source_sp, vec2(in.uv.x,       in.uv.y - 2*y)).rgb;
    var i = textureSample(source_tx, source_sp, vec2(in.uv.x + 2*x, in.uv.y - 2*y)).rgb;

    var j = textureSample(source_tx, source_sp, vec2(in.uv.x - x, in.uv.y + y)).rgb;
    var k = textureSample(source_tx, source_sp, vec2(in.uv.x + x, in.uv.y + y)).rgb;
    var l = textureSample(source_tx, source_sp, vec2(in.uv.x - x, in.uv.y - y)).rgb;
    var m = textureSample(source_tx, source_sp, vec2(in.uv.x + x, in.uv.y - y)).rgb;

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

    var a = textureSample(source_tx, source_sp, vec2(in.uv.x - x, in.uv.y + y)).rgb;
    var b = textureSample(source_tx, source_sp, vec2(in.uv.x,     in.uv.y + y)).rgb;
    var c = textureSample(source_tx, source_sp, vec2(in.uv.x + x, in.uv.y + y)).rgb;

    var d = textureSample(source_tx, source_sp, vec2(in.uv.x - x, in.uv.y)).rgb;
    var e = textureSample(source_tx, source_sp, vec2(in.uv.x,     in.uv.y)).rgb;
    var f = textureSample(source_tx, source_sp, vec2(in.uv.x + x, in.uv.y)).rgb;

    var g = textureSample(source_tx, source_sp, vec2(in.uv.x - x, in.uv.y - y)).rgb;
    var h = textureSample(source_tx, source_sp, vec2(in.uv.x,     in.uv.y - y)).rgb;
    var i = textureSample(source_tx, source_sp, vec2(in.uv.x + x, in.uv.y - y)).rgb;

    var upsample = e*4.0;
    upsample += (b+d+f+h)*2.0;
    upsample += (a+c+g+i);
    upsample *= 1.0 / 16.0;
    return vec4f(upsample, 0.0);
}
