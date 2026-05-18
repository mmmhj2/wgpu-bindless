
const VERTICES = array(
    vec3f(-1.0f,  1.0f, -1.0f),
    vec3f(-1.0f, -1.0f, -1.0f),
    vec3f( 1.0f, -1.0f, -1.0f),
    vec3f( 1.0f, -1.0f, -1.0f),
    vec3f( 1.0f,  1.0f, -1.0f),
    vec3f(-1.0f,  1.0f, -1.0f),

    vec3f(-1.0f, -1.0f,  1.0f),
    vec3f(-1.0f, -1.0f, -1.0f),
    vec3f(-1.0f,  1.0f, -1.0f),
    vec3f(-1.0f,  1.0f, -1.0f),
    vec3f(-1.0f,  1.0f,  1.0f),
    vec3f(-1.0f, -1.0f,  1.0f),

    vec3f( 1.0f, -1.0f, -1.0f),
    vec3f( 1.0f, -1.0f,  1.0f),
    vec3f( 1.0f,  1.0f,  1.0f),
    vec3f( 1.0f,  1.0f,  1.0f),
    vec3f( 1.0f,  1.0f, -1.0f),
    vec3f( 1.0f, -1.0f, -1.0f),

    vec3f(-1.0f, -1.0f,  1.0f),
    vec3f(-1.0f,  1.0f,  1.0f),
    vec3f( 1.0f,  1.0f,  1.0f),
    vec3f( 1.0f,  1.0f,  1.0f),
    vec3f( 1.0f, -1.0f,  1.0f),
    vec3f(-1.0f, -1.0f,  1.0f),

    vec3f(-1.0f,  1.0f, -1.0f),
    vec3f( 1.0f,  1.0f, -1.0f),
    vec3f( 1.0f,  1.0f,  1.0f),
    vec3f( 1.0f,  1.0f,  1.0f),
    vec3f(-1.0f,  1.0f,  1.0f),
    vec3f(-1.0f,  1.0f, -1.0f),

    vec3f(-1.0f, -1.0f, -1.0f),
    vec3f(-1.0f, -1.0f,  1.0f),
    vec3f( 1.0f, -1.0f, -1.0f),
    vec3f( 1.0f, -1.0f, -1.0f),
    vec3f(-1.0f, -1.0f,  1.0f),
    vec3f( 1.0f, -1.0f,  1.0f)
);

struct Camera {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    vp_matrix: mat4x4<f32>
};
@group(1) @binding(0)
var<uniform> camera: Camera;
@group(2) @binding(0)
var skybox_tx: texture_cube<f32>;
@group(2) @binding(1)
var skybox_sp: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uvw: vec3f
}

@vertex
fn vs_main(
    @builtin(vertex_index) gl_VertexIndex: u32
) -> VertexOutput {
    var out: VertexOutput;
    out.uvw = VERTICES[gl_VertexIndex];
    var view_dir = camera.view_matrix * vec4f(VERTICES[gl_VertexIndex], 0.0);
    out.clip_position = camera.proj_matrix * vec4f(view_dir.xyz, 1.0);
    out.clip_position = out.clip_position.xyww;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(skybox_tx, skybox_sp, in.uvw);
}
