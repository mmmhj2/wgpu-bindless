struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) uv0: vec2<f32>
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv0: vec2<f32>
};

@vertex
fn vs_main(
    model: VertexInput,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = camera.vp_matrix * model_matrices[instance_index] * vec4<f32>(model.position.xyz, 1.0);

    out.color = model.color;
    out.uv0 = model.uv0;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color * sample_diffuse(in.uv0);
}
