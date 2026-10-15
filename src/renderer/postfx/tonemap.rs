use crate::renderer::{
    device_interface::DeviceInterface,
    pipeline::sampler_manager::SamplerManager,
    postfx::postfx_stack::PostFxPass,
};

/// ACES tonemapping pass.
///
/// Reads the HDR target and writes the tonemapped result into the swapchain
/// image. This must be the final pass of the stack.
pub struct TonemapPass {
    bgl_tonemap: wgpu::BindGroupLayout,
    ppl: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,

    input_format: wgpu::TextureFormat,
    output_format: wgpu::TextureFormat
}

impl TonemapPass {
    const DBGL_TONEMAP: wgpu::BindGroupLayoutDescriptor<'static> = wgpu::BindGroupLayoutDescriptor {
        label: Some("Tonemapping descriptor set layout"),
        entries: &[
            // Slot for HDR image
            wgpu::BindGroupLayoutEntry{
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false
                },
                count:None,
            },
            // Slot for the average luminance estimate of the framebuffer.
            wgpu::BindGroupLayoutEntry{
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false
                },
                count:None,
            },
            wgpu::BindGroupLayoutEntry{
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None
            }
        ]
    };

    pub fn new(di: &DeviceInterface, input_format: wgpu::TextureFormat, output_format: wgpu::TextureFormat) -> Self {
        let bgl_tonemap = di.get_device().create_bind_group_layout(&Self::DBGL_TONEMAP);

        let module = di.get_device().create_shader_module(wgpu::include_wgsl!("./shaders/aces_tonemap.wgsl"));
        let layout = di.get_device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl_tonemap)], immediate_size: 0 });

        let ppl = di.get_device().create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Default Tonemapping Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState{ count: 1, mask: !0, alpha_to_coverage_enabled: false },
                fragment: Some(wgpu::FragmentState{
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[ Some(wgpu::ColorTargetState{ format: output_format, blend: None, write_mask: wgpu::ColorWrites::ALL }) ]
                }),
                multiview_mask: None,
                cache: None
            }
        );

        let sampler = di.get_device().create_sampler(&wgpu::SamplerDescriptor{
            label: Some("Tonemapping sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None
        });

        Self { bgl_tonemap, ppl, sampler, input_format, output_format }
    }
}

impl PostFxPass for TonemapPass {
    fn name(&self) -> &'static str { "tonemap" }

    fn input_format(&self) -> wgpu::TextureFormat { self.input_format }
    fn output_format(&self) -> wgpu::TextureFormat { self.output_format }

    fn apply(&mut self, di: &DeviceInterface, _samplers: &mut SamplerManager, ce: &mut wgpu::CommandEncoder, input: &wgpu::TextureView, output: &wgpu::TextureView) {
        // The luminance slot is currently unused by the shader; bind the input
        // there so the descriptor set stays complete.
        let bind_group = di.get_device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_tonemap,
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input)
                },
                wgpu::BindGroupEntry{
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(input)
                },
                wgpu::BindGroupEntry{
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler)
                }
            ]
        });

        let mut rp = ce.begin_render_pass(&wgpu::RenderPassDescriptor{
            label: Some("Tonemap"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment{
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations{
                    load: wgpu::LoadOp::DontCare(unsafe { wgpu::LoadOpDontCare::enabled() }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None
        });
        rp.set_bind_group(0, &bind_group, &[]);
        rp.set_pipeline(&self.ppl);
        rp.draw(0..3, 0..1);
    }
}
