use crate::renderer::{
    device_interface::DeviceInterface, pipeline::sampler::SamplerDescription, postfx::{bloom::BloomPass, postfx_stack::{PostFxContext, PostFxPass, PostFxResourceCache}},
};

const TONEMAP_SAMPLER: SamplerDescription = SamplerDescription {
    address_mode: [wgpu::AddressMode::ClampToEdge; 3],
    mag_filter: wgpu::FilterMode::Linear,
    min_filter: wgpu::FilterMode::Linear,
    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
    // For some big brain reason rust decide to not support a const default trait...
    compare: None,
    anisotropy_clamp: 1,
    border_color: None,
};

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
        // Unused for ACES tonemapping.
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

fn build_pipeline(di: &DeviceInterface, layout: &wgpu::PipelineLayout, output_format: wgpu::TextureFormat, fs_name: &str) -> wgpu::RenderPipeline {
    let module = di.get_device().create_shader_module(wgpu::include_wgsl!("./shaders/tonemap.wgsl"));
    di.get_device().create_render_pipeline(
        &wgpu::RenderPipelineDescriptor {
            label: Some("Tonemapping"),
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
                entry_point: Some(fs_name),
                compilation_options: Default::default(),
                targets: &[ Some(wgpu::ColorTargetState{ format: output_format, blend: None, write_mask: wgpu::ColorWrites::ALL }) ]
            }),
            multiview_mask: None,
            cache: None
        }
    )
}

fn generate_bind_group(di: &DeviceInterface, layout: &wgpu::BindGroupLayout, color_in: &wgpu::TextureView, lum_in: &wgpu::TextureView, sampler: &wgpu::Sampler) -> wgpu::BindGroup {
    di.get_device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: layout,
        entries: &[
            wgpu::BindGroupEntry{
                binding: 0,
                resource: wgpu::BindingResource::TextureView(color_in)
            },
            wgpu::BindGroupEntry{
                binding: 1,
                resource: wgpu::BindingResource::TextureView(lum_in)
            },
            wgpu::BindGroupEntry{
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler)
            }
        ]
    })
}

/// ACES tonemapping pass.
/// 
/// The ACES tonemapping does not depend on the average luminance of the image,
/// and can be performed if no color pyramid is built previously.
///
/// Reads the HDR target and writes the tonemapped result into the swapchain
/// image. This must be the final pass of the stack.
pub struct AcesTonemapPass {
    bgl_tonemap: wgpu::BindGroupLayout,
    ppl: wgpu::RenderPipeline,

    input_format: wgpu::TextureFormat,
    output_format: wgpu::TextureFormat
}

impl AcesTonemapPass {
    pub fn new(di: &DeviceInterface, input_format: wgpu::TextureFormat, output_format: wgpu::TextureFormat) -> Self {
        let bgl_tonemap = di.get_device().create_bind_group_layout(&DBGL_TONEMAP);
        let layout = di.get_device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl_tonemap)], immediate_size: 0 });
        Self { bgl_tonemap, ppl: build_pipeline(di, &layout, output_format, "fs_aces"), input_format, output_format }
    }
}

trait TonemapPass: PostFxPass {
    fn name(&self) -> &'static str;
    fn input_format(&self) -> wgpu::TextureFormat;
    fn output_format(&self) -> wgpu::TextureFormat;

    fn requires_average_luminance() -> bool { false }
    fn get_tonemap_bind_group_layout(&self) -> &wgpu::BindGroupLayout;
    fn get_tonemap_pipeline(&self) -> &wgpu::RenderPipeline;
}

impl<T: TonemapPass> PostFxPass for T {
    fn name(&self) -> &'static str { TonemapPass::name(self) }
    fn input_format(&self) -> wgpu::TextureFormat { TonemapPass::input_format(self) }
    fn output_format(&self) -> wgpu::TextureFormat {
        let format = TonemapPass::output_format(self);

        // We are using Vulkan backend, which means even if we use Unorm format,
        // the presentation engine will interpret texels as if they are in sRGB
        // color space. This is definitely not what we want.
        // See https://stackoverflow.com/questions/66401081/vulkan-swapchain-format-unorm-vs-srgb
        if !format.is_srgb() {
            log::warn!("Output format for tonemapping is not in sRGB space. Gamma correction may be off.")
        }
        format
    }

    fn apply(&mut self, ctx: &mut PostFxContext<'_>, cache: &mut PostFxResourceCache) {
        let di = ctx.device;
        let input = ctx.input;
        let output = ctx.output;

        let sampler = match ctx.samplers.push_sampler(di.get_device(), TONEMAP_SAMPLER) {
            Ok((_, sp)) => sp,
            Err(e) => {
                log::error!("failed to get sampler for tonemapping: {:?}", e);
                ctx.samplers.get_default_sampler().1
            },
        };
        
        let lum = if Self::requires_average_luminance() {
            let avg_lum = cache.try_get(BloomPass::BLOOM_MIPCHAIN_HIGHEST_NAME);
            match avg_lum {
                Some(view) => view.as_ref(),
                None => panic!("Average luminance texture is not available!")
            }
        } else {
            input
        };
        let bind_group = generate_bind_group(di, self.get_tonemap_bind_group_layout(), input, lum, sampler);

        let mut rp = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor{
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
        rp.set_pipeline(&self.get_tonemap_pipeline());
        rp.draw(0..3, 0..1);
    }
}

impl TonemapPass for AcesTonemapPass {
    fn name(&self) -> &'static str { "tonemap" }

    fn input_format(&self) -> wgpu::TextureFormat { self.input_format }
    fn output_format(&self) -> wgpu::TextureFormat { self.output_format }
    
    fn get_tonemap_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bgl_tonemap
    }
    fn get_tonemap_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.ppl
    }
}
