use std::num::NonZero;

use crate::renderer::{
    device_interface::DeviceInterface,
    pipeline::sampler::SamplerDescription,
    postfx::{framebuffers::FramebufferMipChain, postfx_stack::{PostFxContext, PostFxPass}},
};

/// Screen space bloom pass.
///
/// The pass operates in place on the HDR target: it downsamples the input into
/// its private half-resolution mip chain and then upsamples it back additively
/// onto the output.
pub struct BloomPass {
    bgl_bloom: wgpu::BindGroupLayout,
    ppl_bloom_down: wgpu::RenderPipeline,
    ppl_bloom_up: wgpu::RenderPipeline,

    internal_format: wgpu::TextureFormat,
    mip_chain: FramebufferMipChain
}

impl BloomPass {
    const BLOOM_RADIUS: f32 = 0.5;
    const BLOOM_STRENGTH: f64 = 0.5;

    const DBGL_BLOOM: wgpu::BindGroupLayoutDescriptor<'static> = wgpu::BindGroupLayoutDescriptor {
        label: Some("Bloom descriptor set layout"),
        entries: &[
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
            wgpu::BindGroupLayoutEntry{
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None
            }
        ]
    };

    const INPUT_SAMPLER: SamplerDescription = SamplerDescription {
        address_mode: [wgpu::AddressMode::ClampToEdge; 3],
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        // For some big brain reason rust decide to not support a const default trait...
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    };

    pub fn new(di: &DeviceInterface, hdr_format: wgpu::TextureFormat) -> Self {
        let bgl_bloom = di.get_device().create_bind_group_layout(&Self::DBGL_BLOOM);

        let module = di.get_device().create_shader_module(wgpu::include_wgsl!("./shaders/bloom.wgsl"));
        let layout = di.get_device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl_bloom)], immediate_size: 4 });

        let internal_format = hdr_format;

        let ppl_bloom_down = di.get_device().create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Bloom Downsample Pipeline"),
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
                    entry_point: Some("fs_downsample"),
                    compilation_options: Default::default(),
                    targets: &[ Some(wgpu::ColorTargetState{ format: internal_format, blend: None, write_mask: wgpu::ColorWrites::ALL }) ]
                }),
                multiview_mask: None,
                cache: None
            }
        );

        let ppl_bloom_up = di.get_device().create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Bloom Upsample Pipeline"),
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
                    entry_point: Some("fs_upsample"),
                    compilation_options: Default::default(),
                    targets: &[ Some(wgpu::ColorTargetState{
                        format: internal_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Constant,
                                dst_factor: wgpu::BlendFactor::OneMinusConstant,
                                operation: wgpu::BlendOperation::Add
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add
                            }
                        }),
                        write_mask: wgpu::ColorWrites::ALL
                    }) ]
                }),
                multiview_mask: None,
                cache: None
            }
        );

        Self { bgl_bloom, ppl_bloom_down, ppl_bloom_up, internal_format, mip_chain: FramebufferMipChain::new(internal_format) }
    }

    fn build_bind_group(&self, di: &DeviceInterface, tx: &wgpu::TextureView, sp: &wgpu::Sampler) -> wgpu::BindGroup {
        di.get_device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_bloom,
            entries: &[
                wgpu::BindGroupEntry{ binding: 0, resource: wgpu::BindingResource::TextureView(tx) },
                wgpu::BindGroupEntry{ binding: 1, resource: wgpu::BindingResource::Sampler(sp)}
            ]
        })
    }
}

impl PostFxPass for BloomPass {
    fn name(&self) -> &'static str { "bloom" }

    fn input_format(&self) -> wgpu::TextureFormat { self.internal_format }
    fn output_format(&self) -> wgpu::TextureFormat { self.internal_format }

    fn resize(&mut self, di: &DeviceInterface, width: NonZero<u32>, height: NonZero<u32>) {
        self.mip_chain.create_textures(di, width, height);
    }

    fn apply(&mut self, ctx: &mut PostFxContext<'_>) {
        let di = ctx.device;
        let input = ctx.input;
        let output = ctx.output;

        let sampler = ctx.samplers.push_sampler(di.get_device(), Self::INPUT_SAMPLER)
            .expect("Failed to push bloom input sampler.").1;
        let ce = &mut ctx.encoder;

        let mipchain = self.mip_chain.get_textures();
        assert!(mipchain.len() > 0, "Bloom pass mipchain is empty; was resize called?");

        if input.texture().format() != self.internal_format {
            log::error!(
                "Bloom pass input texture format ({:?}) does not match internal format ({:?}).",
                input.texture().format(),
                self.internal_format
            );
        }
        if output.texture().format() != self.internal_format {
            log::error!(
                "Bloom pass output texture format ({:?}) does not match internal format ({:?}).",
                output.texture().format(),
                self.internal_format
            );
        }

        // Blit from framebuffer into the first level of the mipchain
        let fb: &wgpu::TextureView = input;
        {
            let mut rp = ce.begin_render_pass(&wgpu::RenderPassDescriptor{
                label: Some("Bloom downsample from framebuffer"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment{
                    view: &mipchain[0],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations{
                        load: wgpu::LoadOp::DontCare(unsafe { wgpu::LoadOpDontCare::enabled() }),
                        store: wgpu::StoreOp::Store
                    }
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let bg = self.build_bind_group(di, fb, sampler);
            rp.set_pipeline(&self.ppl_bloom_down);
            rp.set_immediates(0, bytemuck::cast_slice(&[Self::BLOOM_RADIUS]));
            rp.set_bind_group(0, &bg, &[]);
            rp.draw(0..3, 0..1);
        }

        // Blit along the mipchain
        for mip in 1..mipchain.len() {
            let mut rp = ce.begin_render_pass(&wgpu::RenderPassDescriptor{
                label: Some("Bloom downsample chain"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment{
                    view: &mipchain[mip],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations{
                        load: wgpu::LoadOp::DontCare(unsafe { wgpu::LoadOpDontCare::enabled() }),
                        store: wgpu::StoreOp::Store
                    }
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let bg = self.build_bind_group(di, &mipchain[mip - 1], sampler);
            rp.set_pipeline(&self.ppl_bloom_down);
            rp.set_immediates(0, bytemuck::cast_slice(&[Self::BLOOM_RADIUS]));
            rp.set_bind_group(0, &bg, &[]);
            rp.draw(0..3, 0..1);
        }

        for mip in (1..mipchain.len()).rev() {
            let mut rp = ce.begin_render_pass(&wgpu::RenderPassDescriptor{
                label: Some("Bloom upsample chain"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment{
                    view: &mipchain[mip - 1],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations{
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store
                    }
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let bg = self.build_bind_group(di, &mipchain[mip], sampler);
            rp.set_pipeline(&self.ppl_bloom_up);
            rp.set_blend_constant(wgpu::Color{r: 1.0, g: 1.0, b: 1.0, a: 1.0});
            rp.set_immediates(0, bytemuck::cast_slice(&[Self::BLOOM_RADIUS]));
            rp.set_bind_group(0, &bg, &[]);
            rp.draw(0..3, 0..1);
        }

        // Finally blit to the framebuffer
        {
            let mut rp = ce.begin_render_pass(&wgpu::RenderPassDescriptor{
                label: Some("Bloom unsample and write"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment{
                    view: output,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations{
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store
                    }
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let bg = self.build_bind_group(di, &mipchain[0], sampler);
            rp.set_pipeline(&self.ppl_bloom_up);
            rp.set_blend_constant(wgpu::Color{r: Self::BLOOM_STRENGTH, g: Self::BLOOM_STRENGTH, b: Self::BLOOM_STRENGTH, a: 1.0});
            rp.set_immediates(0, bytemuck::cast_slice(&[Self::BLOOM_RADIUS]));
            rp.set_bind_group(0, &bg, &[]);
            rp.draw(0..3, 0..1);
        }
    }
}
