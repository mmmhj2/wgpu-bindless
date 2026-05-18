
use std::{cmp::max, num::NonZero};


use crate::renderer::device_interface::DeviceInterface;

struct Tonemapper {
    bgl_tonemap: wgpu::BindGroupLayout,
    ppl: wgpu::RenderPipeline
}

impl Tonemapper {
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
            // Slot for the last level of mipmap chain.
            // Used to estimate the average luminance of the framebuffer.
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

    pub fn new(di: &DeviceInterface) -> Self {

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
                    targets: &[ Some(wgpu::ColorTargetState{ format: di.get_default_texture_format(), blend: None, write_mask: wgpu::ColorWrites::ALL }) ]
                }),
                multiview_mask: None,
                cache: None
            }
        );
        Self {
            bgl_tonemap,
            ppl
        }
    }
}

struct Bloomer {
    bgl_bloom: wgpu::BindGroupLayout,
    ppl_bloom_down: wgpu::RenderPipeline,
    ppl_bloom_up: wgpu::RenderPipeline
}

impl Bloomer {
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

    pub fn build_bind_group(&self, di: &DeviceInterface, tx: &wgpu::TextureView, sp: &wgpu::Sampler) -> wgpu::BindGroup {
        di.get_device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_bloom,
            entries: &[
                wgpu::BindGroupEntry{ binding: 0, resource: wgpu::BindingResource::TextureView(tx) },
                wgpu::BindGroupEntry{ binding: 1, resource: wgpu::BindingResource::Sampler(sp)}
            ]
        })
    }

    pub fn new (di: &DeviceInterface, hdr_format: wgpu::TextureFormat) -> Self {
        let bgl_bloom = di.get_device().create_bind_group_layout(&Self::DBGL_BLOOM);

        let module = di.get_device().create_shader_module(wgpu::include_wgsl!("./shaders/bloom.wgsl"));
        let layout = di.get_device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl_bloom)], immediate_size: 4 });

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
                    targets: &[ Some(wgpu::ColorTargetState{ format: hdr_format, blend: None, write_mask: wgpu::ColorWrites::ALL }) ]
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
                        format: hdr_format,
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

        Self {
            bgl_bloom,
            ppl_bloom_down,
            ppl_bloom_up
        }
    }
}

struct Framebuffer {
    format: wgpu::TextureFormat,
    label: &'static str,
    texture: Option<wgpu::TextureView>
}
impl Framebuffer {
    fn new(format: wgpu::TextureFormat, label: &'static str) -> Self {
        Self { format, label, texture: None }
    }

    fn get_format(&self) -> wgpu::TextureFormat { self.format }
    fn get_texture(&self) -> Option<&wgpu::TextureView> { self.texture.as_ref() }

    fn create_texture(&mut self, di: &DeviceInterface, width: NonZero<u32>, height: NonZero<u32>) {
        self.texture = Some(di.get_device().create_texture(&wgpu::TextureDescriptor{
            label: Some(self.label),
            size: wgpu::Extent3d { width: width.get(), height: height.get(), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[]
        }).create_view(&Default::default()));
    }
}

struct FramebufferMipChain {
    format: wgpu::TextureFormat,
    texture: Vec<wgpu::TextureView>
}

impl FramebufferMipChain {
    fn new(format: wgpu::TextureFormat) -> Self {
        Self { format, texture: Vec::new() }
    }

    fn get_format(&self) -> wgpu::TextureFormat { self.format }
    fn get_textures(&self) -> &Vec<wgpu::TextureView> { self.texture.as_ref() }
    fn get_mipchain_length(&self) -> Option<NonZero<usize>> { NonZero::new(self.texture.len()) }
    
    /// Create a mipchain from the framebuffer size.
    /// 
    /// The first miplevel of the mipchain will have a size of (full_res_width / 2, full_res_height / 2).
    fn create_textures(&mut self, di: &DeviceInterface, full_res_width: NonZero<u32>, full_res_height: NonZero<u32>) {
        self.texture.clear();
        let width = full_res_width.get() / 2;
        let height = full_res_height.get() / 2;
        let miplevels = (max(width, height) as f32).log2().floor() as u32 + 1;

        let texture = di.get_device().create_texture(
            &wgpu::TextureDescriptor {
                label: Some("Framebuffer mipchain"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: miplevels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            }
        );

        for i in 0..miplevels {
            self.texture.push(texture.create_view(&wgpu::TextureViewDescriptor{ base_mip_level: i, mip_level_count: Some(1), ..Default::default() }))
        }
    }
}

/// Framebuffers and the post-processing stack.
pub struct Framebuffers {
    fb_size     : Option<(NonZero<u32>, NonZero<u32>)>,
    fb_sampler  : wgpu::Sampler,

    hdr_fb      : Framebuffer,
    tm          : Tonemapper,

    bloom_chain : FramebufferMipChain,
    bloom       : Bloomer
}

impl Framebuffers {

    pub fn new(di: &DeviceInterface, hdr_format: wgpu::TextureFormat) -> Self {
        Self {
            fb_size: None,
            hdr_fb: Framebuffer::new(hdr_format, "HDR Attachment"),
            fb_sampler: di.get_device().create_sampler(&wgpu::SamplerDescriptor{
                label: None,
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
            }),
            tm: Tonemapper::new(di),
            bloom_chain: FramebufferMipChain::new(hdr_format),
            bloom: Bloomer::new(di, hdr_format)
        }
    }

    pub fn configure_with_surface(&mut self, di: &DeviceInterface) {
        let width = NonZero::<u32>::new(di.get_current_surface_configuration().width).expect("framebuffer width should be larger than zero.");
        let height = NonZero::<u32>::new(di.get_current_surface_configuration().height).expect("framebuffer height should be larger than zero.");

        let need_rebuild = if let Some(t) = self.fb_size {
            !(t.0 == width && t.1 == height)
        } else {
            true
        };

        if need_rebuild {
            log::info!("Rebuilding framebuffers with size {}x{}", width, height);
            self.hdr_fb.create_texture(di, width, height);
            self.bloom_chain.create_textures(di, width, height);
            self.fb_size = Some((width, height));
        }
        
    }

    pub fn get_hdr_format(&self) -> wgpu::TextureFormat {
        self.hdr_fb.get_format()
    }

    pub fn get_hdr_framebuffer(&self) -> Option<&wgpu::TextureView> {
        match self.hdr_fb.get_texture() {
            Some(t) => Some(t),
            None => None
        }
    }

    /// Perform bloom within the command encoder.
    /// 
    /// This method will begin several render passes to perform the screen space blooming.
    /// It will first downsample from the HDR framebuffer into the mipchain, and then upsample from the mipchain into the framebuffer.
    pub fn bloom(&self, di: &DeviceInterface, ce: &mut wgpu::CommandEncoder, bloom_radius: f32, strength: f64) {
        let mipchain = self.bloom_chain.get_textures();
        assert!(mipchain.len() > 0);

        // Blit from framebuffer into the first level of the mipchain
        let fb: &wgpu::TextureView = self.hdr_fb.get_texture().expect("call configure_with_surface before this method.");
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

            let bg = self.bloom.build_bind_group(di, fb, &self.fb_sampler);
            rp.set_pipeline(&self.bloom.ppl_bloom_down);
            rp.set_immediates(0, bytemuck::cast_slice(&[bloom_radius]));
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

            let bg = self.bloom.build_bind_group(di, &mipchain[mip - 1], &self.fb_sampler);
            rp.set_pipeline(&self.bloom.ppl_bloom_down);
            rp.set_immediates(0, bytemuck::cast_slice(&[bloom_radius]));
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

            let bg = self.bloom.build_bind_group(di, &mipchain[mip], &self.fb_sampler);
            rp.set_pipeline(&self.bloom.ppl_bloom_up);
            rp.set_blend_constant(wgpu::Color{r: 1.0, g: 1.0, b: 1.0, a: 1.0});
            rp.set_immediates(0, bytemuck::cast_slice(&[bloom_radius]));
            rp.set_bind_group(0, &bg, &[]);
            rp.draw(0..3, 0..1);
        }

        // Finally blit to the framebuffer
        {
            let mut rp = ce.begin_render_pass(&wgpu::RenderPassDescriptor{
                label: Some("Bloom unsample and write"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment{
                    view: fb,
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

            let bg = self.bloom.build_bind_group(di, &mipchain[0], &self.fb_sampler);
            rp.set_pipeline(&self.bloom.ppl_bloom_up);
            rp.set_blend_constant(wgpu::Color{r: strength, g: strength, b: strength, a: 1.0});
            rp.set_immediates(0, bytemuck::cast_slice(&[bloom_radius]));
            rp.set_bind_group(0, &bg, &[]);
            rp.draw(0..3, 0..1);
        }
    }

    /// Perform tonemap with the specified pipeline.
    /// 
    /// This method will begin a render pass with the view as the only color attachment.
    /// The render pass will contain only a binding and one draw call with three indices.
    /// No index or vertex buffers are bound, as vertices should be generated by the vertex shader.
    /// The HDR framebuffer will be bound to the 0-th set and 0-th binding for texture and 1th binding for sampler.
    /// Sample it to do tonemapping.
    /// The format of the color attachment should be *the same* as the surface texture format.
    /// 
    /// Safety
    /// ---
    /// The load operation on the color attachment is *Don't Care*.
    /// The primitive generated by the vertex shader should cover the whole screen space, which is true for the default pipeline.
    /// Otherwise the behavior is undefined.
    pub fn tonemap_to(&self, di: &DeviceInterface, ce: &mut wgpu::CommandEncoder, final_texture_view: &wgpu::TextureView) {

        let bind_group = di.get_device().create_bind_group(&wgpu::BindGroupDescriptor { 
            label: None,
            layout: &self.tm.bgl_tonemap,
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.hdr_fb.get_texture().expect("call configure_with_surface before this method."))
                },
                wgpu::BindGroupEntry{
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(self.bloom_chain.get_textures().last().expect("call configure_with_surface before this method."))
                },
                wgpu::BindGroupEntry{
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.fb_sampler)
                }
            ]
        });

        let mut rp = ce.begin_render_pass(&wgpu::RenderPassDescriptor{
            label: Some("Tonemap"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment{
                view: final_texture_view,
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
        rp.set_pipeline(&self.tm.ppl);
        rp.draw(0..3, 0..1);
    }
}
