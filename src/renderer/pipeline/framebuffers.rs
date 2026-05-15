
use crate::renderer::device_interface::DeviceInterface;

struct Tonemapper {
    bgl_tonemap: wgpu::BindGroupLayout,
    ppl: wgpu::RenderPipeline
}

impl Tonemapper {
    const DBGL_TONEMAP: wgpu::BindGroupLayoutDescriptor<'static> = wgpu::BindGroupLayoutDescriptor {
        label: Some("Tonemapping descriptor set layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry{
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false
                },
                count:None,
            },
            wgpu::BindGroupLayoutEntry{
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None
            }
        ]
    };

    pub fn new(di: &DeviceInterface) -> Self {

        let bgl_tonemap = di.get_device().create_bind_group_layout(&Self::DBGL_TONEMAP);

        let module = di.get_device().create_shader_module(wgpu::include_wgsl!("./shaders/reinhard_tonemap.wgsl"));
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

    fn create_texture(&mut self, di: &DeviceInterface, width: u32, height: u32) {
        self.texture = Some(di.get_device().create_texture(&wgpu::TextureDescriptor{
            label: Some(self.label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[]
        }).create_view(&Default::default()));
    }
}

/// Framebuffers and the post-processing stack.
pub struct Framebuffers {
    fb_size     : Option<(u32, u32)>,
    hdr_fb      : Framebuffer,
    hdr_fb_sp   : wgpu::Sampler,
    tm          : Tonemapper
}

impl Framebuffers {

    pub fn new(di: &DeviceInterface, hdr_format: wgpu::TextureFormat) -> Self {
        Self {
            fb_size: None,
            hdr_fb: Framebuffer::new(hdr_format, "HDR Attachment"),
            hdr_fb_sp: di.get_device().create_sampler(&wgpu::SamplerDescriptor{
                label: None,
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                lod_min_clamp: 0.0,
                lod_max_clamp: 32.0,
                compare: None,
                anisotropy_clamp: 1,
                border_color: None
            }),
            tm: Tonemapper::new(di)
        }
    }

    pub fn configure_with_surface(&mut self, di: &DeviceInterface) {
        let width = di.get_current_surface_configuration().width;
        let height = di.get_current_surface_configuration().height;

        let need_rebuild = if let Some(t) = self.fb_size {
            !(t.0 == width && t.1 == height)
        } else {
            true
        };

        if need_rebuild {
            log::info!("Rebuilding framebuffers with size {}x{}", width, height);
            self.hdr_fb.create_texture(di, width, height);
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
    /// The primitive generated by the vertex shader should cover the whole screen space.
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
                    resource: wgpu::BindingResource::Sampler(&self.hdr_fb_sp)
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
