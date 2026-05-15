use crate::renderer::{device_interface::DeviceInterface, pipeline::{camera::CameraManager, rasterizer_pipeline::RasterizerPipelineDescriptor, sampler::SamplerDescription, texture::Texture}};


pub struct SkyboxManager {
    skybox_textures: Option<wgpu::TextureView>,
    skybox_sampler: wgpu::Sampler,
    skybox_bindgroup_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline
}

impl SkyboxManager {

    const DBG_SKYBOX: [wgpu::BindGroupLayoutEntry; 2] = [
        wgpu::BindGroupLayoutEntry{
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::Cube, multisampled: false },
            count: None
        },
        wgpu::BindGroupLayoutEntry{
            binding: 1,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None
        }
    ];
    const DBGL_SKYBOX: wgpu::BindGroupLayoutDescriptor<'static> = wgpu::BindGroupLayoutDescriptor {
        label: Some("Skybox Descriptor Set Layout"),
        entries: &Self::DBG_SKYBOX,
    };

    pub fn new(
        di: &DeviceInterface,
        cameras: &CameraManager,
        color_attachment_format: wgpu::TextureFormat,
        depth_attachment_format: wgpu::TextureFormat
    ) -> Self {

        let sd = &SamplerDescription{
            address_mode: [wgpu::AddressMode::Repeat, wgpu::AddressMode::Repeat, wgpu::AddressMode::Repeat ],
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None
        };
        let skybox_sampler = di.get_device().create_sampler(&(sd.into()));

        let skybox_bindgroup_layout = di.get_device().create_bind_group_layout(&Self::DBGL_SKYBOX);

        let pipeline_layout = di.get_device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Skybox Pipeline Layout"),
            bind_group_layouts: &[None, Some(cameras.get_bind_group_layout()), Some(&skybox_bindgroup_layout)],
            immediate_size: 0
        });

        let shader_module = di.get_device().create_shader_module(wgpu::include_wgsl!("./shaders/skybox_shaders.wgsl"));
        let vertex_state = wgpu::VertexState{
            module: &shader_module,
            entry_point: None,
            compilation_options: Default::default(),
            buffers: &[],
        };
        let fragment_state = wgpu::FragmentState{
            module: &shader_module,
            entry_point: None,
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format: color_attachment_format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
        };
        let ds_state = wgpu::DepthStencilState{
            format: depth_attachment_format,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        };
        let rpd: RasterizerPipelineDescriptor = (vertex_state, fragment_state, ds_state).into();
        let mut pd: wgpu::RenderPipelineDescriptor = rpd.into();
        pd.layout = Some(&pipeline_layout);
        let pipeline = di.get_device().create_render_pipeline(&pd);

        Self {
            skybox_textures: None,
            skybox_sampler,
            skybox_bindgroup_layout,
            pipeline
        }
    }

    pub fn set_texture(&mut self, texture: Texture) {
        let texture: wgpu::Texture = texture.into();
        assert_eq!(texture.depth_or_array_layers(), 6);
        self.skybox_textures = Some(texture.create_view(&wgpu::TextureViewDescriptor{
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        }));
    }

    /// Draw skybox on the given render pass.
    /// 
    /// The render pass has to be started with the same color and depth format
    /// specified by the constructor.
    /// Camera bind group must be bound, possibly via the `prepare_render_pass`
    /// call.
    /// The second bind group will be mutated by the draw call.
    pub fn draw_skybox(&self, di: &DeviceInterface, rp: &mut wgpu::RenderPass) {
        let bind_group = di.get_device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.skybox_bindgroup_layout,
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(self.skybox_textures.as_ref().expect("set skybox texture before drawing.")),
                },
                wgpu::BindGroupEntry{
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.skybox_sampler)
                }
            ]
        });
        rp.set_bind_group(2, &bind_group, &[]);
        rp.set_pipeline(&self.pipeline);
        rp.draw(0..36, 0..1);
    }
}
