use crate::renderer::{
    pipeline::{
        key::{PipelineKey, PipelineLayoutKey, VertexLayoutKey},
        library::PipelineLibrary,
        rasterizer_pipeline::{RasterizerPipeline, RasterizerPipelineDescriptor, RasterizerPipelineDescriptorBuilder, UseRasterizerPipeline},
    },
    resources::sampler::SamplerDescription,
};


pub struct SkyboxManager {
    skybox_textures: Option<wgpu::TextureView>,
    skybox_sampler: wgpu::Sampler,
    skybox_bindgroup_layout: wgpu::BindGroupLayout,
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
        device: &wgpu::Device,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        color_attachment_format: wgpu::TextureFormat,
        depth_attachment_format: wgpu::TextureFormat,
        pipelines: &mut PipelineLibrary,
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
        let skybox_sampler = device.create_sampler(&(sd.into()));

        let skybox_bindgroup_layout = device.create_bind_group_layout(&Self::DBGL_SKYBOX);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Skybox Pipeline Layout"),
            bind_group_layouts: &[None, Some(camera_bind_group_layout), None, Some(&skybox_bindgroup_layout)],
            immediate_size: 0
        });

        let shader_module = device.create_shader_module(wgpu::include_wgsl!("../pipeline/shaders/skybox_shaders.wgsl"));
        let targets = [Some(wgpu::ColorTargetState {
            format: color_attachment_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL
        })];
        let rpd: RasterizerPipelineDescriptor = RasterizerPipelineDescriptorBuilder::new(wgpu::VertexState{
            module: &shader_module,
            entry_point: None,
            compilation_options: Default::default(),
            buffers: &[],
        }).set_fragment_state(wgpu::FragmentState{
            module: &shader_module,
            entry_point: None,
            compilation_options: Default::default(),
            targets: &targets,
        }).set_depth_stencil_state(wgpu::DepthStencilState{
            format: depth_attachment_format,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }).build();
        let pipeline = RasterizerPipeline::new(device, rpd, Some(&pipeline_layout));
        pipelines.register_layout(PipelineLayoutKey::Skybox, pipeline_layout);
        pipelines.insert_pipeline(
            PipelineKey::Skybox,
            VertexLayoutKey::EMPTY,
            PipelineLayoutKey::Skybox,
            pipeline,
        );

        Self {
            skybox_textures: None,
            skybox_sampler,
            skybox_bindgroup_layout,
        }
    }

    pub fn set_texture(&mut self, texture: wgpu::Texture) {
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
    /// Camera bind group must be bound, possibly via `SceneContext::bind_globals`.
    /// The fourth bind group will be mutated by the draw call.
    pub fn draw_skybox(
        &self,
        device: &wgpu::Device,
        pipelines: &PipelineLibrary,
        rp: &mut wgpu::RenderPass,
    ) {
        let pipeline = pipelines
            .get(PipelineKey::Skybox, VertexLayoutKey::EMPTY, PipelineLayoutKey::Skybox)
            .expect("skybox pipeline should be registered");

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        rp.set_bind_group(3, &bind_group, &[]);
        rp.set_rasterizer_pipeline(pipeline);
        rp.draw(0..36, 0..1);
    }
}
