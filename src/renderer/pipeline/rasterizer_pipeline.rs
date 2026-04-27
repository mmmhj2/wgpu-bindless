use wgpu::{RenderPipelineDescriptor};

use crate::renderer::{device_interface::DeviceInterface, pipeline::rasterizer_pipeline};

const DEFAULT_PRIMITIVE_STATE : wgpu::PrimitiveState = wgpu::PrimitiveState{
    topology: wgpu::PrimitiveTopology::TriangleList,
    strip_index_format: None,
    front_face: wgpu::FrontFace::Ccw,
    cull_mode: Some(wgpu::Face::Back),
    unclipped_depth: false,
    polygon_mode: wgpu::PolygonMode::Fill,
    conservative: false
};

const DEFAULT_MULTISAMPLE_STATE : wgpu::MultisampleState = wgpu::MultisampleState {
    count: 1,
    mask: !0,
    alpha_to_coverage_enabled: false
};

/// Simplified version of RenderPipelineDescriptor
pub struct RasterizerPipelineDescriptor<'a> {
    pub cull_mode       : Option<wgpu::Face>,
    pub vertex          : wgpu::VertexState<'a>,
    pub fragment        : wgpu::FragmentState<'a>,
    pub depthstencil    : Option<wgpu::DepthStencilState>
}

pub struct RasterizerPipeline {
    pipeline : wgpu::RenderPipeline
}

impl RasterizerPipeline {
    pub fn new(di : &DeviceInterface, rpd : RasterizerPipelineDescriptor, layout : Option<&wgpu::PipelineLayout>) -> RasterizerPipeline {
        let desc = RenderPipelineDescriptor{
            label: None,
            layout: layout,
            vertex: rpd.vertex,
            fragment: Some(rpd.fragment),
            primitive: wgpu::PrimitiveState{
                cull_mode: rpd.cull_mode,
                ..DEFAULT_PRIMITIVE_STATE
            },
            depth_stencil: rpd.depthstencil,
            multisample: DEFAULT_MULTISAMPLE_STATE,
            multiview_mask: None,
            cache: None
        };

        RasterizerPipeline{
            pipeline : di.get_device().create_render_pipeline(&desc)
        }
    }
}

pub trait UseRasterizerPipeline {
    fn set_rasterizer_pipeline(&mut self, p: &RasterizerPipeline) -> ();
}

impl UseRasterizerPipeline for wgpu::RenderPass<'_> {
    fn set_rasterizer_pipeline(&mut self, p: &RasterizerPipeline) { self.set_pipeline(&p.pipeline) }
}
