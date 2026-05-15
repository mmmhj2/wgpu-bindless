use wgpu::{RenderPipelineDescriptor};

use crate::renderer::device_interface::DeviceInterface;

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
    pub front_face      : wgpu::FrontFace,
    pub vertex          : wgpu::VertexState<'a>,
    pub fragment        : Option<wgpu::FragmentState<'a>>,
    pub depthstencil    : Option<wgpu::DepthStencilState>
}

impl<'a> From<wgpu::VertexState<'a>> for RasterizerPipelineDescriptor<'a> {
    fn from(value: wgpu::VertexState<'a>) -> Self {
        RasterizerPipelineDescriptor {
            cull_mode: None,
            front_face: wgpu::FrontFace::Ccw,
            vertex: value,
            fragment: None,
            depthstencil: None
        }
    }
}

impl<'a> From<(wgpu::VertexState<'a>, wgpu::FragmentState<'a>)> for RasterizerPipelineDescriptor<'a> {
    fn from(value: (wgpu::VertexState<'a>, wgpu::FragmentState<'a>)) -> Self {
        RasterizerPipelineDescriptor {
            cull_mode: None,
            front_face: wgpu::FrontFace::Ccw,
            vertex: value.0,
            fragment: Some(value.1),
            depthstencil: None
        }
    }
}

impl<'a> From<(wgpu::VertexState<'a>, wgpu::FragmentState<'a>, wgpu::DepthStencilState)> for RasterizerPipelineDescriptor<'a> {
    fn from(value: (wgpu::VertexState<'a>, wgpu::FragmentState<'a>, wgpu::DepthStencilState)) -> Self {
        RasterizerPipelineDescriptor {
            cull_mode: None,
            front_face: wgpu::FrontFace::Ccw,
            vertex: value.0,
            fragment: Some(value.1),
            depthstencil: Some(value.2)
        }
    }
}

impl<'a> From<RasterizerPipelineDescriptor<'a>> for wgpu::RenderPipelineDescriptor<'a> {
    fn from(rpd: RasterizerPipelineDescriptor<'a>) -> Self {
        RenderPipelineDescriptor{
            label: None,
            layout: None,
            vertex: rpd.vertex,
            fragment: rpd.fragment,
            primitive: wgpu::PrimitiveState{
                cull_mode: rpd.cull_mode,
                front_face: rpd.front_face,
                ..DEFAULT_PRIMITIVE_STATE
            },
            depth_stencil: rpd.depthstencil,
            multisample: DEFAULT_MULTISAMPLE_STATE,
            multiview_mask: None,
            cache: None
        }
    }
}

pub struct RasterizerPipeline {
    pipeline : wgpu::RenderPipeline
}

impl RasterizerPipeline {
    pub fn new(di : &DeviceInterface, rpd : RasterizerPipelineDescriptor, layout : Option<&wgpu::PipelineLayout>) -> RasterizerPipeline {
        let mut desc: RenderPipelineDescriptor<'_> = rpd.into();
        desc.layout = layout;

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
