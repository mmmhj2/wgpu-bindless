use wgpu::{RenderPipelineDescriptor};

use crate::renderer::device_interface::DeviceInterface;

const DEFAULT_PRIMITIVE_STATE : wgpu::PrimitiveState = wgpu::PrimitiveState{
    topology: wgpu::PrimitiveTopology::TriangleList,
    strip_index_format: None,
    front_face: wgpu::FrontFace::Ccw,
    cull_mode: None,
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
    cull_mode       : Option<wgpu::Face>,
    front_face      : wgpu::FrontFace,
    vertex          : wgpu::VertexState<'a>,
    fragment        : Option<wgpu::FragmentState<'a>>,
    depthstencil    : Option<wgpu::DepthStencilState>
}
pub struct RasterizerPipelineDescriptorBuilder<'a> {
    rpd: RasterizerPipelineDescriptor<'a>
}

impl<'a> RasterizerPipelineDescriptorBuilder<'a> {
    /// Create a descriptor builder.
    /// 
    /// The build has the following default settings:
    /// - cull mode set to none,
    /// - front face winding set to counter-clockwise,
    /// - fragment state is empty,
    /// - depth stencil state is empty.
    pub fn new(v: wgpu::VertexState<'a>) -> Self {
        Self {
            rpd: RasterizerPipelineDescriptor {
                cull_mode: None,
                front_face: wgpu::FrontFace::Ccw,
                vertex: v,
                fragment: None,
                depthstencil: None
            }
        }
    }

    pub fn set_cull_mode(mut self, cm: Option<wgpu::Face>) -> Self {
        self.rpd.cull_mode = cm;
        self
    }

    pub fn set_front_face(mut self, ff: wgpu::FrontFace) -> Self {
        self.rpd.front_face = ff;
        self
    }

    pub fn set_fragment_state(mut self, f: wgpu::FragmentState<'a>) -> Self {
        self.rpd.fragment = Some(f);
        self
    }

    pub fn set_depth_stencil_state(mut self, ds: wgpu::DepthStencilState) -> Self {
        self.rpd.depthstencil = Some(ds);
        self
    }

    pub fn build(self) -> RasterizerPipelineDescriptor<'a> {
        self.rpd
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

/// A thin wrapper around `wgpu::RenderPipeline` that simplifies pipeline creation.
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
