use crate::renderer::{
    pipeline::recipe::bindless_resources_shader_module,
    postfx::{
        bloom::BloomPass,
        build_context::{POSTFX_LINEAR_SAMPLER, PostFxBuildContext},
        postfx_stack::{PostFxContext, PostFxPass, PostFxResourceCache},
    },
    resources::bindless::SamplerHandle,
};

/// Push constants consumed by the tonemapping shader.
///
/// Must match `Immediates` in `shaders/tonemap.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TonemapImmediates {
    sampler_index: u32,
}

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
        }
    ]
};

fn build_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    output_format: wgpu::TextureFormat,
    fs_name: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(
        &wgpu::RenderPipelineDescriptor {
            label: Some("Tonemapping"),
            layout: Some(layout),
            vertex: wgpu::VertexState { module, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
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
                module,
                entry_point: Some(fs_name),
                compilation_options: Default::default(),
                targets: &[ Some(wgpu::ColorTargetState{ format: output_format, blend: None, write_mask: wgpu::ColorWrites::ALL }) ]
            }),
            multiview_mask: None,
            cache: None
        }
    )
}

fn generate_bind_group(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, color_in: &wgpu::TextureView, lum_in: &wgpu::TextureView) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
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
    /// Sampler registered in the global bindless sampler table.
    sampler: SamplerHandle,

    input_format: wgpu::TextureFormat,
    output_format: wgpu::TextureFormat
}

impl AcesTonemapPass {
    pub fn new(ctx: &mut PostFxBuildContext<'_>) -> Self {
        let device = ctx.device;
        let input_format = ctx.hdr_format();
        let output_format = ctx.output_format();

        let bgl_tonemap = device.create_bind_group_layout(&DBGL_TONEMAP);
        let layout = ctx.create_pipeline_layout(
            "Tonemapping pipeline layout",
            &bgl_tonemap,
            std::mem::size_of::<TonemapImmediates>() as u32,
        );
        let module = bindless_resources_shader_module(
            device,
            "Tonemapping shader",
            include_str!("./shaders/tonemap.wgsl"),
        );
        let sampler = ctx.create_sampler(POSTFX_LINEAR_SAMPLER);
        Self {
            bgl_tonemap,
            ppl: build_pipeline(device, &module, &layout, output_format, "fs_aces"),
            sampler,
            input_format,
            output_format,
        }
    }
}

trait TonemapPass: PostFxPass {
    fn name(&self) -> &'static str;
    fn input_format(&self) -> wgpu::TextureFormat;
    fn output_format(&self) -> wgpu::TextureFormat;

    fn requires_average_luminance() -> bool { false }
    fn get_tonemap_bind_group_layout(&self) -> &wgpu::BindGroupLayout;
    fn get_tonemap_pipeline(&self) -> &wgpu::RenderPipeline;
    fn get_tonemap_sampler(&self) -> SamplerHandle;
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
        let device = ctx.device;
        let input = ctx.input;
        let output = ctx.output;
        let bindless = ctx.bindless;

        let sampler = self.get_tonemap_sampler();
        
        let lum = if Self::requires_average_luminance() {
            let avg_lum = cache.try_get(BloomPass::BLOOM_MIPCHAIN_HIGHEST_NAME);
            match avg_lum {
                Some(view) => &view.clone(),
                None => panic!("Average luminance texture is not available!")
            }
        } else {
            input
        };
        let bind_group = generate_bind_group(device, self.get_tonemap_bind_group_layout(), input, lum);
        let immediates = TonemapImmediates {
            sampler_index: sampler.index() as u32,
        };

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
        rp.set_pipeline(self.get_tonemap_pipeline());
        rp.set_immediates(0, bytemuck::bytes_of(&immediates));
        rp.set_bind_group(0, bindless.bind_group, &[]);
        rp.set_bind_group(1, &bind_group, &[]);
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
    fn get_tonemap_sampler(&self) -> SamplerHandle {
        self.sampler
    }
}

#[cfg(test)]
mod tests {
    use super::TonemapImmediates;

    #[test]
    fn tonemap_immediates_match_wgsl_layout() {
        assert_eq!(std::mem::size_of::<TonemapImmediates>(), 4);
        assert_eq!(std::mem::offset_of!(TonemapImmediates, sampler_index), 0);

        let bytes = bytemuck::bytes_of(&TonemapImmediates { sampler_index: 7 });
        assert_eq!(bytes.len(), 4);
    }
}
