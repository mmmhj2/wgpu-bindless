use crate::renderer::{
    postfx::postfx_stack::{PostFxPass, PostFxStack},
    resources::{
        bindless::{BindlessResources, SamplerHandle},
        sampler::SamplerDescription,
    },
};

/// Sampler state shared by the built-in post-processing passes.
///
/// Registering the same description twice returns the same bindless handle, so
/// passes that need identical filtering share one GPU sampler.
pub(crate) const POSTFX_LINEAR_SAMPLER: SamplerDescription = SamplerDescription {
    address_mode: [wgpu::AddressMode::ClampToEdge; 3],
    mag_filter: wgpu::FilterMode::Linear,
    min_filter: wgpu::FilterMode::Linear,
    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
    // For some big brain reason rust decide to not support a const default trait...
    compare: None,
    anisotropy_clamp: 1,
    border_color: None,
};

/// Borrowed context used to construct and register post-processing passes.
///
/// This bundles everything a pass constructor needs at setup time, replacing
/// ad-hoc tuple-returning split-borrow helpers on the orchestrator:
///
/// * the `wgpu::Device` and `wgpu::Queue` owned by the renderer;
/// * the global bindless table, where the pass registers its sampler;
/// * the [`PostFxStack`] being populated.
///
/// Samplers registered through [`PostFxBuildContext::create_sampler`] persist
/// for as long as the stack lives. Passes store the returned [`SamplerHandle`]
/// and write its index into their immediates when recording.
///
/// Post-processing pipeline layouts must reserve group 0 for the bindless table
/// and group 1 for pass-local inputs. Use
/// [`PostFxBuildContext::create_pipeline_layout`] to get that ordering.
pub struct PostFxBuildContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub bindless: &'a mut BindlessResources,
    stack: &'a mut PostFxStack,
}

impl<'a> PostFxBuildContext<'a> {
    /// Build a context for a custom stack.
    ///
    /// Most applications obtain this from
    /// [`RenderOrchestrator::postfx_build_context`] instead.
    ///
    /// [`RenderOrchestrator::postfx_build_context`]: crate::renderer::orchestrator::RenderOrchestrator::postfx_build_context
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        bindless: &'a mut BindlessResources,
        stack: &'a mut PostFxStack,
    ) -> Self {
        Self {
            device,
            queue,
            bindless,
            stack,
        }
    }

    /// The HDR format the scene is rendered into and most passes operate on.
    pub fn hdr_format(&self) -> wgpu::TextureFormat { self.stack.get_hdr_format() }

    /// The format the final pass must write to (usually the swapchain format).
    pub fn output_format(&self) -> wgpu::TextureFormat { self.stack.get_output_format() }

    /// The bind group layout of the global bindless table (group 0).
    pub fn bindless_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        self.bindless.get_bind_group_layout()
    }

    /// Register a sampler in the global bindless table.
    ///
    /// Identical descriptions return the same handle. Panics when the sampler
    /// table is full, which is a hard capacity error during renderer setup.
    pub fn create_sampler(&mut self, desc: SamplerDescription) -> SamplerHandle {
        self.bindless
            .create_sampler(self.device, desc)
            .expect("bindless sampler table is full")
    }

    /// Create a pipeline layout for a post-processing pass.
    ///
    /// The resulting layout is `[group 0 = bindless, group 1 = pass_layout]`.
    /// `immediate_size` must match the pass shader's `var<immediate>` block.
    pub fn create_pipeline_layout(
        &self,
        label: &'static str,
        pass_layout: &wgpu::BindGroupLayout,
        immediate_size: u32,
    ) -> wgpu::PipelineLayout {
        self.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[
                    Some(self.bindless_bind_group_layout()),
                    Some(pass_layout),
                ],
                immediate_size,
            })
    }

    /// Append a pass to the stack under construction.
    pub fn push_pass<P: PostFxPass + 'static>(&mut self, pass: P) {
        self.stack.push_pass(Box::new(pass));
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use crate::renderer::{
        context::gpu_context::GpuContext,
        postfx::{
            bloom::BloomPass,
            postfx_stack::{BindlessFrame, PostFxExtraInputs},
            tonemap::AcesTonemapPass,
        },
    };

    /// Smoke test for the whole build and execution path: device, bindless
    /// sampler table, WGSL compilation, pipeline layouts, bind groups,
    /// immediates and one submitted frame. Ignored by default because it needs
    /// a Vulkan-capable GPU.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn postfx_stack_builds_and_runs() {
        let mut gpu = GpuContext::new();
        pollster::block_on(gpu.create_device(None)).expect("failed to create Vulkan device");

        let device = gpu.device();
        let queue = gpu.queue();
        let mut bindless = BindlessResources::new(device, queue);
        let sampler_count_before = bindless.sampler_count();
        let mut stack = PostFxStack::new(
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Bgra8UnormSrgb,
        );

        {
            let mut ctx = PostFxBuildContext::new(device, queue, &mut bindless, &mut stack);
            let bloom = BloomPass::new(&mut ctx);
            ctx.push_pass(bloom);
            let tonemap = AcesTonemapPass::new(&mut ctx);
            ctx.push_pass(tonemap);
        }

        // Both passes use the same description, so only one new sampler is
        // registered in the global table.
        assert_eq!(bindless.sampler_count(), sampler_count_before + 1);

        let size = 64;
        let size = (NonZeroU32::new(size).unwrap(), NonZeroU32::new(size).unwrap());
        stack.configure_with_size(device, size);

        let final_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PostFX smoke test final target"),
            size: wgpu::Extent3d {
                width: size.0.get(),
                height: size.1.get(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let final_view = final_texture.create_view(&Default::default());

        let bindless_bind_group = bindless.get_bind_group(device);
        let bindless_frame = BindlessFrame {
            resources: &bindless,
            bind_group: &bindless_bind_group,
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("PostFX smoke test encoder"),
        });
        let inputs = PostFxExtraInputs::new();
        stack.run(device, &mut encoder, &final_view, &inputs, &bindless_frame);
        queue.submit(std::iter::once(encoder.finish()));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll failed");
    }
}
