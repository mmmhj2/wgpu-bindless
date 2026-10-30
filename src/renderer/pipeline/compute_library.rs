use std::collections::HashMap;

use wgpu::{BindGroupLayout, ComputePipeline, PipelineLayout};

use crate::renderer::pipeline::compute_recipe::{ComputeRecipe, GpuSkinningComputeRecipe};


/// Key for compute pipelines.
/// 
/// Contrary to the rasterizer pipeline, compute pipeline keys corresponds to
/// both pipeline itself and its layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ComputePipelineKey {
    GpuSkinning
}

/// One compiled compute pipeline plus the layouts it was built with.
#[derive(Debug)]
struct ComputeEntry {
    layout: PipelineLayout,
    pipeline: ComputePipeline,
    /// Bind group layouts appended after the global bindless group.
    ///
    /// `bind_group_layouts[0]` is pipeline layout group 1, and so on.
    extra_bind_group_layouts: Vec<BindGroupLayout>,
}

/// Pipeline library for compute shaders.
/// 
/// Compute shaders are much simpler than graphics pipelines and should be
/// managed separately.
#[derive(Debug)]
pub struct ComputePipelineLibrary {
    cache: HashMap<ComputePipelineKey, ComputeEntry>
}

impl ComputePipelineLibrary {
    pub fn new(device: &wgpu::Device, bindless_layout: &wgpu::BindGroupLayout) -> Self {
        let mut cache = HashMap::new();

        let recipe = GpuSkinningComputeRecipe::new(device);
        let extra_bind_group_layouts = vec![recipe.bind_group_layout().clone()];
        let (layout, pipeline) = recipe.build_pipeline_and_layout(device, bindless_layout);
        cache.insert(
            ComputePipelineKey::GpuSkinning,
            ComputeEntry { layout, pipeline, extra_bind_group_layouts },
        );

        Self { cache }
    }

    /// Get the compute pipeline.
    /// 
    /// TODO: add a `get_or_create` variant.
    pub fn get_pipeline(&self, k: ComputePipelineKey) -> Option<&ComputePipeline> {
        self.cache.get(&k).map(|x| &x.pipeline)
    }
    pub fn get_layout(&self, k: ComputePipelineKey) -> Option<&PipelineLayout> {
        self.cache.get(&k).map(|x| &x.layout)
    }

    /// Layout of the per-dispatch bind group, i.e. pipeline layout group 1.
    ///
    /// Needed by callers that build the per-object bind group themselves, such
    /// as the skinned mesh skinning dispatch.
    pub fn get_dispatch_bind_group_layout(&self, k: ComputePipelineKey) -> Option<&BindGroupLayout> {
        self.cache.get(&k).and_then(|entry| entry.extra_bind_group_layouts.first())
    }
}
