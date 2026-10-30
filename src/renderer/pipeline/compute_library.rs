use std::{collections::HashMap, default};

use wgpu::{ComputePipeline, PipelineLayout};

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

/// Pipeline library for compute shaders.
/// 
/// Compute shaders are much simpler than graphics pipelines and should be
/// managed separately.
#[derive(Debug)]
pub struct ComputePipelineLibrary {
    cache: HashMap<ComputePipelineKey, (PipelineLayout, ComputePipeline)>
}

impl ComputePipelineLibrary {
    pub fn new(device: &wgpu::Device, bindless_layout: &wgpu::BindGroupLayout) -> Self {
        let mut cache = HashMap::new();

        cache.insert(ComputePipelineKey::GpuSkinning, GpuSkinningComputeRecipe::new(device).build_pipeline_and_layout(device, bindless_layout));

        Self { cache }
    }

    /// Get the compute pipeline.
    /// 
    /// TODO: add a `get_or_create` variant.
    pub fn get(&self, k: ComputePipelineKey) -> Option<&ComputePipeline> {
        self.cache.get(&k).map(|x| &x.1)
    }
}
