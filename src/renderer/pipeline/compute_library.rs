use std::{collections::HashMap, ops::Index};

use wgpu::{BindGroupLayout, ComputePipeline, PipelineLayout};

use crate::renderer::pipeline::compute_recipe::{
    ComputeRecipe, GpuSkinningComputeRecipe, GpuVertexMorphingRecipe,
};


/// Key for compute pipelines.
/// 
/// Contrary to the rasterizer pipeline, compute pipeline keys corresponds to
/// both pipeline itself and its layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ComputePipelineKey {
    GpuSkinning,
    GpuVertexMorphing
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

        let skinning = GpuSkinningComputeRecipe::new(device);
        let skinning_bgl = skinning.bind_group_layout().clone();
        let extra_bind_group_layouts = vec![skinning_bgl.clone()];
        let (layout, pipeline) = skinning.build_pipeline_and_layout(device, bindless_layout);
        cache.insert(
            ComputePipelineKey::GpuSkinning,
            ComputeEntry { layout, pipeline, extra_bind_group_layouts },
        );

        let morphing = GpuVertexMorphingRecipe::new(device, skinning_bgl);
        let extra_bind_group_layouts = vec![
            morphing.skinning_bind_group_layout().clone(),
            morphing.morph_bind_group_layout().clone(),
        ];
        let (layout, pipeline) = morphing.build_pipeline_and_layout(device, bindless_layout);
        cache.insert(
            ComputePipelineKey::GpuVertexMorphing,
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

    /// Get the layout of the extra bind groups.
    /// 
    /// This bind group is indexed from 1 (0 is reserved for bindless resources).
    /// So with `index = 0`, the very first extra bind group is returned.
    /// 
    /// Mesh paths should go through [`ComputeDispatchBindingSource`] instead,
    /// which keeps the layout private.
    pub fn get_dispatch_bind_group_layout(&self, k: ComputePipelineKey, index: usize) -> Option<&BindGroupLayout> {
        // try_index is still not available...
        self.cache.get(&k).and_then(|entry|
            if entry.extra_bind_group_layouts.len() <= index {
                None
            } else {
                Some(entry.extra_bind_group_layouts.index(index))
            })
    }
}

/// Source of per-dispatch (pipeline layout group 1) bind groups for compute
/// shaders.
///
/// The library owning the pipeline supplies the layout; the caller supplies the
/// buffer bindings, because only the caller knows what each entry means. Like
/// [`crate::renderer::resources::mesh_layout::ModelMatrixBindingSource`], this
/// is the seam a future binding cache plugs into without changing call sites.
pub trait ComputeDispatchBindingSource {
    /// Create a group-1 bind group for `key`'s pipeline from `entries`.
    fn create_dispatch_bind_group(
        &self,
        device: &wgpu::Device,
        key: ComputePipelineKey,
        label: &str,
        index: usize,
        entries: &[wgpu::BindGroupEntry<'_>],
    ) -> wgpu::BindGroup;
}

impl ComputeDispatchBindingSource for ComputePipelineLibrary {
    fn create_dispatch_bind_group(
        &self,
        device: &wgpu::Device,
        key: ComputePipelineKey,
        label: &str,
        index: usize,
        entries: &[wgpu::BindGroupEntry<'_>],
    ) -> wgpu::BindGroup {
        let layout = self
            .get_dispatch_bind_group_layout(key, index)
            .unwrap_or_else(|| panic!("compute pipeline {key:?} has no per-dispatch bind group layout {index}"));
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries,
        })
    }
}
