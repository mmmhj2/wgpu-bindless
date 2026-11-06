
use crate::renderer::pipeline::recipe::bindless_resources_shader_module;

pub(super) trait ComputeRecipe {
    const LABEL: &'static str;
    const SHADER_STRING: &'static str;
    const ENTRY_POINT: &'static str;
    const IMMEDIATE_SIZE: u32;

    fn build_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
        bindless_resources_shader_module(device, Self::LABEL, Self::SHADER_STRING)
    }

    fn build_pipeline_and_layout(
        &self,
        device: &wgpu::Device,
        bindless_layout: &wgpu::BindGroupLayout
    ) -> (wgpu::PipelineLayout, wgpu::ComputePipeline) {
        let mut bgl = vec![Some(bindless_layout)];
        bgl.extend_from_slice(&self.get_extra_bind_groups());
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(Self::LABEL),
            bind_group_layouts: &bgl,
            immediate_size: Self::IMMEDIATE_SIZE
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(Self::LABEL),
            layout: Some(&layout),
            module: &Self::build_shader_module(device),
            entry_point: Some(Self::ENTRY_POINT),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        (layout, pipeline)
    }

    /// Get extra bind groups other than the bindless bind group
    /// 
    /// The bindless bind group always occupies the zeroth set. These optional
    /// bind groups are appended to the bindless group.
    fn get_extra_bind_groups<'a>(&'a self) -> Vec<Option<&'a wgpu::BindGroupLayout>>;
}

pub(super) struct GpuSkinningComputeRecipe {
    bgl: wgpu::BindGroupLayout
}

impl GpuSkinningComputeRecipe {
    /// Predetermined bind group layouts for skinning compute shader.
    /// 
    /// Note: Texture with baked animation can use bindless slots (if 2D only, recommended)
    /// or go here (if is 2D array or 3D).
    const BGL_SKINNING: [wgpu::BindGroupLayoutEntry; 5] = [
        // Vertex position input
        wgpu::BindGroupLayoutEntry{
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None
            },
            count: None,
        },
        // Vertex attribute input
        wgpu::BindGroupLayoutEntry{
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None
            },
            count: None,
        },
        // Vertex position output
        wgpu::BindGroupLayoutEntry{
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None
            },
            count: None,
        },
        // Vertex attribute output
        wgpu::BindGroupLayoutEntry{
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None
            },
            count: None,
        },
        // Bone indices and weights
        wgpu::BindGroupLayoutEntry{
            binding: 4,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None
            },
            count: None,
        }
    ];

    pub fn new(device: &wgpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor { 
                label: None,
                entries: &Self::BGL_SKINNING
            }
        );

        Self { bgl }
    }

    /// Layout of the per-dispatch bind group (pipeline layout group 1).
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout { &self.bgl }
}

impl ComputeRecipe for GpuSkinningComputeRecipe {
    const LABEL: &'static str = "GPU Skinning";
    const SHADER_STRING: &'static str = include_str!("shaders/gpu_skinning.wgsl");
    const ENTRY_POINT: &'static str = "cs_main";
    const IMMEDIATE_SIZE: u32 = (std::mem::size_of::<u32>() * 2) as u32;

    fn get_extra_bind_groups(&self) -> Vec<Option<&wgpu::BindGroupLayout>> {
        vec![Some(&self.bgl)]
    }
}

pub(super) struct GpuVertexMorphingRecipe {
    /// Group 1: the same five storage buffers the skinning shader uses.
    skinning_bgl: wgpu::BindGroupLayout,
    /// Group 2: the morph layout table and the packed delta buffer.
    morph_bgl: wgpu::BindGroupLayout,
}

impl GpuVertexMorphingRecipe {
    /// Predetermined bind group layouts for the morphing compute shader.
    const BGL_MORPHING: [wgpu::BindGroupLayoutEntry; 2] = [
        // Data-driven layout table: channel count, masks and offsets.
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        // Packed attribute deltas for every channel.
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ];

    pub fn new(device: &wgpu::Device, skinning_bgl: wgpu::BindGroupLayout) -> Self {
        let morph_bgl = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GPU Vertex morphing bind group"),
                entries: &Self::BGL_MORPHING,
            }
        );

        Self { skinning_bgl, morph_bgl }
    }

    /// Group 1 layout, shared with the skinning pipeline.
    pub fn skinning_bind_group_layout(&self) -> &wgpu::BindGroupLayout { &self.skinning_bgl }

    /// Group 2 layout, owned by the morphing pipeline.
    pub fn morph_bind_group_layout(&self) -> &wgpu::BindGroupLayout { &self.morph_bgl }
}

impl ComputeRecipe for GpuVertexMorphingRecipe {
    const LABEL: &'static str = "GPU Vertex morphing";
    const SHADER_STRING: &'static str = include_str!("shaders/gpu_vertex_morph.wgsl");
    const ENTRY_POINT: &'static str = "cs_main";
    /// `{ tex: u32, frame: f32, channels: u32 }`.
    const IMMEDIATE_SIZE: u32 = (std::mem::size_of::<u32>() * 3) as u32;

    fn get_extra_bind_groups<'a>(&'a self) -> Vec<Option<&'a wgpu::BindGroupLayout>> {
        vec![Some(&self.skinning_bgl), Some(&self.morph_bgl)]
    }
}
