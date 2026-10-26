use std::collections::HashMap;

use crate::renderer::pipeline::{
    builtin::{CookTorranceRecipe, DefaultRecipe},
    key::{PipelineKey, PipelineLayoutKey, RenderPipelineKey, VertexLayoutKey},
    rasterizer_pipeline::RasterizerPipeline,
    recipe::PipelineRecipe,
    vertex_layout::{VertexLayout, VertexLayoutRegistry},
};

/// Bind-group layouts shared by the scene pipeline layouts.
pub struct SceneBindGroupLayouts<'a> {
    pub bindless: &'a wgpu::BindGroupLayout,
    pub camera: &'a wgpu::BindGroupLayout,
    pub lighting: &'a wgpu::BindGroupLayout,
    pub mesh: &'a wgpu::BindGroupLayout,
}

pub struct PipelineLibrary {
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    layouts: HashMap<PipelineLayoutKey, wgpu::PipelineLayout>,
    vertex_layouts: VertexLayoutRegistry,
    recipes: HashMap<PipelineKey, Box<dyn PipelineRecipe>>,
    pipelines: HashMap<RenderPipelineKey, RasterizerPipeline>,
}

impl PipelineLibrary {
    /// Create the library and register the built-in scene programs.
    pub fn build_default(
        device: &wgpu::Device,
        bgls: &SceneBindGroupLayouts<'_>,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let default_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Default scene pipeline layout"),
            bind_group_layouts: &[
                Some(bgls.bindless),
                Some(bgls.camera),
                Some(bgls.lighting),
                Some(bgls.mesh),
            ],
            immediate_size: (std::mem::size_of::<u32>() * 3) as u32,
        });

        let mut layouts = HashMap::new();
        layouts.insert(PipelineLayoutKey::Default, default_layout);

        let mut recipes: HashMap<PipelineKey, Box<dyn PipelineRecipe>> = HashMap::new();
        let default_recipe: Box<dyn PipelineRecipe> = Box::new(DefaultRecipe);
        let pbr_recipe: Box<dyn PipelineRecipe> = Box::new(CookTorranceRecipe);
        recipes.insert(default_recipe.key(), default_recipe);
        recipes.insert(pbr_recipe.key(), pbr_recipe);

        let mut library = Self {
            color_format,
            depth_format,
            layouts,
            vertex_layouts: VertexLayoutRegistry::new(),
            recipes,
            pipelines: HashMap::new(),
        };

        library.get_or_create(device, PipelineKey::Default, VertexLayoutKey::BASIC);
        library.get_or_create(device, PipelineKey::CookTorrancePbr, VertexLayoutKey::BASIC);
        library
    }

    pub fn color_format(&self) -> wgpu::TextureFormat { self.color_format }
    pub fn depth_format(&self) -> wgpu::TextureFormat { self.depth_format }

    /// Register an additional pipeline layout, for example the skybox layout.
    pub fn register_layout(&mut self, key: PipelineLayoutKey, layout: wgpu::PipelineLayout) {
        self.layouts.insert(key, layout);
    }

    pub fn register_vertex_layout(&mut self, key: VertexLayoutKey, layout: VertexLayout) {
        self.vertex_layouts.register(key, layout);
    }

    /// Insert an externally-built pipeline.
    ///
    /// Passes such as the skybox own a custom bind group layout and build their
    /// pipeline during setup. Registering it here keeps pipeline ownership in
    /// one place.
    pub fn insert_pipeline(
        &mut self,
        program: PipelineKey,
        vertex_layout: VertexLayoutKey,
        layout: PipelineLayoutKey,
        pipeline: RasterizerPipeline,
    ) {
        if let Some(_) = self.pipelines.insert(
            RenderPipelineKey { program, vertex_layout, layout },
            pipeline,
        ) {
            log::warn!("An externally registered pipeline overwrites another pipeline: {:?} {:?} {:?}", program, vertex_layout, layout)
        };
    }

    /// Register a recipe that allows lazy creation of a pipeline.
    /// 
    /// See [`PipelineRecipe`] for customization points of the recipe.
    pub fn register_recipe(&mut self, recipe: Box<dyn PipelineRecipe>) {
        let key = recipe.key();
        if let Some(_) = self.recipes.insert(key, recipe) {
            log::warn!("A new pipeline recipe overwrites an older one: {:?}", key)
        }
    }

    /// Get a pipeline, creating it on demand from its recipe and vertex layout.
    /// 
    /// Panic
    /// ---
    /// If the given program has not been associated with a recipe, the method
    /// panicks.
    pub fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        program: PipelineKey,
        vertex_layout: VertexLayoutKey,
    ) -> &RasterizerPipeline {
        let recipe = self
            .recipes
            .get(&program)
            .unwrap_or_else(|| panic!("pipeline program {:?} is not registered", program));
        let key = RenderPipelineKey {
            program,
            vertex_layout,
            layout: recipe.layout_key(),
        };

        if !self.pipelines.contains_key(&key) {
            let layout = self
                .layouts
                .get(&key.layout)
                .unwrap_or_else(|| panic!("pipeline layout {:?} is not registered", key.layout));
            let vertex = self.vertex_layouts.get(vertex_layout);
            let pipeline = self
                .recipes
                .get(&program)
                .expect("recipe disappeared")
                .create(
                    device,
                    layout,
                    self.color_format,
                    self.depth_format,
                    vertex.buffers(),
                );
            self.pipelines.insert(key, pipeline);
        }

        self.pipelines.get(&key).expect("pipeline was just inserted")
    }

    /// Query an already-created pipeline without triggering compilation.
    pub fn get(
        &self,
        program: PipelineKey,
        vertex_layout: VertexLayoutKey,
        layout: PipelineLayoutKey,
    ) -> Option<&RasterizerPipeline> {
        self.pipelines.get(&RenderPipelineKey { program, vertex_layout, layout })
    }

    pub fn layout(&self, key: PipelineLayoutKey) -> Option<&wgpu::PipelineLayout> {
        self.layouts.get(&key)
    }
}
