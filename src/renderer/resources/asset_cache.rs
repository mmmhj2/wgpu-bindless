use std::{collections::HashMap, sync::Arc};

use crate::{
    asset::{
        asset_manager::AssetManager,
        asset_types::{
            AssetData,
            GpuAssetType,
            static_mesh_asset::{StaticMeshAsset, StaticMeshMaterial},
        },
    }, renderer::{
        drawable::static_mesh::StaticMesh, pipeline::{compute_library::ComputePipelineLibrary, pbr_material::PBRMaterial}, resources::{
            bindless::{BindlessResources, SampledTextureHandle, SamplerHandle},
            mesh_layout::MeshLayout,
        },
    },
};

/// Owns the GPU-resident resources uploaded from the asset database.
///
/// The cache itself does not own the bindless tables.
/// Instead, they are only modified through upload methods, which borrow
/// them explicitly through [`MeshUploadContext`] or an orchestrator helper.
pub struct GpuAssetCache {
    /// Asset name -> (texture handle, sampler handle) in the bindless table.
    texture_bindings: HashMap<String, (SampledTextureHandle, SamplerHandle)>,
    /// Mesh subasset name -> uploaded, material-resolved mesh.
    mesh_cache: HashMap<String, Arc<StaticMesh>>,
}

impl GpuAssetCache {
    pub fn new() -> Self {
        Self {
            texture_bindings: HashMap::new(),
            mesh_cache: HashMap::new(),
        }
    }
}

impl Default for GpuAssetCache {
    fn default() -> Self { Self::new() }
}

/// Borrowed handles needed to upload mesh and texture assets to the GPU.
pub struct MeshUploadContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub bindless: &'a mut BindlessResources,
    pub mesh_layout: &'a MeshLayout,
    // Upload context requiring compute library is not a good design.
    // Figure out some other way to pass compute pipeline layout to asset uploading (i.e. `OnceCell`` for mesh bind groups?)
    pub compute_pipeline: &'a ComputePipelineLibrary,
    pub cache: &'a mut GpuAssetCache,
    pub assets: &'a AssetManager,
}

impl MeshUploadContext<'_> {
    /// Resolve a texture asset (by name) to its bindless texture/sampler
    /// handles, uploading it at most once.
    fn texture_binding(&mut self, name: &str) -> (SampledTextureHandle, SamplerHandle) {
        if let Some(binding) = self.cache.texture_bindings.get(name) {
            return *binding;
        }

        let asset = self.assets.get_asset(name).unwrap_or_else(|| panic!("texture asset '{name}' not found."));
        let asset = asset.read().expect("failed to acquire read lock for texture asset.");
        let texture = match asset.get_data() {
            AssetData::TextureAssetType(t) => t,
            _ => panic!("asset '{name}' is not a texture asset."),
        };

        let uploaded = texture.upload(name, self.device, self.queue);
        let tx = self.bindless
            .create_sampled_texture(uploaded.create_view(&Default::default()))
            .expect("failed to push sampled texture to bindless resources.");
        let sp = self.bindless
            .create_sampler(self.device, texture.sampler.clone().into())
            .expect("failed to push sampler to bindless resources.");

        let binding = (tx, sp);
        self.cache.texture_bindings.insert(name.to_string(), binding);
        binding
    }

    /// Build a `PBRMaterial` from a static mesh asset's material description.
    fn material_from_asset(&mut self, material: &StaticMeshMaterial) -> PBRMaterial {
        let default_sampler = self.bindless.get_default_sampler();

        let (diffuse_tx, diffuse_sp) = match &material.diffuse_texture_name {
            Some(name) => self.texture_binding(name),
            None => (self.bindless.get_white_texture(), default_sampler),
        };

        let (normal_tx, normal_sp) = match &material.normal_texture_name {
            Some(name) => self.texture_binding(name),
            None => (self.bindless.get_default_bump_texture(), default_sampler),
        };

        let (mrao_tx, mrao_sp) = match &material.mrao_texture_name {
            Some(name) => self.texture_binding(name),
            None => (self.bindless.get_default_mrao_texture(), default_sampler),
        };

        PBRMaterial::new(
            diffuse_tx,
            normal_tx,
            mrao_tx,
            diffuse_sp,
            normal_sp,
            mrao_sp,
        )
    }

    /// Upload a static mesh asset to the GPU and resolve its material.
    ///
    /// The geometry is uploaded once per asset name and shared across instances
    /// through an `Arc`.
    pub(crate) fn static_mesh_from_asset(&mut self, name: &str, mesh: &StaticMeshAsset) -> Arc<StaticMesh> {
        if let Some(cached) = self.cache.mesh_cache.get(name) {
            return cached.clone();
        }

        let geometry = mesh.upload(name, self.device, self.queue);
        let material = self.material_from_asset(&mesh.material);
        let uploaded = Arc::new(StaticMesh::new(geometry, material));
        self.cache.mesh_cache.insert(name.to_string(), uploaded.clone());
        uploaded
    }
}
