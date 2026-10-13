use std::{collections::HashMap, sync::Arc};

use crate::{
    asset::{
        asset_manager::AssetManager,
        asset_types::{
            AssetData,
            GpuAssetType,
            static_mesh_asset::{StaticMeshAsset, StaticMeshMaterial},
        },
    },
    renderer::{
        device_interface::DeviceInterface,
        mesh::static_mesh::StaticMesh,
        pipeline::{
            bindless_resource_manager::BindlessResourceManager,
            pbr_material::PBRMaterial,
        },
    },
};

/// Owns the GPU-resident resources uploaded from the asset database.
///
/// This is the single seam between the `asset` layer and the renderer's GPU
/// resources. It maps asset names to bindless texture/sampler indices
/// (uploading each referenced texture at most once) and to uploaded
/// `StaticMesh`es (shared across instances), so mesh code no longer reaches
/// into `AssetManager` / `BindlessResourceManager` directly.
pub struct GpuAssetRegistry {
    bindless: BindlessResourceManager,
    /// Asset name -> (texture index, sampler index) in the bindless manager.
    texture_bindings: HashMap<String, (usize, usize)>,
    /// Mesh subasset name -> uploaded, material-resolved mesh.
    mesh_cache: HashMap<String, Arc<StaticMesh>>,
}

impl GpuAssetRegistry {
    pub fn new(di: &DeviceInterface) -> Self {
        Self {
            bindless: BindlessResourceManager::new(di),
            texture_bindings: HashMap::new(),
            mesh_cache: HashMap::new(),
        }
    }

    pub fn get_bindless_resource_manager(&self) -> &BindlessResourceManager { &self.bindless }
    pub fn get_bindless_resource_manager_mut(&mut self) -> &mut BindlessResourceManager { &mut self.bindless }

    /// Resolve a texture asset (by name) to its bindless (texture, sampler)
    /// indices, uploading it at most once.
    fn texture_binding(&mut self, di: &DeviceInterface, assets: &AssetManager, name: &str) -> (usize, usize) {
        if let Some(binding) = self.texture_bindings.get(name) {
            return *binding;
        }

        let asset = assets.get_asset(name).unwrap_or_else(|| panic!("texture asset '{name}' not found."));
        let asset = asset.read().expect("failed to acquire read lock for texture asset.");
        let texture = match asset.get_data() {
            AssetData::TextureAssetType(t) => t,
            _ => panic!("asset '{name}' is not a texture asset."),
        };

        let uploaded = texture.upload(name, di.get_device(), di.get_queue());
        let tx = self.bindless.push_texture(uploaded.create_view(&Default::default()))
            .expect("failed to push texture to bindless manager.");
        let sp = self.bindless.push_sampler(di.get_device(), texture.sampler.clone().into())
            .expect("failed to push sampler to bindless manager.");

        let binding = (tx, sp);
        self.texture_bindings.insert(name.to_string(), binding);
        binding
    }

    /// Build a `PBRMaterial` from a static mesh asset's material description.
    ///
    /// Textures are resolved by name through `assets`; missing slots fall back
    /// to the bindless manager's default textures.
    pub(crate) fn material_from_asset(
        &mut self,
        di: &DeviceInterface,
        assets: &AssetManager,
        material: &StaticMeshMaterial
    ) -> PBRMaterial {
        let (diffuse_tx, diffuse_sp) = match &material.diffuse_texture_name {
            Some(name) => self.texture_binding(di, assets, name),
            None => (self.bindless.get_white_texture(), self.bindless.get_default_sampler()),
        };

        let (normal_tx, normal_sp) = match &material.normal_texture_name {
            Some(name) => self.texture_binding(di, assets, name),
            None => (self.bindless.get_default_bump_texture(), self.bindless.get_default_sampler()),
        };

        let (mrao_tx, mrao_sp) = match &material.mrao_texture_name {
            Some(name) => self.texture_binding(di, assets, name),
            None => (self.bindless.get_default_mrao_texture(), self.bindless.get_default_sampler()),
        };

        PBRMaterial::new(diffuse_tx, normal_tx, mrao_tx, diffuse_sp, normal_sp, mrao_sp)
    }

    /// Upload a static mesh asset to the GPU and resolve its material.
    ///
    /// The geometry is uploaded once per asset name and shared across instances
    /// through an `Arc`.
    pub(crate) fn static_mesh_from_asset(
        &mut self,
        di: &DeviceInterface,
        assets: &AssetManager,
        name: &str,
        mesh: &StaticMeshAsset
    ) -> Arc<StaticMesh> {
        if let Some(cached) = self.mesh_cache.get(name) {
            return cached.clone();
        }

        let geometry = mesh.upload(name, di.get_device(), di.get_queue());
        let material = self.material_from_asset(di, assets, &mesh.material);
        let uploaded = Arc::new(StaticMesh::new(geometry, material));
        self.mesh_cache.insert(name.to_string(), uploaded.clone());
        uploaded
    }
}
