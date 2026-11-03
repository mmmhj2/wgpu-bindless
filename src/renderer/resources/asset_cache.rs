use std::{collections::HashMap, sync::Arc};

use crate::{
    asset::{
        asset_manager::AssetManager,
        asset_types::{
            Asset,
            AssetData,
            AssetKind,
            scene_asset::SceneAsset,
            skinned_mesh_asset::SkinnedMeshAsset,
            static_mesh_asset::{StaticMeshAsset, StaticMeshMaterial},
        },
    },
    renderer::{
        pipeline::pbr_material::PBRMaterial,
        drawable::{
            draw_list::DrawList,
            skinned_mesh::{SkinnedAnimation, SkinnedMesh},
            static_mesh::StaticMesh,
            static_mesh_batch::StaticMeshBatch,
        },
        resources::bindless::SampledTextureHandle,
        scene::graph::SceneGraph,
        upload::{
            context::UploadContext,
            error::UploadError,
            gpu_mesh::GpuSkinnedMesh,
            gpu_type::GpuAssetType,
            texture::TextureBinding,
        },
    },
};

pub struct AnimationTable(HashMap<String, HashMap<String, (u32, f32)>>);

/// Owns every GPU-resident resource uploaded from the asset database.
///
/// This is the single cache between the asset database and the drawables. An
/// asset is uploaded at most once per name, and the result is shared with every
/// instance that references it. Uploads only happen through
/// [`Self::prepare_scene`] (or the individual accessors), which need an explicit
/// [`SceneUploadContext`] and should be unreachable from the frame loop.
pub struct GpuAssetStore {
    /// Texture asset name -> uploaded texture + its bindless slots.
    textures: HashMap<String, TextureBinding>,
    /// Static mesh subasset name -> uploaded, material-resolved mesh.
    static_meshes: HashMap<String, Arc<StaticMesh>>,
    /// Skinned mesh subasset name -> uploaded rest pose + skinning attributes.
    ///
    /// Shared across instances: only the deformed vertex buffers are
    /// per-instance, the rest pose geometry is immutable.
    skinned_geometry: HashMap<String, Arc<GpuSkinnedMesh>>,
    /// Animation subasset name -> baked animation texture handle.
    skinned_animations: HashMap<String, SampledTextureHandle>,
}

impl GpuAssetStore {
    pub fn new() -> Self {
        Self {
            textures: HashMap::new(),
            static_meshes: HashMap::new(),
            skinned_geometry: HashMap::new(),
            skinned_animations: HashMap::new(),
        }
    }

    pub fn texture_count(&self) -> usize { self.textures.len() }
    pub fn static_mesh_count(&self) -> usize { self.static_meshes.len() }
    pub fn skinned_mesh_count(&self) -> usize { self.skinned_geometry.len() }
    pub fn animation_count(&self) -> usize { self.skinned_animations.len() }

    /// Resolve a texture asset to its uploaded binding, uploading it at most once.
    pub(crate) fn texture<'s>(
        &'s mut self,
        ctx: &mut SceneUploadContext<'_>,
        name: &str,
    ) -> Result<&'s TextureBinding, UploadError> {
        if !self.textures.contains_key(name) {
            let asset = ctx.asset(name)?;
            let asset = asset.read().map_err(|_| UploadError::AssetLockPoisoned(name.to_string()))?;
            let AssetData::TextureAssetType(texture) = asset.get_data() else {
                return Err(UploadError::WrongKind {
                    name: name.to_string(),
                    actual: asset.get_data().kind(),
                    expected: AssetKind::Texture,
                });
            };
            let binding = ctx.upload.upload_texture(name, texture)?;
            self.textures.insert(name.to_string(), binding);
        }

        Ok(self.textures.get(name).expect("texture was just inserted"))
    }

    /// Build a `PBRMaterial` from a mesh's material description.
    ///
    /// Missing texture slots fall back to the bindless placeholders, so an asset
    /// with an incomplete material still renders.
    fn material(
        &mut self,
        ctx: &mut SceneUploadContext<'_>,
        material: &StaticMeshMaterial,
    ) -> Result<PBRMaterial, UploadError> {
        let default_sampler = ctx.upload.bindless().get_default_sampler();

        let (diffuse_tx, diffuse_sp) = match &material.diffuse_texture_name {
            Some(name) => {
                let binding = self.texture(ctx, name)?;
                (binding.handle, binding.sampler)
            },
            None => (ctx.upload.bindless().get_white_texture(), default_sampler),
        };

        let (normal_tx, normal_sp) = match &material.normal_texture_name {
            Some(name) => {
                let binding = self.texture(ctx, name)?;
                (binding.handle, binding.sampler)
            },
            None => (ctx.upload.bindless().get_default_bump_texture(), default_sampler),
        };

        let (mrao_tx, mrao_sp) = match &material.mrao_texture_name {
            Some(name) => {
                let binding = self.texture(ctx, name)?;
                (binding.handle, binding.sampler)
            },
            None => (ctx.upload.bindless().get_default_mrao_texture(), default_sampler),
        };

        Ok(PBRMaterial::new(
            diffuse_tx,
            normal_tx,
            mrao_tx,
            diffuse_sp,
            normal_sp,
            mrao_sp,
        ))
    }

    /// Upload a static mesh asset, resolving its material. Cached by name.
    fn static_mesh(
        &mut self,
        ctx: &mut SceneUploadContext<'_>,
        name: &str,
        mesh: &StaticMeshAsset,
    ) -> Result<Arc<StaticMesh>, UploadError> {
        if let Some(cached) = self.static_meshes.get(name) {
            return Ok(cached.clone());
        }

        let geometry = mesh.upload(name, &mut ctx.upload)?;
        let material = self.material(ctx, &mesh.material)?;
        let uploaded = Arc::new(StaticMesh::new(geometry, material));
        self.static_meshes.insert(name.to_string(), uploaded.clone());
        Ok(uploaded)
    }

    /// Upload a skinned mesh's rest pose and skinning attributes. Cached by name.
    fn skinned_mesh(
        &mut self,
        ctx: &mut SceneUploadContext<'_>,
        name: &str,
        mesh: &SkinnedMeshAsset,
    ) -> Result<Arc<GpuSkinnedMesh>, UploadError> {
        if let Some(cached) = self.skinned_geometry.get(name) {
            return Ok(cached.clone());
        }

        let uploaded = Arc::new(mesh.upload(name, &mut ctx.upload)?);
        self.skinned_geometry.insert(name.to_string(), uploaded.clone());
        Ok(uploaded)
    }

    /// Upload a baked animation texture, if the named asset is one.
    fn animation_texture(
        &mut self,
        ctx: &mut SceneUploadContext<'_>,
        name: &str,
    ) -> Result<Option<SampledTextureHandle>, UploadError> {
        if let Some(cached) = self.skinned_animations.get(name) {
            return Ok(Some(*cached));
        }

        let asset = ctx.asset(name)?;
        let asset = asset.read().map_err(|_| UploadError::AssetLockPoisoned(name.to_string()))?;
        let AssetData::SkinnedAnimationAssetType(animation) = asset.get_data() else {
            return Ok(None);
        };

        let handle = ctx.upload.upload_texture(name, &animation.texture)?.handle;
        self.skinned_animations.insert(name.to_string(), handle);
        Ok(Some(handle))
    }

    /// Turn a scene asset into a draw list, uploading everything it references.
    ///
    /// Errors
    /// ---
    ///
    /// Any [`UploadError`] the individual assets produce are forwarded.
    /// 
    /// Panic
    /// ---
    /// Scene hierarchy validation failure panics directly, as documented in
    /// [`crate::renderer::scene::graph::SceneGraph::from_asset`].
    pub fn prepare_scene(
        &mut self,
        ctx: &mut SceneUploadContext<'_>,
        scene: &SceneAsset,
    ) -> Result<DrawList, UploadError> {
        let graph = SceneGraph::from_asset(scene);
        let plans = graph.plan_mesh_batches();

        // Resolve which baked animation drives each skin once, before any GPU
        // resource is created.
        let animations = animations_by_skin(ctx.assets);

        let mut static_batches = Vec::new();
        let mut static_instances = 0usize;
        let mut skinned_instances: Vec<(&str, [[f32; 4]; 4])> = Vec::new();

        for plan in &plans {
            match ctx.asset_kind(plan.mesh_subasset_name)? {
                AssetKind::SkinnedMesh => {
                    // Skinned meshes are never instanced.
                    for world_matrix in &plan.world_matrices {
                        skinned_instances.push((plan.mesh_subasset_name, *world_matrix));
                    }
                },
                AssetKind::StaticMesh => {
                    static_instances += plan.world_matrices.len();
                    static_batches.push(self.static_batch(ctx, plan.mesh_subasset_name, &plan.world_matrices)?);
                },
                actual => {
                    return Err(UploadError::WrongKind {
                        name: plan.mesh_subasset_name.to_string(),
                        actual,
                        expected: AssetKind::StaticMesh,
                    });
                },
            }
        }

        log::info!(
            "Prepared {} static mesh batches ({} instances) and {} skinned meshes from {} mesh instances",
            static_batches.len(),
            static_instances,
            skinned_instances.len(),
            graph.mesh_instance_count()
        );

        let mut skinned = Vec::with_capacity(skinned_instances.len());
        for (name, world_matrix) in skinned_instances {
            skinned.push(self.skinned_instance(ctx, name, world_matrix, &animations)?);
        }

        Ok(DrawList::new(static_batches, skinned))
    }

    /// Create one instanced draw covering `world_matrices` of a static mesh.
    fn static_batch(
        &mut self,
        ctx: &mut SceneUploadContext<'_>,
        name: &str,
        world_matrices: &[[[f32; 4]; 4]],
    ) -> Result<StaticMeshBatch, UploadError> {
        let asset = ctx.asset(name)?;
        let asset = asset.read().map_err(|_| UploadError::AssetLockPoisoned(name.to_string()))?;
        let AssetData::StaticMeshAssetType(mesh) = asset.get_data() else {
            return Err(UploadError::WrongKind {
                name: name.to_string(),
                actual: asset.get_data().kind(),
                expected: AssetKind::StaticMesh,
            });
        };

        let mesh = self.static_mesh(ctx, name, mesh)?;
        StaticMeshBatch::new(
            ctx.upload.device(),
            ctx.upload.queue(),
            mesh,
            world_matrices,
        )
    }

    /// Create one skinned mesh instance, with its baked animation if it has one.
    /// 
    /// For skins with multiple animations, an indeterminate one is uploaded
    /// and bound to the mesh.
    /// 
    /// TODO: Address the above problem.
    fn skinned_instance(
        &mut self,
        ctx: &mut SceneUploadContext<'_>,
        name: &str,
        world_matrix: [[f32; 4]; 4],
        animations: &AnimationTable,
    ) -> Result<SkinnedMesh, UploadError> {
        let asset = ctx.asset(name)?;
        let asset = asset.read().map_err(|_| UploadError::AssetLockPoisoned(name.to_string()))?;
        let AssetData::SkinnedMeshAssetType(mesh) = asset.get_data() else {
            return Err(UploadError::WrongKind {
                name: name.to_string(),
                actual: asset.get_data().kind(),
                expected: AssetKind::SkinnedMesh,
            });
        };

        let geometry = self.skinned_mesh(ctx, name, mesh)?;
        let material = self.material(ctx, &mesh.restpose.material)?;

        let animation = match animations.0.get(&mesh.skin) {
            Some(table) => {
                let (animation_name, (frames, fps)) = table.iter().next().expect("animation table should have at least one entry");
                // TODO: upload all animations, and select on run time.
                self
                .animation_texture(ctx, animation_name)?
                .map(|texture| SkinnedAnimation { texture, frames: *frames, fps: *fps })
            },
            None => {
                log::warn!("Mesh {name} does not have animations attached to it.");
                None
            },
        };

        SkinnedMesh::new(
            ctx.upload.device(),
            ctx.upload.queue(),
            geometry,
            material,
            world_matrix,
            animation,
        )
    }
}

impl Default for GpuAssetStore {
    fn default() -> Self { Self::new() }
}

/// Map every skin asset name to the animation that drives it.
fn animations_by_skin(assets: &AssetManager) -> AnimationTable {
    let database = assets.get_database();
    let mut candidates: Vec<(String, String, u32, f32)> = Vec::new();

    for (name, asset) in database.iter() {
        let Ok(asset) = asset.read() else { continue };
        let AssetData::SkinnedAnimationAssetType(animation) = asset.get_data() else { continue };
        candidates.push((animation.skin.clone(), name.clone(), animation.frames, animation.fps));
    }

    let mut by_skin = HashMap::new();
    for (skin, name, frames, fps) in candidates {
        log::debug!("Found animation {name} for skin {skin} with {frames} frames at {fps} fps.");
        let table = by_skin.entry(skin.clone()).or_insert(HashMap::new());
        if let Some(_) = table.insert(name.clone(), (frames, fps)) {
            log::warn!("Replacing animation {name} for skin {skin}.");
        }
    }
    AnimationTable(by_skin)
}

/// Everything the upload phase needs besides the asset data itself.
///
/// Note: Bind groups are renderer resources, and are not created in asset
/// upload precedures.
pub struct SceneUploadContext<'a> {
    pub upload: UploadContext<'a>,
    pub assets: &'a AssetManager,
}

impl<'a> SceneUploadContext<'a> {
    pub fn new(
        upload: UploadContext<'a>,
        assets: &'a AssetManager,
    ) -> Self {
        Self {
            upload,
            assets,
        }
    }

    /// Look up an asset by name.
    fn asset(&self, name: &str) -> Result<Arc<std::sync::RwLock<Asset>>, UploadError> {
        self.assets
            .get_asset(name)
            .ok_or_else(|| UploadError::MissingAsset(name.to_string()))
    }

    /// Concrete kind of an asset in the database.
    fn asset_kind(&self, name: &str) -> Result<AssetKind, UploadError> {
        let asset = self.asset(name)?;
        let asset = asset.read().map_err(|_| UploadError::AssetLockPoisoned(name.to_string()))?;
        Ok(asset.get_data().kind())
    }
}
