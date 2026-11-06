//! Upload cache between the asset database and the drawables.
//!
//! Split out of the former `asset_cache.rs`: clip discovery lives in
//! [`super::clip_catalog`] and the stage gate in [`super::rig_plan`]. This
//! module owns the GPU-resident assets (`GpuAssetStore`), the scene upload
//! context, and the per-instance assembly in `prepare_scene`.

use std::{collections::HashMap, sync::Arc};

use crate::{
    asset::{
        asset_manager::AssetManager,
        asset_types::{
            Asset,
            AssetData,
            AssetKind,
            animation_asset::BakedTrack,
            scene_asset::SceneAsset,
            deformable_mesh_asset::DeformableMeshAsset,
            static_mesh_asset::{StaticMeshAsset, StaticMeshMaterial},
            texture_asset::TextureAsset,
        },
    },
    renderer::{
        animation::{AnimationClip, AnimationManager, RigId},
        drawable::{
            draw_list::DrawList,
            skinned_mesh::AnimatedMesh,
            static_mesh::StaticMesh,
            static_mesh_batch::StaticMeshBatch,
        },
        pipeline::pbr_material::PBRMaterial,
        resources::{
            bindless::SampledTextureHandle,
            clip_catalog::ClipCatalog,
            rig_plan::{RigSpec, RigStages},
        },
        scene::graph::SceneGraph,
        upload::{
            context::UploadContext,
            error::UploadError,
            gpu_mesh::{GpuMorphTargets, GpuSkinnedMesh},
            gpu_type::GpuAssetType,
            texture::TextureBinding,
        },
    },
};

/// Owns every GPU-resident resource uploaded from the asset database.
///
/// This is the single cache between the asset database and the drawables. An
/// asset is uploaded at most once per name, and the result is shared with every
/// instance that references it. Uploads only happen through
/// [`Self::prepare_scene`] (or the individual accessors), which need an explicit
/// [`SceneUploadContext`] and should be unreachable from the frame loop.
///
/// TODO: every map here is keyed by `String`, forcing clones on each lookup.
/// We should intern asset names into ids (see the TODO in `asset_manager.rs`).
/// TODO: the asset -> read -> kind-check boilerplate repeats in `texture`,
/// `upload_clip_tracks`, `upload_morph_target`, `static_batch` and
/// `skinned_instance`. Factor it into a generic `asset_of_kind` helper.
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
    /// `(animation asset name, track index)` -> baked joint-matrix texture.
    ///
    /// Keyed per track, so a clip shared by several skinned instances occupies
    /// one bindless slot per track instead of one per instance.
    joint_textures: HashMap<(String, usize), SampledTextureHandle>,
    /// Morph target subasset name -> uploaded layout + deltas + base weights.
    ///
    /// Shared across instances, like the rest pose: only the weights vary.
    morph_targets: HashMap<String, Arc<GpuMorphTargets>>,
    /// `(animation asset name, track index)` -> baked morph weight texture.
    weight_textures: HashMap<(String, usize), SampledTextureHandle>,
}

impl GpuAssetStore {
    pub fn new() -> Self {
        Self {
            textures: HashMap::new(),
            static_meshes: HashMap::new(),
            skinned_geometry: HashMap::new(),
            joint_textures: HashMap::new(),
            morph_targets: HashMap::new(),
            weight_textures: HashMap::new(),
        }
    }

    pub fn texture_count(&self) -> usize { self.textures.len() }
    pub fn static_mesh_count(&self) -> usize { self.static_meshes.len() }
    pub fn skinned_mesh_count(&self) -> usize { self.skinned_geometry.len() }
    pub fn animation_count(&self) -> usize { self.joint_textures.len() }
    pub fn morph_target_count(&self) -> usize { self.morph_targets.len() }
    pub fn morph_animation_count(&self) -> usize { self.weight_textures.len() }

    /// Uploaded morph targets of `name`, if they were uploaded.
    pub fn morph_target(&self, name: &str) -> Option<&Arc<GpuMorphTargets>> {
        self.morph_targets.get(name)
    }

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
        mesh: &DeformableMeshAsset,
    ) -> Result<Arc<GpuSkinnedMesh>, UploadError> {
        if let Some(cached) = self.skinned_geometry.get(name) {
            return Ok(cached.clone());
        }

        let uploaded = Arc::new(mesh.upload(name, &mut ctx.upload)?);
        self.skinned_geometry.insert(name.to_string(), uploaded.clone());
        Ok(uploaded)
    }

    /// Upload the tracks of one animation asset that `rig` can consume.
    ///
    /// Each texture is cached per `(asset, track index)`, so a clip shared
    /// by several instances of one rig uploads each texture once.
    fn upload_clip_tracks(
        &mut self,
        ctx: &mut SceneUploadContext<'_>,
        asset_name: &str,
        rig: &RigSpec<'_>,
    ) -> Result<(Option<SampledTextureHandle>, Option<SampledTextureHandle>), UploadError> {
        let asset = ctx.asset(asset_name)?;
        let asset = asset.read().map_err(|_| UploadError::AssetLockPoisoned(asset_name.to_string()))?;
        let AssetData::AnimationAssetType(animation) = asset.get_data() else {
            return Err(UploadError::WrongKind {
                name: asset_name.to_string(),
                actual: asset.get_data().kind(),
                expected: AssetKind::Animation,
            });
        };

        let mut joints = None;
        let mut weights = None;
        for (index, track) in animation.tracks.iter().enumerate() {
            match track {
                BakedTrack::Joints { skin, texture } if rig.has_skin && rig.skin == Some(skin.as_str()) => {
                    joints = Some(Self::upload_track_texture(
                        &mut self.joint_textures,
                        ctx,
                        format!("{asset_name}.joints"),
                        (asset_name.to_string(), index),
                        texture,
                    )?);
                }
                BakedTrack::MorphWeights { mesh, texture, .. } if rig.has_target && mesh == rig.mesh => {
                    weights = Some(Self::upload_track_texture(
                        &mut self.weight_textures,
                        ctx,
                        format!("{asset_name}.weights"),
                        (asset_name.to_string(), index),
                        texture,
                    )?);
                }
                _ => {}
            }
        }

        Ok((joints, weights))
    }

    /// Upload one baked track texture, cached by `key`.
    fn upload_track_texture(
        cache: &mut HashMap<(String, usize), SampledTextureHandle>,
        ctx: &mut SceneUploadContext<'_>,
        label: String,
        key: (String, usize),
        texture: &TextureAsset,
    ) -> Result<SampledTextureHandle, UploadError> {
        if let Some(cached) = cache.get(&key) {
            return Ok(*cached);
        }

        let handle = ctx.upload.upload_texture(&label, texture)?.handle;
        cache.insert(key, handle);
        Ok(handle)
    }

    /// Upload a morph target asset's layout, deltas and base weights.
    ///
    /// Cached by asset name: a target shared by several instances of the same
    /// mesh is uploaded once.
    fn upload_morph_target(
        &mut self,
        ctx: &mut SceneUploadContext<'_>,
        name: &str,
    ) -> Result<Arc<GpuMorphTargets>, UploadError> {
        if let Some(cached) = self.morph_targets.get(name) {
            return Ok(cached.clone());
        }

        let asset = ctx.asset(name)?;
        let asset = asset.read().map_err(|_| UploadError::AssetLockPoisoned(name.to_string()))?;
        let AssetData::VertexMorphTargetAssetType(target) = asset.get_data() else {
            return Err(UploadError::WrongKind {
                name: name.to_string(),
                actual: asset.get_data().kind(),
                expected: AssetKind::VertexMorphTarget,
            });
        };

        let uploaded = Arc::new(target.upload(name, &mut ctx.upload)?);
        self.morph_targets.insert(name.to_string(), uploaded.clone());
        Ok(uploaded)
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

        // Discover every baked track up front and sort deterministically.
        // Textures are still uploaded lazily below, once per asset that the
        // scene actually references.
        let catalog = ClipCatalog::from_assets(ctx.assets);
        let mut animations = AnimationManager::new();

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
            skinned.push(self.skinned_instance(
                ctx,
                name,
                world_matrix,
                &catalog,
                &mut animations,
            )?);
        }

        log::info!(
            "Registered {} animation clips for {} skinned instances",
            animations.clips().len(),
            animations.player_count()
        );

        Ok(DrawList::new(static_batches, skinned, animations))
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

    /// Create one animated mesh instance, its clips and its animation player.
    ///
    /// Every clip of the rig is uploaded and registered, so the player can
    /// switch between them at runtime without touching the upload phase. The
    /// deforming stages ([`RigStages`](super::rig_plan::RigStages)) are decided
    /// here and remain immutable.
    /// 
    /// TODO: glTF describes one animation as a single unit covering both
    /// joints and morph weights, but the asset layer splits it into two assets
    /// that are recombined here by their shared `source` string. The halves
    /// can only be checked against each other at this point, and the check is
    /// incomplete: a frame-count mismatch silently drops the weight track, and
    /// a mismatched `fps` is not detected at all. Model one clip as one asset
    /// holding every track it drives, so the halves cannot disagree.
    /// 
    /// TODO: maybe try lazy uploading?
    fn skinned_instance(
        &mut self,
        ctx: &mut SceneUploadContext<'_>,
        name: &str,
        world_matrix: [[f32; 4]; 4],
        catalog: &ClipCatalog,
        animations: &mut AnimationManager,
    ) -> Result<AnimatedMesh, UploadError> {
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

        // A missing morph target asset degrades the rig instead of failing the
        // whole scene. The clip catalog drops the weight tracks and the gate
        // falls back to a skin-only (or rest pose) plan.
        let morph = match mesh.morph_target.as_deref() {
            Some(target_name) => match self.upload_morph_target(ctx, target_name) {
                Ok(target) => Some(target),
                Err(error) => {
                    log::error!(
                        "Rig '{name}' names morph target '{target_name}', which cannot be uploaded ({error}); morphing is disabled for this rig."
                    );
                    None
                }
            },
            None => None,
        };
        let has_skin = match (mesh.skin.as_deref(), mesh.skinning.is_some()) {
            (Some(skin), true) => {
                let exists = ctx.has_asset_of_kind(skin, AssetKind::Skin);
                if !exists {
                    log::error!(
                        "Rig '{name}' names skin '{skin}', which cannot be uploaded; skinning is disabled for this rig."
                    );
                }
                exists
            }
            (Some(skin), false) => {
                // The asset claims a skin but carries no per-vertex joint data.
                // Treat it as a morph-only (or rest-pose) mesh instead of
                // letting the skinning stage bind a missing buffer.
                log::error!(
                    "Rig '{name}' names skin '{skin}' but has no skinning attributes; skinning is disabled for this rig."
                );
                false
            }
            (None, _) => false,
        };

        let rig = RigSpec {
            mesh: name,
            skin: mesh.skin.as_deref(),
            has_skin,
            has_target: morph.is_some(),
            base_weights_nonzero: morph.as_ref().is_some_and(|target| target.base_weights.is_some()),
        };

        // Filter every animation asset down to the tracks this rig can consume.
        let clips = catalog.clips_for_rig(&rig);
        let rig_id = RigId::new(name);
        for clip in &clips {
            // TODO: lazy uploading of animation resources
            let (joints, weights) = self.upload_clip_tracks(ctx, &clip.asset, &rig)?;
            animations.register_clip(AnimationClip::new(
                clip.name.clone(),
                rig_id.clone(),
                joints,
                weights,
                clip.frames,
                clip.fps,
            ));
        }

        let stages = RigStages::from_assets(&rig, &clips);
        if clips.is_empty() {
            log::warn!("Rig '{name}' has no animation clips; it renders its rest pose.");
        } else {
            log::debug!(
                "Rig '{name}': {:?}, {} clip(s), final {:?}.",
                stages,
                clips.len(),
                stages.final_buffer()
            );
        }

        // A player always exists, one per draw list entry, even without a clip.
        let default_clip = animations.clips_for_rig(&rig_id).first().copied();
        let player = animations.add_player(rig_id, default_clip);

        AnimatedMesh::new(
            ctx.upload.device(),
            ctx.upload.queue(),
            geometry,
            material,
            world_matrix,
            player,
            stages,
            morph,
        )
    }
}

impl Default for GpuAssetStore {
    fn default() -> Self { Self::new() }
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

    /// Whether `name` is loaded and holds a `kind` asset.
    ///
    /// Used for capability checks, where a missing or mismatched asset only
    /// disables a stage instead of failing the whole scene.
    fn has_asset_of_kind(&self, name: &str, kind: AssetKind) -> bool {
        self.asset_kind(name).is_ok_and(|actual| actual == kind)
    }
}
