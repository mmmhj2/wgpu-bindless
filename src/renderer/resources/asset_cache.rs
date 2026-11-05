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
        animation::{AnimationClip, AnimationManager, RigId},
        pipeline::pbr_material::PBRMaterial,
        drawable::{
            draw_list::DrawList,
            skinned_mesh::AnimatedMesh,
            static_mesh::StaticMesh,
            static_mesh_batch::StaticMeshBatch,
        },
        resources::bindless::SampledTextureHandle,
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

/// One baked track discovered in the asset database.
#[derive(Debug, Clone, PartialEq)]
struct BakedTrack {
    /// Import source shared by both halves of one glTF animation.
    source: String,
    /// Asset name of the track, used to upload its texture.
    asset_name: String,
    frames: u32,
    fps: f32,
}

/// Every baked track of the asset database, indexed by what it drives.
///
/// Clips are assembled per rig by [`ClipCatalog::clips_for_rig`],
/// because only the rig knows which of the two halves it can consume.
#[derive(Default)]
struct ClipCatalog {
    joints_by_skin: HashMap<String, Vec<BakedTrack>>,
    weights_by_mesh: HashMap<String, Vec<BakedTrack>>,
}

impl ClipCatalog {
    fn from_assets(assets: &AssetManager) -> Self {
        let mut catalog = Self::default();

        for (name, asset) in assets.get_database().iter() {
            let Ok(asset) = asset.read() else { continue };
            match asset.get_data() {
                AssetData::SkinnedAnimationAssetType(animation) => catalog
                    .joints_by_skin
                    .entry(animation.skin.clone())
                    .or_default()
                    .push(BakedTrack {
                        source: animation.source.clone(),
                        asset_name: name.clone(),
                        frames: animation.frames,
                        fps: animation.fps,
                    }),
                AssetData::VertexMorphAnimationAssetType(animation) => catalog
                    .weights_by_mesh
                    .entry(animation.mesh.clone())
                    .or_default()
                    .push(BakedTrack {
                        source: animation.source.clone(),
                        asset_name: name.clone(),
                        frames: animation.frames,
                        fps: animation.fps,
                    }),
                _ => {}
            }
        }

        // Sorted by source, so the clip order (and therefore the default clip)
        // does not depend on hash-map iteration. The asset name breaks ties.
        for tracks in catalog.joints_by_skin.values_mut() {
            tracks.sort_by(|a, b| (&a.source, &a.asset_name).cmp(&(&b.source, &b.asset_name)));
        }
        for tracks in catalog.weights_by_mesh.values_mut() {
            tracks.sort_by(|a, b| (&a.source, &a.asset_name).cmp(&(&b.source, &b.asset_name)));
        }

        catalog
    }

    /// Assemble the clips `rig` can play, merged from both track kinds.
    ///
    /// Tracks that the rig cannot consume (e.g. a skeletal track for a
    /// mesh whose skin is missing) are dropped here with an error instead
    /// of failing later.
    fn clips_for_rig(&self, rig: &RigSpec<'_>) -> Vec<ClipTracks> {
        let mut merged: HashMap<String, ClipTracks> = HashMap::new();

        if let Some(tracks) = rig.skin.and_then(|skin| self.joints_by_skin.get(skin)) {
            if rig.has_skin {
                for track in tracks {
                    let clip = merged.entry(track.source.clone()).or_insert_with(|| ClipTracks::new(track));
                    clip.joints = Some(track.asset_name.clone());
                }
            } else if !tracks.is_empty() {
                log::error!(
                    "Dropping {} skeletal track(s) of rig '{}': its skin '{}' was not imported.",
                    tracks.len(),
                    rig.mesh,
                    rig.skin.unwrap_or("<unnamed>")
                );
            }
        }

        if let Some(tracks) = self.weights_by_mesh.get(rig.mesh) {
            if rig.has_target {
                for track in tracks {
                    let clip = merged.entry(track.source.clone()).or_insert_with(|| ClipTracks::new(track));
                    if clip.frames != track.frames {
                        log::warn!(
                            "Clip '{}' of rig '{}' has {} baked frames in its skeletal track but {} in its weight track; using the skeletal count.",
                            track.source,
                            rig.mesh,
                            clip.frames,
                            track.frames
                        );
                    }
                    clip.weights = Some(track.asset_name.clone());
                }
            } else if !tracks.is_empty() {
                log::error!(
                    "Dropping {} morph weight track(s) of rig '{}': the mesh has no morph target asset.",
                    tracks.len(),
                    rig.mesh
                );
            }
        }

        let mut clips: Vec<ClipTracks> = merged.into_values().collect();
        clips.sort_by(|a, b| (&a.source, &a.name).cmp(&(&b.source, &b.name)));
        clips
    }
}

/// One clip of a rig, after rig-compatibility filtering.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipTracks {
    /// Clip name: the asset name of its primary track (joints first).
    pub name: String,
    /// Import source shared by both halves of the glTF animation.
    pub source: String,
    pub frames: u32,
    pub fps: f32,
    /// Baked skeletal track, if the clip carries one.
    pub joints: Option<String>,
    /// Baked morph weight track, if the clip carries one.
    pub weights: Option<String>,
}

impl ClipTracks {
    fn new(track: &BakedTrack) -> Self {
        Self {
            name: track.asset_name.clone(),
            source: track.source.clone(),
            frames: track.frames,
            fps: track.fps,
            joints: None,
            weights: None,
        }
    }
}

/// How one deformable mesh asset looks to the clip catalog and the stage gate.
#[derive(Debug, Clone, Copy)]
pub struct RigSpec<'a> {
    /// Asset name of the deformable mesh, i.e. the rig identity.
    pub mesh: &'a str,
    /// Skin named by the mesh, if any.
    pub skin: Option<&'a str>,
    /// `has_skin`: the mesh names a skin and that asset exists.
    pub has_skin: bool,
    /// `has_target`: the mesh names a morph target and that asset exists.
    pub has_target: bool,
    /// The target's `default_weights` exist and are not all zero.
    pub base_weights_nonzero: bool,
}

/// The five inputs of the stage gate (section 3.2 of the plan).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RigCapabilities {
    /// The mesh names a skin, and that skin asset exists.
    pub has_skin: bool,
    /// At least one clip of the rig carries a skeletal track.
    pub has_joint_track: bool,
    /// The mesh names a morph target, and that target asset exists.
    pub has_target: bool,
    /// At least one clip of the rig carries a weight track.
    pub has_weight_track: bool,
    /// The target has non-zero static base weights.
    pub base_weights_nonzero: bool,
}

/// One vertex-deformation stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RigStage {
    /// Reads the rest pose (or the skipped stage's output) and applies morph
    /// target deltas.
    Morph,
    /// Reads the rest pose or the morph output and applies joint matrices.
    Skin,
}

/// Buffer holding the vertex stream the rasterizer reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalBuffer {
    /// Shared rest pose, no per-instance buffers at all.
    Restpose,
    /// Per-instance morph output.
    MorphOut,
    /// Per-instance skin output.
    SkinOut,
}

/// The stage set of one rig, in dispatch order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RigStages {
    None,
    Morph,
    Skin,
    MorphThenSkin,
}

impl RigStages {
    pub fn as_slice(self) -> &'static [RigStage] {
        match self {
            Self::None => &[],
            Self::Morph => &[RigStage::Morph],
            Self::Skin => &[RigStage::Skin],
            Self::MorphThenSkin => &[RigStage::Morph, RigStage::Skin],
        }
    }

    pub fn final_buffer(self) -> FinalBuffer {
        match self {
            Self::None => FinalBuffer::Restpose,
            Self::Morph => FinalBuffer::MorphOut,
            Self::Skin | Self::MorphThenSkin => FinalBuffer::SkinOut,
        }
    }

    pub fn contains(self, stage: RigStage) -> bool { self.as_slice().contains(&stage) }
}

/// Which deformation stages one rig runs, and what its final vertex stream is.
/// 
/// TODO: maybe combine this struct with `RigStages`, as they are currently
/// identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RigPlan {
    stages: RigStages,
}

impl RigPlan {
    /// Maps rig capabilities to rig plan.
    /// 
    /// - A rig with both skin and a joint track attached will have skin stage.
    /// - A rig with both morph target and a weight track (or non zero base
    /// weight) will have morph stage.
    /// - A rig satisfying both conditions will have both stages enabled.
    pub fn from_capabilities(caps: RigCapabilities) -> Self {
        let skin = caps.has_skin && caps.has_joint_track;
        let morph = caps.has_target && (caps.has_weight_track || caps.base_weights_nonzero);

        let stages = match (morph, skin) {
            (false, false) => RigStages::None,
            (true, false) => RigStages::Morph,
            (false, true) => RigStages::Skin,
            (true, true) => RigStages::MorphThenSkin,
        };

        Self { stages }
    }

    /// Determine which stages are used for one rig from its assets and its clip catalog.
    pub fn from_assets(rig: &RigSpec<'_>, clips: &[ClipTracks]) -> Self {
        Self::from_capabilities(RigCapabilities {
            has_skin: rig.has_skin,
            has_joint_track: clips.iter().any(|clip| clip.joints.is_some()),
            has_target: rig.has_target,
            has_weight_track: clips.iter().any(|clip| clip.weights.is_some()),
            base_weights_nonzero: rig.base_weights_nonzero,
        })
    }

    pub fn stages(self) -> RigStages { self.stages }
    /// Stages in dispatch order.
    pub fn stage_list(self) -> &'static [RigStage] { self.stages.as_slice() }
    pub fn final_buffer(self) -> FinalBuffer { self.stages.final_buffer() }
    /// Whether any stage runs at all.
    pub fn is_empty(self) -> bool { matches!(self.stages, RigStages::None) }
    /// Whether the rig runs the skinning stage.
    pub fn skins(self) -> bool { self.stages.contains(RigStage::Skin) }
    /// Whether the rig runs the morphing stage.
    pub fn morphs(self) -> bool { self.stages.contains(RigStage::Morph) }
}

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
    /// Morph target subasset name -> uploaded layout + deltas + base weights.
    ///
    /// Shared across instances, like the rest pose: only the weights vary.
    morph_targets: HashMap<String, Arc<GpuMorphTargets>>,
    /// Morph animation subasset name -> baked weight texture handle.
    morph_animations: HashMap<String, SampledTextureHandle>,
}

impl GpuAssetStore {
    pub fn new() -> Self {
        Self {
            textures: HashMap::new(),
            static_meshes: HashMap::new(),
            skinned_geometry: HashMap::new(),
            skinned_animations: HashMap::new(),
            morph_targets: HashMap::new(),
            morph_animations: HashMap::new(),
        }
    }

    pub fn texture_count(&self) -> usize { self.textures.len() }
    pub fn static_mesh_count(&self) -> usize { self.static_meshes.len() }
    pub fn skinned_mesh_count(&self) -> usize { self.skinned_geometry.len() }
    pub fn animation_count(&self) -> usize { self.skinned_animations.len() }
    pub fn morph_target_count(&self) -> usize { self.morph_targets.len() }
    pub fn morph_animation_count(&self) -> usize { self.morph_animations.len() }

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
        mesh: &SkinnedMeshAsset,
    ) -> Result<Arc<GpuSkinnedMesh>, UploadError> {
        if let Some(cached) = self.skinned_geometry.get(name) {
            return Ok(cached.clone());
        }

        let uploaded = Arc::new(mesh.upload(name, &mut ctx.upload)?);
        self.skinned_geometry.insert(name.to_string(), uploaded.clone());
        Ok(uploaded)
    }

    /// Upload a baked animation texture.
    ///
    /// Cached by asset name, so a clip shared by several skinned instances only
    /// occupies one bindless texture slot.
    fn animation_texture(
        &mut self,
        ctx: &mut SceneUploadContext<'_>,
        name: &str,
    ) -> Result<SampledTextureHandle, UploadError> {
        if let Some(cached) = self.skinned_animations.get(name) {
            return Ok(*cached);
        }

        let asset = ctx.asset(name)?;
        let asset = asset.read().map_err(|_| UploadError::AssetLockPoisoned(name.to_string()))?;
        let AssetData::SkinnedAnimationAssetType(animation) = asset.get_data() else {
            return Err(UploadError::WrongKind {
                name: name.to_string(),
                actual: asset.get_data().kind(),
                expected: AssetKind::SkinnedAnimation,
            });
        };

        let handle = ctx.upload.upload_texture(name, &animation.texture)?.handle;
        self.skinned_animations.insert(name.to_string(), handle);
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

    /// Upload a baked morph weight texture.
    ///
    /// Cached by asset name, like [`Self::animation_texture`].
    fn upload_morph_animation(
        &mut self,
        ctx: &mut SceneUploadContext<'_>,
        name: &str,
    ) -> Result<SampledTextureHandle, UploadError> {
        if let Some(cached) = self.morph_animations.get(name) {
            return Ok(*cached);
        }

        let asset = ctx.asset(name)?;
        let asset = asset.read().map_err(|_| UploadError::AssetLockPoisoned(name.to_string()))?;
        let AssetData::VertexMorphAnimationAssetType(animation) = asset.get_data() else {
            return Err(UploadError::WrongKind {
                name: name.to_string(),
                actual: asset.get_data().kind(),
                expected: AssetKind::VertexMorphAnimation,
            });
        };

        let handle = ctx.upload.upload_texture(name, &animation.texture)?.handle;
        self.morph_animations.insert(name.to_string(), handle);
        Ok(handle)
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
    /// deforming stages ([`RigStages`]) are decided here and remains immutable.
    /// 
    /// TODO: Combining two halves of clips seems unnatural: glTF has
    /// a unified animation description for both morph and joints,
    /// but we split it in the asset system and recombine it here.
    /// Figure out a better way as a part of the animation system rewrite.
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
        let has_skin = match mesh.skin.as_deref() {
            Some(skin) => {
                let exists = ctx.has_asset_of_kind(skin, AssetKind::Skin);
                if !exists {
                    log::error!(
                        "Rig '{name}' names skin '{skin}', which cannot be uploaded; skinning is disabled for this rig."
                    );
                }
                exists
            }
            None => false,
        };

        let rig = RigSpec {
            mesh: name,
            skin: mesh.skin.as_deref(),
            has_skin,
            has_target: morph.is_some(),
            base_weights_nonzero: morph.as_ref().is_some_and(|target| target.base_weights.is_some()),
        };

        // Merge the two baked halves of every animation into one clip per
        // `(source, rig)`. tracks the rig cannot consume are dropped there.
        let clips = catalog.clips_for_rig(&rig);
        let rig_id = RigId::new(name);
        for clip in &clips {
            // TODO: lazy uploading of animation resources
            let joints = match &clip.joints {
                Some(asset_name) => Some(self.animation_texture(ctx, asset_name)?),
                None => None,
            };
            let weights = match &clip.weights {
                Some(asset_name) => Some(self.upload_morph_animation(ctx, asset_name)?),
                None => None,
            };
            animations.register_clip(AnimationClip::new(
                clip.name.clone(),
                rig_id.clone(),
                joints,
                weights,
                clip.frames,
                clip.fps,
            ));
        }

        let plan = RigPlan::from_assets(&rig, &clips);
        if clips.is_empty() {
            log::warn!("Rig '{name}' has no animation clips; it renders its rest pose.");
        } else {
            log::debug!(
                "Rig '{name}': {:?}, {} clip(s), final {:?}.",
                plan.stages(),
                clips.len(),
                plan.final_buffer()
            );
        }

        // A player always exists, one per draw list entry, even without a clip.
        let default_clip = animations.clips_for_rig(&rig_id).first().copied();
        animations.add_player(rig_id, default_clip);

        // TODO: allocate the buffers of every planned stage and swap
        // `get_vertex_buffer` over to `plan.final_buffer()`. Until then only a
        // skinning stage allocates deformed buffers, and morph-only rigs render
        // their rest pose.
        let animated = plan.skins();

        AnimatedMesh::new(
            ctx.upload.device(),
            ctx.upload.queue(),
            geometry,
            material,
            world_matrix,
            animated,
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

#[cfg(test)]
mod test {
    use super::*;

    fn caps(
        has_skin: bool,
        has_joint_track: bool,
        has_target: bool,
        has_weight_track: bool,
        base_weights_nonzero: bool,
    ) -> RigCapabilities {
        RigCapabilities {
            has_skin,
            has_joint_track,
            has_target,
            has_weight_track,
            base_weights_nonzero,
        }
    }

    /// Exhaustive table of all possible combinations of vertex morph or skinning
    /// for mesh. Used for testing.
    fn gate_rows() -> Vec<(&'static str, RigCapabilities, RigStages, FinalBuffer)> {
        vec![
            ("R1", caps(false, false, false, false, false), RigStages::None, FinalBuffer::Restpose),
            ("R2", caps(false, false, true, false, false), RigStages::None, FinalBuffer::Restpose),
            ("R3", caps(false, false, true, false, true), RigStages::Morph, FinalBuffer::MorphOut),
            ("R4", caps(false, false, true, true, false), RigStages::Morph, FinalBuffer::MorphOut),
            ("R5", caps(true, false, false, false, false), RigStages::None, FinalBuffer::Restpose),
            ("R6", caps(true, false, true, false, false), RigStages::None, FinalBuffer::Restpose),
            ("R7", caps(true, false, true, false, true), RigStages::Morph, FinalBuffer::MorphOut),
            ("R8", caps(true, false, true, true, false), RigStages::Morph, FinalBuffer::MorphOut),
            ("R9", caps(true, true, false, false, false), RigStages::Skin, FinalBuffer::SkinOut),
            ("R10", caps(true, true, true, false, false), RigStages::Skin, FinalBuffer::SkinOut),
            (
                "R11",
                caps(true, true, true, false, true),
                RigStages::MorphThenSkin,
                FinalBuffer::SkinOut,
            ),
            (
                "R12",
                caps(true, true, true, true, false),
                RigStages::MorphThenSkin,
                FinalBuffer::SkinOut,
            ),
        ]
    }

    #[test]
    fn stage_gate_matches_all_twelve_rows() {
        let rows = gate_rows();
        assert_eq!(rows.len(), 12, "the table is exhaustive");

        for (label, capabilities, expected_stages, expected_final) in rows {
            let plan = RigPlan::from_capabilities(capabilities);
            assert_eq!(plan.stages(), expected_stages, "{label}: stages");
            assert_eq!(plan.final_buffer(), expected_final, "{label}: final buffer");
        }
    }

    #[test]
    fn morph_runs_before_skin_and_only_the_final_stage_feeds_the_rasterizer() {
        assert_eq!(RigStages::None.as_slice(), &[]);
        assert_eq!(RigStages::Morph.as_slice(), &[RigStage::Morph]);
        assert_eq!(RigStages::Skin.as_slice(), &[RigStage::Skin]);
        assert_eq!(
            RigStages::MorphThenSkin.as_slice(),
            &[RigStage::Morph, RigStage::Skin],
            "morph feeds skin, never the other way around"
        );

        let plan = RigPlan::from_capabilities(caps(true, true, true, true, false));
        assert!(plan.skins() && plan.morphs());
        assert!(!plan.is_empty());
    }

    fn track(source: &str, asset_name: &str) -> BakedTrack {
        BakedTrack {
            source: source.to_string(),
            asset_name: asset_name.to_string(),
            frames: 30,
            fps: 30.0,
        }
    }

    fn rig<'a>(mesh: &'a str, skin: Option<&'a str>, has_skin: bool, has_target: bool) -> RigSpec<'a> {
        RigSpec { mesh, skin, has_skin, has_target, base_weights_nonzero: false }
    }

    #[test]
    fn both_halves_of_one_animation_merge_into_one_clip() {
        let mut catalog = ClipCatalog::default();
        catalog.joints_by_skin.insert("rig.skin".into(), vec![track("scene0/anim0", "rig.anim_0_0")]);
        catalog.weights_by_mesh.insert("rig.skpr".into(), vec![track("scene0/anim0", "rig.morphanim_0_0_0_0")]);

        let clips = catalog.clips_for_rig(&rig("rig.skpr", Some("rig.skin"), true, true));

        assert_eq!(clips.len(), 1, "the halves share their source, so they are one clip");
        assert_eq!(clips[0].joints.as_deref(), Some("rig.anim_0_0"));
        assert_eq!(clips[0].weights.as_deref(), Some("rig.morphanim_0_0_0_0"));
        assert_eq!(clips[0].name, "rig.anim_0_0", "the skeletal asset names a mixed clip");
    }

    #[test]
    fn mixed_catalog_keeps_both_stages_when_one_clip_lacks_a_track() {
        let mut catalog = ClipCatalog::default();
        catalog.joints_by_skin.insert("rig.skin".into(), vec![track("scene0/anim0", "rig.anim_0_0")]);
        catalog.weights_by_mesh.insert("rig.skpr".into(), vec![track("scene0/anim1", "rig.morphanim_0_1_0_0")]);

        let rig = rig("rig.skpr", Some("rig.skin"), true, true);
        let clips = catalog.clips_for_rig(&rig);

        assert_eq!(clips.len(), 2);
        assert_eq!(clips[0].source, "scene0/anim0");
        assert!(clips[0].joints.is_some() && clips[0].weights.is_none(), "clip A drives only the skeleton");
        assert_eq!(clips[1].source, "scene0/anim1");
        assert!(clips[1].joints.is_none() && clips[1].weights.is_some(), "clip B drives only morph weights");

        // The gates are a union over the rig's catalog, so both stages exist even
        // though no single clip drives both.
        let plan = RigPlan::from_assets(&rig, &clips);
        assert_eq!(plan.stages(), RigStages::MorphThenSkin);
        assert_eq!(plan.final_buffer(), FinalBuffer::SkinOut);
    }

    #[test]
    fn joints_track_without_a_skin_is_dropped() {
        let mut catalog = ClipCatalog::default();
        catalog.joints_by_skin.insert("missing.skin".into(), vec![track("scene0/anim0", "missing.anim")]);

        // The mesh names a skin, but that skin asset was never imported.
        let rig = rig("rig.skpr", Some("missing.skin"), false, true);
        let clips = catalog.clips_for_rig(&rig);

        assert!(clips.is_empty(), "the only track is unusable: {clips:?}");
        let plan = RigPlan::from_assets(&rig, &clips);
        assert_eq!(plan.stages(), RigStages::None, "the rig falls back to its rest pose");
        assert_eq!(plan.final_buffer(), FinalBuffer::Restpose);
    }

    #[test]
    fn weight_track_without_a_morph_target_is_dropped() {
        let mut catalog = ClipCatalog::default();
        catalog.weights_by_mesh.insert("rig.skpr".into(), vec![track("scene0/anim0", "rig.morphanim_0_0_0_0")]);

        // The weights exist, but the mesh has no morph target asset to apply them.
        let rig = rig("rig.skpr", None, false, false);
        let clips = catalog.clips_for_rig(&rig);

        assert!(clips.is_empty(), "the only track is unusable: {clips:?}");
        let plan = RigPlan::from_assets(&rig, &clips);
        assert_eq!(plan.stages(), RigStages::None);
        assert_eq!(plan.final_buffer(), FinalBuffer::Restpose);
    }

    #[test]
    fn a_clip_survives_losing_one_of_its_tracks() {
        let mut catalog = ClipCatalog::default();
        catalog.joints_by_skin.insert("rig.skin".into(), vec![track("scene0/anim0", "rig.anim_0_0")]);
        catalog.weights_by_mesh.insert("rig.skpr".into(), vec![track("scene0/anim0", "rig.morphanim_0_0_0_0")]);

        // The skin is fine, but the mesh's morph target asset is gone.
        let rig = rig("rig.skpr", Some("rig.skin"), true, false);
        let clips = catalog.clips_for_rig(&rig);

        assert_eq!(clips.len(), 1);
        assert!(clips[0].joints.is_some(), "the skeletal half is still usable");
        assert!(clips[0].weights.is_none(), "the weight half is dropped");

        let plan = RigPlan::from_assets(&rig, &clips);
        assert_eq!(plan.stages(), RigStages::Skin);
    }

    #[test]
    fn base_weights_alone_enable_the_morph_stage() {
        for (has_skin, has_joint_track, expected) in [
            (false, false, RigStages::Morph),
            (true, false, RigStages::Morph),
        ] {
            let mut rig = rig("rig.skpr", None, has_skin, true);
            rig.base_weights_nonzero = true;
            let caps = RigCapabilities {
                has_skin: rig.has_skin,
                has_joint_track,
                has_target: rig.has_target,
                has_weight_track: false,
                base_weights_nonzero: true,
            };
            assert_eq!(RigPlan::from_capabilities(caps).stages(), expected);
        }
    }
}
