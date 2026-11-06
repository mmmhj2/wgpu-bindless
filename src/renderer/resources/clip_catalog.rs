//! Discovery and merging of baked animation tracks.
//!
//! Split out of the former `asset_cache.rs`. The catalog scans the asset
//! database once and merges the skeletal and morph-weight halves of one glTF
//! animation into a single [`ClipTracks`] per rig.

use std::collections::HashMap;

use crate::{
    asset::{asset_manager::AssetManager, asset_types::AssetData},
    renderer::resources::rig_plan::RigSpec,
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
pub(crate) struct ClipCatalog {
    joints_by_skin: HashMap<String, Vec<BakedTrack>>,
    weights_by_mesh: HashMap<String, Vec<BakedTrack>>,
}

impl ClipCatalog {
    /// TODO: this scans (and string-clones out of) the entire asset database
    /// on every `prepare_scene`. Once assets carry interned ids, build the
    /// catalog incrementally on insert, or at least avoid the clones.
    pub(crate) fn from_assets(assets: &AssetManager) -> Self {
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
    pub(crate) fn clips_for_rig(&self, rig: &RigSpec<'_>) -> Vec<ClipTracks> {
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
                        log::error!(
                            "Dropping weight track '{}' of rig '{}'. The clip has {} baked frames but the weight track has {}.",
                            track.asset_name,
                            rig.mesh,
                            clip.frames,
                            track.frames
                        );
                        continue;
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

        // A track can be dropped above, leaving a clip with nothing to play.
        merged.retain(|_, clip| clip.joints.is_some() || clip.weights.is_some());

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

#[cfg(test)]
mod test {
    use super::*;
    use crate::renderer::resources::rig_plan::{FinalBuffer, RigStages};

    fn track(source: &str, asset_name: &str) -> BakedTrack {
        track_with_frames(source, asset_name, 30)
    }

    fn track_with_frames(source: &str, asset_name: &str, frames: u32) -> BakedTrack {
        BakedTrack {
            source: source.to_string(),
            asset_name: asset_name.to_string(),
            frames,
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
        let stages = RigStages::from_assets(&rig, &clips);
        assert_eq!(stages, RigStages::MorphThenSkin);
        assert_eq!(stages.final_buffer(), FinalBuffer::SkinOut);
    }

    #[test]
    fn joints_track_without_a_skin_is_dropped() {
        let mut catalog = ClipCatalog::default();
        catalog.joints_by_skin.insert("missing.skin".into(), vec![track("scene0/anim0", "missing.anim")]);

        // The mesh names a skin, but that skin asset was never imported.
        let rig = rig("rig.skpr", Some("missing.skin"), false, true);
        let clips = catalog.clips_for_rig(&rig);

        assert!(clips.is_empty(), "the only track is unusable: {clips:?}");
        let stages = RigStages::from_assets(&rig, &clips);
        assert_eq!(stages, RigStages::None, "the rig falls back to its rest pose");
        assert_eq!(stages.final_buffer(), FinalBuffer::Restpose);
    }

    #[test]
    fn weight_track_without_a_morph_target_is_dropped() {
        let mut catalog = ClipCatalog::default();
        catalog.weights_by_mesh.insert("rig.skpr".into(), vec![track("scene0/anim0", "rig.morphanim_0_0_0_0")]);

        // The weights exist, but the mesh has no morph target asset to apply them.
        let rig = rig("rig.skpr", None, false, false);
        let clips = catalog.clips_for_rig(&rig);

        assert!(clips.is_empty(), "the only track is unusable: {clips:?}");
        let stages = RigStages::from_assets(&rig, &clips);
        assert_eq!(stages, RigStages::None);
        assert_eq!(stages.final_buffer(), FinalBuffer::Restpose);
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

        let stages = RigStages::from_assets(&rig, &clips);
        assert_eq!(stages, RigStages::Skin);
    }

    #[test]
    fn mismatched_frame_counts_drop_the_weight_track() {
        let mut catalog = ClipCatalog::default();
        catalog.joints_by_skin.insert("rig.skin".into(), vec![track("scene0/anim0", "rig.anim_0_0")]);
        catalog.weights_by_mesh.insert(
            "rig.skpr".into(),
            vec![track_with_frames("scene0/anim0", "rig.morphanim_0_0_0_0", 15)],
        );

        let rig = rig("rig.skpr", Some("rig.skin"), true, true);
        let clips = catalog.clips_for_rig(&rig);

        assert_eq!(clips.len(), 1, "the skeletal half still plays");
        assert!(clips[0].joints.is_some());
        assert!(clips[0].weights.is_none(), "the mismatched weight track is dropped");

        // The dropped track no longer enables the morph stage.
        assert_eq!(RigStages::from_assets(&rig, &clips), RigStages::Skin);
    }
}
