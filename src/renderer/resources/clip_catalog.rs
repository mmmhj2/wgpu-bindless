//! Selection of the baked tracks one rig can consume.

use crate::{
    asset::{
        asset_manager::AssetManager,
        asset_types::{animation_asset::BakedTrack, AssetData},
    },
    renderer::resources::rig_plan::RigSpec,
};

/// What one animation asset drives, indexed for rig matching.
struct ClipEntry {
    /// Asset name, used to upload the tracks.
    asset_name: String,
    /// Clip identity, `scene{scene}/anim{animation}`.
    source: String,
    frames: u32,
    fps: f32,
    /// Skin the joint track drives, if the animation carries one.
    joints: Option<String>,
    /// Meshes whose morph weights the animation drives.
    weights: Vec<String>,
}

/// Every animation asset of the database, in deterministic order.
#[derive(Default)]
pub(crate) struct ClipCatalog {
    entries: Vec<ClipEntry>,
}

impl ClipCatalog {
    /// TODO: this scans (and string-clones out of) the entire asset database
    /// on every `prepare_scene`. Once assets carry interned ids, build the
    /// catalog incrementally on insert, or at least avoid the clones.
    pub(crate) fn from_assets(assets: &AssetManager) -> Self {
        let mut entries = Vec::new();

        for (name, asset) in assets.get_database().iter() {
            let Ok(asset) = asset.read() else { continue };
            let AssetData::AnimationAssetType(animation) = asset.get_data() else { continue };

            let mut joints = None;
            let mut weights = Vec::new();
            for track in &animation.tracks {
                match track {
                    BakedTrack::Joints { skin, .. } => joints = Some(skin.clone()),
                    BakedTrack::MorphWeights { mesh, .. } => weights.push(mesh.clone()),
                }
            }

            entries.push(ClipEntry {
                asset_name: name.clone(),
                source: animation.source.clone(),
                frames: animation.frames,
                fps: animation.fps,
                joints,
                weights,
            });
        }

        Self { entries }
    }

    /// Clips `rig` can play, filtered from every animation asset.
    ///
    /// Tracks the rig cannot consume are dropped here with a warning instead of
    /// failing later. A clip that keeps neither track is not a clip at all and is
    /// omitted.
    pub(crate) fn clips_for_rig(&self, rig: &RigSpec<'_>) -> Vec<ClipTracks> {
        // A rig whose skin was not imported cannot consume any joint track.
        if !rig.has_skin {
            if let Some(skin) = rig.skin {
                let dropped = self.entries.iter().filter(|entry| entry.joints.as_deref() == Some(skin)).count();
                if dropped > 0 {
                    log::warn!(
                        "Dropping {dropped} joint track(s) of rig '{}': its skin '{skin}' was not imported.",
                        rig.mesh
                    );
                }
            }
        }

        // A mesh without a morph target asset cannot consume any weight track.
        if !rig.has_target {
            let dropped = self.entries.iter().filter(|entry| entry.weights.iter().any(|mesh| mesh == rig.mesh)).count();
            if dropped > 0 {
                log::warn!(
                    "Dropping {dropped} weight track(s) of rig '{}': the mesh has no morph target asset.",
                    rig.mesh
                );
            }
        }

        let mut clips: Vec<ClipTracks> = self
            .entries
            .iter()
            .filter_map(|entry| {
                let joints = rig.has_skin && entry.joints.as_deref() == rig.skin;
                let weights = rig.has_target && entry.weights.iter().any(|mesh| mesh == rig.mesh);
                if !joints && !weights {
                    return None;
                }

                Some(ClipTracks {
                    asset: entry.asset_name.clone(),
                    name: entry.source.clone(),
                    frames: entry.frames,
                    fps: entry.fps,
                    joints,
                    weights,
                })
            })
            .collect();

        // Sorted by source, so the clip order (and therefore the default clip)
        // does not depend on how the assets happened to be discovered.
        clips.sort_by(|a, b| (&a.name, &a.asset).cmp(&(&b.name, &b.asset)));
        clips
    }
}

/// One clip of a rig, after rig-compatibility filtering.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipTracks {
    /// Asset name of the animation, used to upload its tracks.
    pub asset: String,
    /// Clip identity: the animation's `source`, e.g. `scene0/anim1`.
    ///
    /// This is what the runtime registers the clip under, so `play_by_name`
    /// speaks one naming scheme for every clip.
    pub name: String,
    pub frames: u32,
    pub fps: f32,
    /// Whether the clip carries a joint track this rig can consume.
    pub joints: bool,
    /// Whether the clip carries a weight track this rig can consume.
    pub weights: bool,
}

#[cfg(test)]
mod test {
    use super::*;

    fn entry(source: &str, asset: &str, joints: Option<&str>, weights: &[&str]) -> ClipEntry {
        ClipEntry {
            asset_name: asset.to_string(),
            source: source.to_string(),
            frames: 30,
            fps: 30.0,
            joints: joints.map(str::to_string),
            weights: weights.iter().map(|mesh| mesh.to_string()).collect(),
        }
    }

    fn catalog(entries: Vec<ClipEntry>) -> ClipCatalog { ClipCatalog { entries } }

    fn rig<'a>(mesh: &'a str, skin: Option<&'a str>, has_skin: bool, has_target: bool) -> RigSpec<'a> {
        RigSpec { mesh, skin, has_skin, has_target, base_weights_nonzero: false }
    }

    #[test]
    fn one_animation_asset_yields_one_clip_with_both_tracks() {
        let catalog = catalog(vec![entry(
            "scene0/anim0",
            "rig.anim_0_0",
            Some("rig.skin"),
            &["rig.skpr"],
        )]);

        let clips = catalog.clips_for_rig(&rig("rig.skpr", Some("rig.skin"), true, true));

        assert_eq!(clips.len(), 1, "one asset is one clip");
        assert_eq!(clips[0].name, "scene0/anim0", "the clip is named by its source");
        assert_eq!(clips[0].asset, "rig.anim_0_0");
        assert!(clips[0].joints && clips[0].weights, "both tracks are usable");
    }

    #[test]
    fn a_clip_survives_losing_one_of_its_tracks() {
        let catalog = catalog(vec![entry(
            "scene0/anim0",
            "rig.anim_0_0",
            Some("rig.skin"),
            &["rig.skpr"],
        )]);

        // The skin is fine, but the mesh's morph target asset is gone.
        let clips = catalog.clips_for_rig(&rig("rig.skpr", Some("rig.skin"), true, false));

        assert_eq!(clips.len(), 1);
        assert!(clips[0].joints, "the joint track is still usable");
        assert!(!clips[0].weights, "the weight track is dropped");
    }

    #[test]
    fn joint_track_of_a_foreign_skin_is_not_selected() {
        let catalog = catalog(vec![entry("scene0/anim0", "other.anim", Some("other.skin"), &[])]);

        let clips = catalog.clips_for_rig(&rig("rig.skpr", Some("rig.skin"), true, true));

        assert!(clips.is_empty(), "the track drives a different skin");
    }

    #[test]
    fn joint_track_without_a_skin_is_dropped() {
        let catalog = catalog(vec![entry("scene0/anim0", "rig.anim_0_0", Some("missing.skin"), &[])]);

        // The mesh names a skin, but that skin asset was never imported.
        let clips = catalog.clips_for_rig(&rig("rig.skpr", Some("missing.skin"), false, true));

        assert!(clips.is_empty(), "the only track is unusable: {clips:?}");
    }

    #[test]
    fn weight_track_without_a_morph_target_is_dropped() {
        let catalog = catalog(vec![entry("scene0/anim0", "rig.anim_0_0", None, &["rig.skpr"])]);

        // The weights exist, but the mesh has no morph target asset to apply them.
        let clips = catalog.clips_for_rig(&rig("rig.skpr", None, false, false));

        assert!(clips.is_empty(), "the only track is unusable: {clips:?}");
    }

    #[test]
    fn clips_are_sorted_by_source() {
        let catalog = catalog(vec![
            entry("scene0/anim2", "rig.anim_0_2", Some("rig.skin"), &[]),
            entry("scene0/anim0", "rig.anim_0_0", Some("rig.skin"), &[]),
            entry("scene0/anim1", "rig.anim_0_1", Some("rig.skin"), &[]),
        ]);

        let clips = catalog.clips_for_rig(&rig("rig.skpr", Some("rig.skin"), true, true));
        let names: Vec<&str> = clips.iter().map(|clip| clip.name.as_str()).collect();

        assert_eq!(names, ["scene0/anim0", "scene0/anim1", "scene0/anim2"]);
    }

    #[test]
    fn a_morph_only_clip_belongs_to_the_mesh_it_drives() {
        let catalog = catalog(vec![
            entry("scene0/anim0", "a.anim_0_0", None, &["a.skpr"]),
            entry("scene0/anim0", "b.anim_0_0", None, &["b.skpr"]),
        ]);

        let clips = catalog.clips_for_rig(&rig("b.skpr", None, false, true));

        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].asset, "b.anim_0_0");
        assert!(!clips[0].joints && clips[0].weights);
    }
}
