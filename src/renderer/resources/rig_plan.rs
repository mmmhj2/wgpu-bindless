//! Stage gate and final-vertex-stream decisions for one deformable rig.
//!
//! Split out of the former `asset_cache.rs`. The gate is a pure function of
//! five booleans, so it is decided once in `prepare_scene` and never changes at
//! runtime.

use super::clip_catalog::ClipTracks;

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

/// The five inputs of the stage gate.
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
    /// Stages in dispatch order.
    pub fn as_slice(self) -> &'static [RigStage] {
        match self {
            Self::None => &[],
            Self::Morph => &[RigStage::Morph],
            Self::Skin => &[RigStage::Skin],
            Self::MorphThenSkin => &[RigStage::Morph, RigStage::Skin],
        }
    }

    /// Buffer holding the vertex stream the rasterizer reads.
    pub fn final_buffer(self) -> FinalBuffer {
        match self {
            Self::None => FinalBuffer::Restpose,
            Self::Morph => FinalBuffer::MorphOut,
            Self::Skin | Self::MorphThenSkin => FinalBuffer::SkinOut,
        }
    }

    pub fn contains(self, stage: RigStage) -> bool { self.as_slice().contains(&stage) }

    /// Whether any stage runs at all.
    pub fn is_empty(self) -> bool { matches!(self, Self::None) }

    /// Whether the rig runs the skinning stage.
    pub fn skins(self) -> bool { self.contains(RigStage::Skin) }

    /// Whether the rig runs the morphing stage.
    pub fn morphs(self) -> bool { self.contains(RigStage::Morph) }

    /// Maps rig capabilities to the stage set.
    ///
    /// - A rig with both skin and a joint track attached will have skin stage.
    /// - A rig with both morph target and a weight track (or non zero base
    ///   weight) will have morph stage.
    /// - A rig satisfying both conditions will have both stages enabled.
    pub fn from_capabilities(caps: RigCapabilities) -> Self {
        let skin = caps.has_skin && caps.has_joint_track;
        let morph = caps.has_target && (caps.has_weight_track || caps.base_weights_nonzero);

        match (morph, skin) {
            (false, false) => Self::None,
            (true, false) => Self::Morph,
            (false, true) => Self::Skin,
            (true, true) => Self::MorphThenSkin,
        }
    }

    /// Determine which stages are used for one rig from its assets and its clip catalog.
    pub fn from_assets(rig: &RigSpec<'_>, clips: &[ClipTracks]) -> Self {
        Self::from_capabilities(RigCapabilities {
            has_skin: rig.has_skin,
            has_joint_track: clips.iter().any(|clip| clip.joints),
            has_target: rig.has_target,
            has_weight_track: clips.iter().any(|clip| clip.weights),
            base_weights_nonzero: rig.base_weights_nonzero,
        })
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
            let stages = RigStages::from_capabilities(capabilities);
            assert_eq!(stages, expected_stages, "{label}: stages");
            assert_eq!(stages.final_buffer(), expected_final, "{label}: final buffer");
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

        let stages = RigStages::from_capabilities(caps(true, true, true, true, false));
        assert!(stages.skins() && stages.morphs());
        assert!(!stages.is_empty());
    }

    #[test]
    fn base_weights_alone_enable_the_morph_stage() {
        for (has_skin, has_joint_track, expected) in [
            (false, false, RigStages::Morph),
            (true, false, RigStages::Morph),
        ] {
            let caps = RigCapabilities {
                has_skin,
                has_joint_track,
                has_target: true,
                has_weight_track: false,
                base_weights_nonzero: true,
            };
            assert_eq!(RigStages::from_capabilities(caps), expected);
        }
    }
}
