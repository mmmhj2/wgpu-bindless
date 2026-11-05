//! CPU-side animation playback.
//! 
//! Contains four main classes:
//! - `AnimationManager`: entry point for animation management. Stores clips
//! and players.
//! - `AnimationPlayer`: CPU side animation playback state tracker.
//! - `AnimationClip`: abstraction over GPU resources of a clip.
//! - `AnimationFrame`: information for GPU skinning dispatch.

use std::collections::HashMap;

use crate::renderer::resources::bindless::SampledTextureHandle;

/// Upper bound applied to one [`AnimationManager::update`] step.
///
/// Delta times are clamped by this value to avoid animation jumping forward
/// abruptly.
/// 
/// TODO: figure out better reconciliation method. This only fix animation
/// desync ostensibly.
pub const MAX_ANIMATION_DELTA: f32 = 0.25;

/// Stable identifier of one clip inside an [`AnimationManager`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AnimationClipId(usize);

impl AnimationClipId {
    pub fn index(self) -> usize { self.0 }
}

/// How a player behaves when it reaches the end of its clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMode {
    /// Wrap back to the start of the clip.
    Loop,
    /// Stop on the last unique frame and mark the player as finished.
    Once,
}

/// One baked animation clip.
///
/// Note: The texture contains one extra row that repeats frame 0, so the shader
/// can interpolate from the last unique frame to the start of the next loop
/// without a modulo.
#[derive(Debug, Clone)]
pub struct AnimationClip {
    asset_name: String,
    skin: String,
    texture: SampledTextureHandle,
    /// frame numbers, excluding the last repeated wrapping frame.
    frames: u32,
    fps: f32,
}

impl AnimationClip {
    pub(crate) fn new(
        asset_name: String,
        skin: String,
        texture: SampledTextureHandle,
        frames: u32,
        fps: f32,
    ) -> Self {
        assert!(!asset_name.is_empty(), "animation clip must have an asset name");
        assert!(!skin.is_empty(), "animation clip '{asset_name}' must name its skin");
        assert!(frames > 0, "animation clip '{asset_name}' has no baked frames");
        assert!(
            fps.is_finite() && fps > 0.0,
            "animation clip '{asset_name}' has an invalid FPS ({fps})"
        );

        Self { asset_name, skin, texture, frames, fps }
    }

    pub fn asset_name(&self) -> &str { &self.asset_name }
    pub fn skin(&self) -> &str { &self.skin }
    pub fn texture(&self) -> SampledTextureHandle { self.texture }
    pub fn frames(&self) -> u32 { self.frames }
    pub fn fps(&self) -> f32 { self.fps }

    /// Playback period used by [`LoopMode::Loop`].
    pub fn duration(&self) -> f32 { self.frames as f32 / self.fps }

    /// Time of the last *unique* frame, used as the end point of
    /// [`LoopMode::Once`].
    pub fn last_unique_time(&self) -> f32 {
        if self.frames <= 1 {
            0.0
        } else {
            (self.frames - 1) as f32 / self.fps
        }
    }

    /// Resolve a playback time into the fractional frame index fed to the
    /// skinning shader.
    ///
    /// Invariants guaranteed for the shader:
    /// - [`LoopMode::Loop`]: `0.0 <= frame < frames`;
    /// - [`LoopMode::Once`]: `0.0 <= frame <= frames - 1`.
    ///
    /// This keeps the shader's `frame + 1` fetch inside the baked texture.
    pub fn frame_at(&self, time: f32, mode: LoopMode) -> f32 {
        assert!(time.is_finite(), "animation time must be finite, got {time}");

        let frame = match mode {
            LoopMode::Loop => {
                let time = time.rem_euclid(self.duration());
                let frame = time * self.fps;
                if frame >= self.frames as f32 { 0.0 } else { frame }
            }
            LoopMode::Once => {
                let time = time.clamp(0.0, self.last_unique_time());
                (time * self.fps).min((self.frames - 1) as f32)
            }
        };

        debug_assert!(
            frame.is_finite() && frame >= 0.0,
            "resolved a non-finite animation frame ({frame}) for clip '{}'",
            self.asset_name
        );
        frame
    }
}

/// Resolved skinning input for one dispatch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkinnedAnimationFrame {
    texture: SampledTextureHandle,
    frame: f32,
}

impl SkinnedAnimationFrame {
    /// Build a skinning input from a bindless texture handle and a fractional
    /// frame.
    ///
    /// This is the low-level escape hatch for callers that dispatch manually.
    /// Use [`AnimationManager::frame`] whenever possible.
    ///
    /// Panics if `frame` is negative or not finite.
    pub fn new(texture: SampledTextureHandle, frame: f32) -> Self {
        assert!(
            frame.is_finite() && frame >= 0.0,
            "skinning frame must be finite and non-negative, got {frame}"
        );
        Self { texture, frame }
    }

    pub fn texture(&self) -> SampledTextureHandle { self.texture }
    pub fn frame(&self) -> f32 { self.frame }
}

pub struct MorphAnimationFrame {
    // TODO: stub code
}

/// Playback state of one skinned mesh instance.
///
/// Players only hold time and policy, and the clip metadata stays in the owning
/// [`AnimationManager`]. Read-only accessors are public, and all mutations must
/// go through the manager.
#[derive(Debug, Clone)]
pub struct AnimationPlayer {
    skin: String,
    clip: Option<AnimationClipId>,
    time: f32,
    speed: f32,
    paused: bool,
    loop_mode: LoopMode,
    finished: bool,
}

impl AnimationPlayer {
    fn new(skin: &str, clip: Option<AnimationClipId>) -> Self {
        Self {
            skin: skin.to_string(),
            clip,
            time: 0.0,
            speed: 1.0,
            paused: false,
            loop_mode: LoopMode::Loop,
            finished: false,
        }
    }

    pub fn skin(&self) -> &str { &self.skin }
    pub fn clip(&self) -> Option<AnimationClipId> { self.clip }
    pub fn time(&self) -> f32 { self.time }
    pub fn speed(&self) -> f32 { self.speed }
    pub fn loop_mode(&self) -> LoopMode { self.loop_mode }
    pub fn is_paused(&self) -> bool { self.paused }
    /// Whether a [`LoopMode::Once`] player reached the last unique frame.
    pub fn is_finished(&self) -> bool { self.finished }

    fn advance(&mut self, delta_seconds: f32, clip: &AnimationClip) {
        if self.paused || delta_seconds <= 0.0 {
            return;
        }
        if self.loop_mode == LoopMode::Once && self.finished {
            return;
        }

        match self.loop_mode {
            LoopMode::Loop => {
                self.time = (self.time + delta_seconds).rem_euclid(clip.duration());
            }
            LoopMode::Once => {
                let end = clip.last_unique_time();
                self.time = (self.time + delta_seconds).min(end);
                if self.time >= end {
                    self.finished = true;
                }
            }
        }
    }
}

/// Owns the clip catalog and every player of one prepared scene.
///
/// The manager encapsulates CPU side management only. Actual skinning on the 
/// GPU side is performed via [`DrawList`].
/// 
/// [`DrawList`]: crate::renderer:drawable::draw_list::DrawList
#[derive(Debug)]
pub struct AnimationManager {
    clips: Vec<AnimationClip>,
    clip_by_asset: HashMap<String, AnimationClipId>,
    clips_by_skin: HashMap<String, Vec<AnimationClipId>>,
    players: Vec<AnimationPlayer>,
    paused: bool,
    time_scale: f32,
}

impl AnimationManager {
    pub(crate) fn new() -> Self {
        Self {
            clips: Vec::new(),
            clip_by_asset: HashMap::new(),
            clips_by_skin: HashMap::new(),
            players: Vec::new(),
            paused: false,
            time_scale: 1.0,
        }
    }

    /// Register a clip, deduplicating by asset name.
    /// 
    /// Called after each skinned mesh asset is registered into the asset store.
    pub(crate) fn register_clip(&mut self, clip: AnimationClip) -> AnimationClipId {
        if let Some(existing) = self.clip_by_asset.get(clip.asset_name()) {
            let existing_clip = &self.clips[existing.index()];
            assert_eq!(
                existing_clip.skin(),
                clip.skin(),
                "animation asset '{}' was registered for two skins",
                clip.asset_name()
            );
            log::warn!("animation asset {} is duplicated.", clip.asset_name());
            return *existing;
        }

        let id = AnimationClipId(self.clips.len());
        self.clips_by_skin
            .entry(clip.skin().to_string())
            .or_default()
            .push(id);
        self.clip_by_asset.insert(clip.asset_name().to_string(), id);
        self.clips.push(clip);
        id
    }

    /// Append one player, returning the index used by every player accessor.
    ///
    /// The index is expected to line up with the owning draw list's skinned
    /// mesh order.
    pub(crate) fn add_player(&mut self, skin: &str, default: Option<AnimationClipId>) -> usize {
        if let Some(clip) = default {
            let clip_skin = self.clip(clip).skin();
            assert_eq!(
                clip_skin, skin,
                "default clip '{}' drives skin '{clip_skin}', not '{skin}'",
                self.clip(clip).asset_name()
            );
        }

        let index = self.players.len();
        self.players.push(AnimationPlayer::new(skin, default));
        index
    }

    /// Advance every active player by `dt` seconds of wall-clock time.
    ///
    /// `dt` is clamped to [`MAX_ANIMATION_DELTA`].
    /// 
    /// The manager can handle pausing and scaling. A paused manager advances
    /// nothing, and [`Self::set_time_scale`] multiplies every player's local
    /// speed.
    ///
    /// Panics if `dt` is negative or not finite.
    pub fn update(&mut self, dt: f32) {
        assert!(dt.is_finite(), "animation delta time must be finite, got {dt}");
        assert!(dt >= 0.0, "animation delta time must not be negative, got {dt}");

        if self.paused {
            return;
        }
        let dt = dt.min(MAX_ANIMATION_DELTA) * self.time_scale;
        if dt <= 0.0 {
            return;
        }

        let clips = &self.clips;
        for player in &mut self.players {
            let Some(clip_id) = player.clip else { continue };
            let clip = &clips[clip_id.index()];
            let delta = dt * player.speed;
            player.advance(delta, clip);
        }
    }

    /// Pause or resume every player at once.
    pub fn set_paused(&mut self, paused: bool) { self.paused = paused; }
    pub fn is_paused(&self) -> bool { self.paused }

    /// Global time scale applied before each player's own speed.
    ///
    /// Panics unless `scale` is finite and non-negative.
    pub fn set_time_scale(&mut self, scale: f32) {
        assert!(
            scale.is_finite() && scale >= 0.0,
            "animation time scale must be finite and non-negative, got {scale}"
        );
        self.time_scale = scale;
    }
    pub fn time_scale(&self) -> f32 { self.time_scale }

    /// Every registered clip, in registration order.
    pub fn clips(&self) -> &[AnimationClip] { &self.clips }

    /// Clip metadata by id.
    ///
    /// Panics if `id` is out of range.
    pub fn clip(&self, id: AnimationClipId) -> &AnimationClip {
        self.clips.get(id.index()).unwrap_or_else(|| {
            panic!(
                "animation clip index {} is out of range (clip count {})",
                id.index(),
                self.clips.len()
            )
        })
    }

    /// Look up a clip by its asset name.
    pub fn find_clip(&self, asset_name: &str) -> Option<AnimationClipId> {
        self.clip_by_asset.get(asset_name).copied()
    }

    /// Clips registered for `skin`, in registration order.
    pub fn clips_for_skin(&self, skin: &str) -> &[AnimationClipId] {
        self.clips_by_skin.get(skin).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn player_count(&self) -> usize { self.players.len() }

    /// Player state by draw-list index.
    ///
    /// Panics if `index` is out of range.
    pub fn player(&self, index: usize) -> &AnimationPlayer {
        self.players.get(index).unwrap_or_else(|| {
            panic!(
                "animation player index {index} is out of range (player count {})",
                self.players.len()
            )
        })
    }

    fn player_mut(&mut self, index: usize) -> &mut AnimationPlayer {
        let count = self.players.len();
        self.players.get_mut(index).unwrap_or_else(|| {
            panic!("animation player index {index} is out of range (player count {count})")
        })
    }

    /// Resolve the skinning input for one player.
    ///
    /// Returns `None` when the player has no clip (the mesh renders its rest
    /// pose) or when `index` is out of range.
    pub fn skinned_frame(&self, index: usize) -> Option<SkinnedAnimationFrame> {
        let player = self.players.get(index)?;
        let clip_id = player.clip?;
        let clip = self.clip(clip_id);
        Some(SkinnedAnimationFrame::new(
            clip.texture(),
            clip.frame_at(player.time, player.loop_mode),
        ))
    }

    /// Select a clip and restart playback from time zero.
    ///
    /// Panics if the clip belongs to a different skin than the player, or if
    /// either id is out of range.
    pub fn play(&mut self, index: usize, clip: AnimationClipId) {
        let clip_skin = self.clip(clip).skin().to_string();
        let player = self.player_mut(index);
        assert_eq!(
            player.skin, clip_skin,
            "cannot play a clip for skin '{clip_skin}' on a '{}' player",
            player.skin
        );

        player.clip = Some(clip);
        player.time = 0.0;
        player.paused = false;
        player.finished = false;
    }

    /// Select a clip by asset name.
    ///
    /// Returns `false` (and logs a warning) when the name is unknown or the
    /// clip belongs to another skin.
    pub fn play_by_name(&mut self, index: usize, asset_name: &str) -> bool {
        let player_skin = self.player(index).skin().to_string();
        match self.find_clip(asset_name) {
            Some(clip) if self.clip(clip).skin() == player_skin => {
                self.play(index, clip);
                true
            }
            Some(clip) => {
                log::warn!(
                    "Animation clip '{asset_name}' drives skin '{}', not '{player_skin}'.",
                    self.clip(clip).skin()
                );
                false
            }
            None => {
                log::warn!("Animation clip '{asset_name}' was not registered.");
                false
            }
        }
    }

    pub fn pause(&mut self, index: usize) { self.player_mut(index).paused = true; }
    pub fn resume(&mut self, index: usize) { self.player_mut(index).paused = false; }

    /// Restart the current clip from time zero without changing the selection.
    pub fn restart(&mut self, index: usize) {
        let player = self.player_mut(index);
        player.time = 0.0;
        player.finished = false;
        player.paused = false;
    }

    /// Jump to `time`, normalizing it for the player's loop mode.
    ///
    /// Panics if `time` is not finite.
    pub fn seek(&mut self, index: usize, time: f32) {
        assert!(time.is_finite(), "animation seek time must be finite, got {time}");

        let (clip, mode) = {
            let player = self.player(index);
            (player.clip, player.loop_mode)
        };
        let Some(clip_id) = clip else {
            self.player_mut(index).time = 0.0;
            return;
        };
        let (duration, end) = {
            let clip = self.clip(clip_id);
            (clip.duration(), clip.last_unique_time())
        };

        let player = self.player_mut(index);
        match mode {
            LoopMode::Loop => {
                player.time = time.rem_euclid(duration);
                player.finished = false;
            }
            LoopMode::Once => {
                player.time = time.clamp(0.0, end);
                player.finished = player.time >= end;
            }
        }
    }

    /// Set one player's local time scale.
    ///
    /// Panics unless `speed` is finite and non-negative.
    pub fn set_player_speed(&mut self, index: usize, speed: f32) {
        assert!(
            speed.is_finite() && speed >= 0.0,
            "animation player speed must be finite and non-negative, got {speed}"
        );
        self.player_mut(index).speed = speed;
    }

    /// Switch one player between looping and one-shot playback.
    pub fn set_player_loop_mode(&mut self, index: usize, mode: LoopMode) {
        let end = {
            let player = self.player(index);
            player.clip.map(|clip| self.clip(clip).last_unique_time())
        };

        let player = self.player_mut(index);
        if player.loop_mode == mode {
            return;
        }
        player.loop_mode = mode;

        match mode {
            LoopMode::Loop => {
                player.finished = false;
            }
            LoopMode::Once => {
                if let Some(end) = end {
                    if player.time >= end {
                        player.time = end;
                        player.finished = true;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn clip(name: &str, skin: &str, frames: u32, fps: f32) -> AnimationClip {
        AnimationClip::new(
            name.to_string(),
            skin.to_string(),
            SampledTextureHandle::from_index(0),
            frames,
            fps,
        )
    }

    fn approx(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-4,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn loop_frames_wrap_and_stay_inside_the_texture() {
        let clip = clip("clip", "skin", 30, 30.0);

        approx(clip.frame_at(0.0, LoopMode::Loop), 0.0);
        approx(clip.frame_at(0.5, LoopMode::Loop), 15.0);
        approx(clip.frame_at(1.25, LoopMode::Loop), 7.5);
        approx(clip.frame_at(-0.5, LoopMode::Loop), 15.0);
        approx(clip.frame_at(1.0, LoopMode::Loop), 0.0);

        for step in 0..1_000 {
            let time = step as f32 * 0.0173;
            let frame = clip.frame_at(time, LoopMode::Loop);
            assert!((0.0..30.0).contains(&frame), "frame {frame} out of range at {time}");
        }
    }

    #[test]
    fn once_frames_stop_on_the_last_unique_row() {
        let clip = clip("clip", "skin", 30, 30.0);
        let last = clip.frame_at(10.0, LoopMode::Once);

        approx(last, 29.0);
        assert!(last <= 29.0);
        approx(clip.last_unique_time(), 29.0 / 30.0);
    }

    #[test]
    fn update_advances_looping_players_and_clamps_dt() {
        let mut manager = AnimationManager::new();
        let id = manager.register_clip(clip("a", "skin", 30, 30.0));
        manager.add_player("skin", Some(id));

        manager.update(0.1);
        approx(manager.player(0).time(), 0.1);

        // A stalled frame must not teleport the animation.
        manager.update(10.0);
        approx(manager.player(0).time(), 0.35);

        // And looping wraps around.
        for _ in 0..10 {
            manager.update(0.1);
        }
        approx(manager.player(0).time(), 0.35);
    }

    #[test]
    fn pause_freeze_and_speed_scale_are_independent() {
        let mut manager = AnimationManager::new();
        let id = manager.register_clip(clip("a", "skin", 30, 30.0));
        manager.add_player("skin", Some(id));

        manager.pause(0);
        manager.update(0.1);
        approx(manager.player(0).time(), 0.0);

        manager.resume(0);
        manager.set_player_speed(0, 2.0);
        manager.update(0.1);
        approx(manager.player(0).time(), 0.2);

        manager.set_paused(true);
        manager.update(0.1);
        approx(manager.player(0).time(), 0.2);

        manager.set_paused(false);
        manager.set_time_scale(0.5);
        manager.update(0.1);
        approx(manager.player(0).time(), 0.3);
    }

    #[test]
    fn once_players_finish_and_hold_the_last_frame() {
        let mut manager = AnimationManager::new();
        let id = manager.register_clip(clip("a", "skin", 30, 30.0));
        manager.add_player("skin", Some(id));
        manager.set_player_loop_mode(0, LoopMode::Once);

        for _ in 0..20 {
            manager.update(0.1);
        }

        let player = manager.player(0);
        assert!(player.is_finished());
        approx(player.time(), 29.0 / 30.0);
        approx(manager.skinned_frame(0).expect("player has a clip").frame(), 29.0);
    }

    #[test]
    fn switching_clips_restarts_and_rejects_other_skins() {
        let mut manager = AnimationManager::new();
        let first = manager.register_clip(clip("a", "skin", 30, 30.0));
        let second = manager.register_clip(clip("b", "skin", 60, 30.0));
        let foreign = manager.register_clip(clip("c", "other", 30, 30.0));
        manager.add_player("skin", Some(first));
        manager.add_player("skin", Some(first));

        manager.update(0.5);
        manager.play(0, second);
        assert_eq!(manager.player(0).clip(), Some(second));
        approx(manager.player(0).time(), 0.0);

        manager.update(0.5);
        assert_eq!(manager.player(1).clip(), Some(first));
        approx(manager.player(1).time(), 0.5);

        assert!(!manager.play_by_name(0, "c"));
        assert!(!manager.play_by_name(0, "missing"));
        assert!(manager.play_by_name(0, "a"));
        assert_eq!(manager.player(0).clip(), Some(first));

        // Cross-skin selection through the id API is a programmer error.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            manager.play(0, foreign);
        }));
        assert!(result.is_err());
    }

    #[test]
    fn clips_are_grouped_by_skin_in_registration_order() {
        let mut manager = AnimationManager::new();
        manager.register_clip(clip("a1", "skin", 30, 30.0));
        manager.register_clip(clip("a0", "skin", 30, 30.0));
        manager.register_clip(clip("b0", "other", 30, 30.0));
        manager.add_player("skin", None);

        let skin = manager.clips_for_skin("skin");
        assert_eq!(skin.len(), 2);
        assert_eq!(manager.clip(skin[0]).asset_name(), "a1");
        assert_eq!(manager.clip(skin[1]).asset_name(), "a0");
        assert_eq!(manager.clips_for_skin("other").len(), 1);
        assert!(manager.clips_for_skin("missing").is_empty());
        assert_eq!(manager.find_clip("a0"), Some(skin[1]));
        assert!(manager.skinned_frame(0).is_none());
    }

    #[test]
    fn seek_normalizes_for_the_loop_mode() {
        let mut manager = AnimationManager::new();
        let id = manager.register_clip(clip("a", "skin", 30, 30.0));
        manager.add_player("skin", Some(id));

        manager.seek(0, 1.25);
        approx(manager.player(0).time(), 0.25);

        manager.set_player_loop_mode(0, LoopMode::Once);
        manager.seek(0, 99.0);
        approx(manager.player(0).time(), 29.0 / 30.0);
        assert!(manager.player(0).is_finished());

        manager.set_player_loop_mode(0, LoopMode::Loop);
        assert!(!manager.player(0).is_finished());
    }
}
