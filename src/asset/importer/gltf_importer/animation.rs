//! Baking of glTF animation channels into 2D textures.
//!
//! The renderer never evaluates glTF animation curves at runtime. Every frame of
//! an animation is sampled here, the joint hierarchy is resolved on the CPU, and
//! the resulting object-space skinning matrices are packed into an
//! `Rgba16Float` texture: one row per frame, four texels per joint, holding the
//! four columns of that joint's matrix.
//!
//! A glTF animation may drive morph target weights instead of (or in addition
//! to) node transforms. Those channels are baked into a separate weight texture
//! with the same frame count, so both tracks of one clip stay in sync.

use std::{collections::{HashMap, HashSet}, num::NonZero};

use cgmath::{InnerSpace, Matrix4, Quaternion, SquareMatrix, Vector3};

use crate::asset::asset_types::{
    animation_asset::BakedTrack,
    texture_asset::{Sampler, TexelData, TextureAsset, TextureDimension::D2, TextureFormat},
};
use crate::util::f16::f32_to_f16_bits;

use super::common::GltfImportError;
use super::scene::SkinUsage;

/// Rate at which animation frames are baked.
pub(super) const BAKE_FPS: f32 = 30.0;

/// Upper bound on baked frames, so a malformed animation cannot allocate an
/// enormous texture.
const MAX_FRAMES: u32 = 4096;

/// Import source identifier of one glTF animation inside one glTF scene.
///
/// Every track of one glTF animation belongs to the same [`AnimationAsset`], and
/// this string is that asset's clip identity.
///
/// [`AnimationAsset`]: crate::asset::asset_types::animation_asset::AnimationAsset
pub(super) fn clip_source(scene: usize, animation: usize) -> String {
    format!("scene{scene}/anim{animation}")
}

/// Longest input time among *all* channels of an animation, in seconds.
///
/// glTF allows the channels of one animation to end at different times, so this
/// is a maximum rather than an equality assertion.
pub(super) fn animation_duration(animation: &gltf::Animation, buffers: &[gltf::buffer::Data]) -> f32 {
    let mut duration = 0.0f32;
    for channel in animation.channels() {
        let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));
        if let Some(last) = reader.read_inputs().and_then(|times| times.last()) {
            duration = duration.max(last);
        }
    }
    duration
}

/// Number of baked frames of an animation that lasts `duration` seconds.
pub(super) fn frame_count_for(duration: f32, fps: f32, max_frames: u32) -> u32 {
    let frames = ((duration * fps).ceil() as u32).saturating_add(1);
    if frames > max_frames {
        log::warn!("Animation is {duration}s long; clamping the bake to {max_frames} frames.");
        return max_frames;
    }
    frames
}

/// Number of frames every track of one clip is baked at.
///
/// Exposed so the importer can stamp the count on the [`AnimationAsset`] without
/// having to re-derive it from a baked texture.
///
/// [`AnimationAsset`]: crate::asset::asset_types::animation_asset::AnimationAsset
pub(super) fn clip_frame_count(duration: f32) -> u32 {
    frame_count_for(duration, BAKE_FPS, MAX_FRAMES)
}

/// Split a flat morph weight sequence into one entry per keyframe.
///
/// glTF stores a weights channel as `keyframe_count * target_count` floats, with
/// all targets of one keyframe adjacent. A trailing partial keyframe is dropped.
pub(super) fn split_weights(flat: &[f32], target_count: usize) -> Vec<Vec<f32>> {
    if target_count == 0 {
        return Vec::new();
    }
    if flat.len() % target_count != 0 {
        log::warn!(
            "Morph weight channel has {} values, which is not a multiple of {} targets; the trailing partial keyframe is dropped.",
            flat.len(),
            target_count
        );
    }
    flat.chunks_exact(target_count).map(<[f32]>::to_vec).collect()
}

/// One mesh whose morph weights an animation drives.
pub(super) struct MorphWeightTrack<'a> {
    /// Index of the glTF mesh the weights channel targets.
    pub mesh: usize,
    /// The weights channel; every morph primitive of `mesh` is baked from it.
    pub channel: gltf::animation::Channel<'a>,
}

/// What one glTF animation drives.
#[derive(Default)]
pub(super) struct AnimationClassification<'a> {
    /// Skin whose joints the transform channels animate.
    pub skin: Option<usize>,
    /// Meshes with an animated weights channel, deduplicated, first-seen order.
    pub morph_tracks: Vec<MorphWeightTrack<'a>>,
    /// Scene the animation is scoped to, used to build the clip `source`.
    pub scene: Option<usize>,
}

/// Decide what a glTF animation drives.
///
/// Transform channels bind the animation to the first skin that owns one of the
/// animated nodes; weights channels bind it to the mesh of the animated node.
/// A weights-only animation therefore gets no skin, but still produces assets.
pub(super) fn classify_animation<'a>(
    animation: &gltf::Animation<'a>,
    document: &gltf::Document,
    skin_usage: &HashMap<usize, SkinUsage>,
) -> AnimationClassification<'a> {
    let mut animated_nodes = HashSet::new();
    let mut morph_tracks: Vec<MorphWeightTrack<'a>> = Vec::new();
    let mut morph_meshes = HashSet::new();

    for channel in animation.channels() {
        // glTF channels always target a node; `Target::node` panics otherwise.
        let node = channel.target().node();
        if !matches!(channel.target().property(), gltf::animation::Property::MorphTargetWeights) {
            animated_nodes.insert(node.index());
            continue;
        }

        let Some(mesh) = node.mesh() else {
            log::warn!(
                "Animation '{}' animates morph weights of node {} which has no mesh; the channel is ignored.",
                animation.name().unwrap_or("<unnamed>"),
                node.index()
            );
            continue;
        };
        if !morph_meshes.insert(mesh.index()) {
            log::warn!(
                "Animation '{}' drives morph weights of mesh {} from several nodes; only the first channel is baked.",
                animation.name().unwrap_or("<unnamed>"),
                mesh.index()
            );
            continue;
        }

        morph_tracks.push(MorphWeightTrack { mesh: mesh.index(), channel });
    }

    // A glTF animation may drive joints of more than one skin.
    // Issue a warning and degrade gracefully.
    let animated_skins: Vec<usize> = if animated_nodes.is_empty() {
        Vec::new()
    } else {
        document
            .skins()
            .filter(|skin| skin.joints().any(|joint| animated_nodes.contains(&joint.index())))
            .map(|skin| skin.index())
            .collect()
    };
    if animated_skins.len() > 1 {
        log::warn!(
            "Animation '{}' drives {} skins ({animated_skins:?}); only skin {} is baked, the rest stay unanimated.",
            animation.name().unwrap_or("<unnamed>"),
            animated_skins.len(),
            animated_skins[0]
        );
    }
    let skin = animated_skins.first().copied();

    // Both tracks of one animation must end up with the same `source`, so the
    // scene is resolved once: from the skin when there is one, otherwise from
    // the node that carries the morph weights.
    let scene = skin
        .and_then(|skin| skin_usage.get(&skin).map(|usage| usage.scene))
        .or_else(|| {
            morph_tracks.first()
                .and_then(|track| super::scene::scene_of_node(document, track.channel.target().node().index()))
        });

    AnimationClassification { skin, morph_tracks, scene }
}

/// Local rest transform of one node, in TRS form.
#[derive(Clone, Copy)]
struct NodeTransform {
    translation: Vector3<f32>,
    rotation: Quaternion<f32>,
    scale: Vector3<f32>,
}

impl NodeTransform {
    fn from_gltf(transform: gltf::scene::Transform) -> Self {
        let (translation, rotation, scale) = transform.decomposed();
        Self {
            translation: Vector3::from(translation),
            // glTF stores quaternions as (x, y, z, w).
            rotation: Quaternion::new(rotation[3], rotation[0], rotation[1], rotation[2]),
            scale: Vector3::from(scale),
        }
    }

    fn to_matrix(self) -> Matrix4<f32> {
        Matrix4::from_translation(self.translation)
            * Matrix4::from(self.rotation)
            * Matrix4::from_nonuniform_scale(self.scale.x, self.scale.y, self.scale.z)
    }
}

/// Keyframe values of one channel.
enum ChannelValues {
    Translation(Vec<[f32; 3]>),
    Rotation(Vec<[f32; 4]>),
    Scale(Vec<[f32; 3]>),
}

/// One sampled glTF animation channel.
struct Channel {
    node: usize,
    interpolation: gltf::animation::Interpolation,
    times: Vec<f32>,
    values: ChannelValues,
}

/// Value of one channel at one point in time.
enum Sample {
    Translation(Vector3<f32>),
    Rotation(Quaternion<f32>),
    Scale(Vector3<f32>),
}

impl Channel {
    fn sample(&self, time: f32) -> Sample {
        let (first, second, amount) = segment(&self.times, time);
        let step = self.interpolation == gltf::animation::Interpolation::Step;

        match &self.values {
            ChannelValues::Translation(values) => {
                let a = Vector3::from(values[first]);
                if step || first == second {
                    Sample::Translation(a)
                } else {
                    Sample::Translation(a + (Vector3::from(values[second]) - a) * amount)
                }
            }
            ChannelValues::Rotation(values) => {
                let a = to_quaternion(values[first]);
                if step || first == second {
                    Sample::Rotation(a)
                } else {
                    Sample::Rotation(slerp(a, to_quaternion(values[second]), amount))
                }
            }
            ChannelValues::Scale(values) => {
                let a = Vector3::from(values[first]);
                if step || first == second {
                    Sample::Scale(a)
                } else {
                    Sample::Scale(a + (Vector3::from(values[second]) - a) * amount)
                }
            }
        }
    }
}

fn to_quaternion(value: [f32; 4]) -> Quaternion<f32> {
    // glTF stores quaternions as (x, y, z, w).
    Quaternion::new(value[3], value[0], value[1], value[2])
}

/// Locate the keyframe interval containing `time`.
///
/// Returns `(first, second, amount)` where `amount` is the normalized position
/// between the two keyframes. Both indices are equal when `time` is clamped to
/// the ends of the track.
fn segment(times: &[f32], time: f32) -> (usize, usize, f32) {
    let last = times.len().saturating_sub(1);
    if last == 0 || time <= times[0] {
        return (0, 0, 0.0);
    }
    if time >= times[last] {
        return (last, last, 0.0);
    }

    let second = times.partition_point(|&t| t <= time);
    let first = second.saturating_sub(1);
    let span = times[second] - times[first];
    let amount = if span > f32::EPSILON {
        (time - times[first]) / span
    } else {
        0.0
    };
    (first, second, amount)
}

/// Shortest-path spherical interpolation between two rotations.
/// 
/// We can use cgmath slerp, but it seems to be buggy:
/// https://github.com/rustgd/cgmath/issues/300
fn slerp(a: Quaternion<f32>, b: Quaternion<f32>, amount: f32) -> Quaternion<f32> {
    let mut b = b;
    let mut dot = a.dot(b);
    if dot < 0.0 {
        b = -b;
        dot = -dot;
    }

    if dot > 0.9995 {
        // Nearly identical: nlerp is stable and avoids a division by ~0.
        return (a + (b - a) * amount).normalize();
    }

    let theta = dot.clamp(-1.0, 1.0).acos();
    let sin_theta = theta.sin();
    let weight_a = ((1.0 - amount) * theta).sin() / sin_theta;
    let weight_b = (amount * theta).sin() / sin_theta;
    a * weight_a + b * weight_b
}

/// Build the parent map of the whole node hierarchy.
fn build_parents(document: &gltf::Document) -> Vec<Option<usize>> {
    let mut parents = vec![None; document.nodes().len()];
    for node in document.nodes() {
        for child in node.children() {
            parents[child.index()] = Some(node.index());
        }
    }
    parents
}

/// Resolve the world matrix of one node, memoizing along the parent chain.
fn world_matrix(
    index: usize,
    locals: &[Matrix4<f32>],
    parents: &[Option<usize>],
    cache: &mut [Option<Matrix4<f32>>],
) -> Matrix4<f32> {
    if let Some(world) = cache[index] {
        return world;
    }

    // Walk up to the first cached ancestor, then compose back down.
    let mut chain = Vec::new();
    let mut current = index;
    let mut world = loop {
        if let Some(world) = cache[current] {
            break world;
        }
        chain.push(current);
        match parents[current] {
            Some(parent) => current = parent,
            None => break Matrix4::from_scale(1.0),
        }
    };

    for &node in chain.iter().rev() {
        world = world * locals[node];
        cache[node] = Some(world);
    }
    world
}

/// Resolve every joint matrix of one pose.
///
/// `jointMatrix = inverse(mesh_world) * joint_world * inverse_bind_matrix`, so
/// the result maps a vertex from the mesh's local space to the mesh's local
/// space; the rasterizer still applies the mesh's model matrix.
fn bake_frame(
    locals: &[Matrix4<f32>],
    parents: &[Option<usize>],
    joint_indices: &[usize],
    inverse_bind_matrices: &[[[f32; 4]; 4]],
    inverse_mesh_world: &Matrix4<f32>,
) -> Vec<[[f32; 4]; 4]> {
    let mut cache = vec![None; locals.len()];

    joint_indices
        .iter()
        .zip(inverse_bind_matrices)
        .map(|(&joint, inverse_bind_matrix)| {
            let world = world_matrix(joint, locals, parents, &mut cache);
            let matrix = *inverse_mesh_world
                * world
                * Matrix4::from(*inverse_bind_matrix);
            *AsRef::<[[f32; 4]; 4]>::as_ref(&matrix)
        })
        .collect()
}

/// Read every usable transform channel of an animation.
///
/// Morph weight channels are baked separately by [`bake_weight_track`].
/// Channels without keyframe times are skipped.
fn read_channels(animation: &gltf::Animation, buffers: &[gltf::buffer::Data]) -> Result<Vec<Channel>, GltfImportError> {
    use gltf::animation::util::ReadOutputs;

    let mut channels = Vec::new();

    for channel in animation.channels() {
        // glTF channels always target a node; `Target::node` panics otherwise.
        let node = channel.target().node();
        let sampler = channel.sampler();
        let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));

        let Some(times) = reader.read_inputs() else { continue };
        let times: Vec<f32> = times.collect();
        if times.is_empty() {
            continue;
        }

        let Some(outputs) = reader.read_outputs() else { continue };
        let values = match outputs {
            ReadOutputs::Translations(iter) => ChannelValues::Translation(iter.collect()),
            ReadOutputs::Rotations(rotations) => ChannelValues::Rotation(rotations.into_f32().collect()),
            ReadOutputs::Scales(iter) => ChannelValues::Scale(iter.collect()),
            ReadOutputs::MorphTargetWeights(_) => continue,
        };

        let is_cubic = sampler.interpolation() == gltf::animation::Interpolation::CubicSpline;
        if is_cubic {
            return Err(GltfImportError::UnsupportedFeature("cubic interpolation"))
        }
        let interpolation = sampler.interpolation();

        channels.push(Channel { node: node.index(), interpolation, times, values });
    }

    Ok(channels)
}

/// Bake the joint track of one glTF animation into a joint-matrix texture.
///
/// `skin` supplies the joint order and inverse bind matrices; `mesh_node_index`
/// is the node the skinned mesh is attached to, whose *rest* world transform is
/// removed so the result stays in mesh space.
///
/// `duration` is the animation's duration, shared with the weight tracks so that
/// every track of one clip produces the same frame count.
pub(super) fn bake_joint_track(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    animation: &gltf::Animation,
    skin: &gltf::Skin,
    inverse_bind_matrices: &[[[f32; 4]; 4]],
    mesh_node_index: usize,
    skin_name: String,
    duration: f32,
    fps: f32,
) -> Result<BakedTrack, GltfImportError> {
    let joint_indices: Vec<usize> = skin.joints().map(|node| node.index()).collect();
    if joint_indices.is_empty() {
        return Err(GltfImportError::MissingSkinningAttributes);
    }
    if inverse_bind_matrices.len() != joint_indices.len() {
        return Err(GltfImportError::SkinJointCountMismatch {
            joints: joint_indices.len(),
            matrices: inverse_bind_matrices.len(),
        });
    }

    let channels = read_channels(animation, buffers)?;
    let parents = build_parents(document);
    let rest: Vec<NodeTransform> = document.nodes().map(|node| NodeTransform::from_gltf(node.transform())).collect();

    // The mesh node's rest transform is removed from every joint matrix.
    let mut cache = vec![None; rest.len()];
    let rest_locals: Vec<Matrix4<f32>> = rest.iter().map(|t| t.to_matrix()).collect();
    let mesh_world = world_matrix(mesh_node_index, &rest_locals, &parents, &mut cache);

    // this inverse matrix is needed in case the root of the mesh is not identical to the origin.
    let inverse_mesh_world = mesh_world.invert().ok_or(GltfImportError::NonInvertibleMeshTransform)?;

    let frame_count = frame_count_for(duration, fps, MAX_FRAMES);

    let joint_count = joint_indices.len();
    // Four texels per joint, four `f16` channels per texel.
    let mut texels = Vec::with_capacity(joint_count * 4 * frame_count as usize * 8);

    for frame in 0..frame_count {
        let time = frame as f32 / fps;

        // Start from the rest pose and override the animated properties.
        let mut pose = rest.clone();
        for channel in &channels {
            let target = &mut pose[channel.node];
            match channel.sample(time) {
                Sample::Translation(value) => target.translation = value,
                Sample::Rotation(value) => target.rotation = value,
                Sample::Scale(value) => target.scale = value,
            }
        }

        let locals: Vec<Matrix4<f32>> = pose.iter().map(|transform| transform.to_matrix()).collect();
        let matrices = bake_frame(&locals, &parents, &joint_indices, inverse_bind_matrices, &inverse_mesh_world);

        for matrix in matrices {
            for column in matrix {
                for channel in column {
                    texels.extend_from_slice(&f32_to_f16_bits(channel).to_le_bytes());
                }
            }
        }
    }
    // Append a final frame 0 for wrapping around gracefully
    {
        let time = 0 as f32 / fps;
        // Start from the rest pose and override the animated properties.
        let mut pose = rest.clone();
        for channel in &channels {
            let target = &mut pose[channel.node];
            match channel.sample(time) {
                Sample::Translation(value) => target.translation = value,
                Sample::Rotation(value) => target.rotation = value,
                Sample::Scale(value) => target.scale = value,
            }
        }

        let locals: Vec<Matrix4<f32>> = pose.iter().map(|transform| transform.to_matrix()).collect();
        let matrices = bake_frame(&locals, &parents, &joint_indices, inverse_bind_matrices, &inverse_mesh_world);

        for matrix in matrices {
            for column in matrix {
                for channel in column {
                    texels.extend_from_slice(&f32_to_f16_bits(channel).to_le_bytes());
                }
            }
        }
    }

    let texture = TextureAsset {
        format: TextureFormat::Rgba16Float,
        dimension: D2,
        width: NonZero::new((joint_count * 4) as u32)
            .ok_or(GltfImportError::MissingSkinningAttributes)?,
        height: Some(NonZero::new(frame_count + 1).ok_or(GltfImportError::MissingSkinningAttributes)?),
        depth_or_array_slice: None,
        sampler: Sampler::default(),
        texels: TexelData::Plain(texels.into_boxed_slice()),
    };

    Ok(BakedTrack::Joints { skin: skin_name, texture })
}

/// Sample one keyframe of morph weights.
///
/// `keyframes` holds one entry per baked keyframe, each with `target_count`
/// weights. Missing values (a truncated or short channel) read as zero, which is
/// the glTF default weight.
fn sample_weights(
    times: &[f32],
    keyframes: &[Vec<f32>],
    target_count: usize,
    time: f32,
    step: bool,
) -> Vec<f32> {
    let mut weights = vec![0.0f32; target_count];
    if keyframes.is_empty() || times.is_empty() {
        return weights;
    }

    let (first, second, amount) = segment(times, time);
    let first = first.min(keyframes.len() - 1);
    let second = second.min(keyframes.len() - 1);
    let a = &keyframes[first];
    let b = &keyframes[second];

    for (target, slot) in weights.iter_mut().enumerate() {
        let a = a.get(target).copied().unwrap_or(0.0);
        *slot = if step || first == second {
            a
        } else {
            let b = b.get(target).copied().unwrap_or(0.0);
            a + (b - a) * amount
        };
    }

    weights
}

/// Append one texture row of morph weights.
///
/// Weights of up to four targets share a texel, so the row is `width` RGBA16F
/// texels wide; targets beyond `weights` are zero.
fn write_weight_row(texels: &mut Vec<u8>, weights: &[f32], width: u32) {
    for x in 0..width as usize {
        for channel in 0..4usize {
            let value = weights.get(x * 4 + channel).copied().unwrap_or(0.0);
            texels.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
        }
    }
}

/// Bake the weight track of one glTF animation into a weight texture.
///
/// `duration` is the animation's duration, shared with every other track of the
/// clip so they all produce the same frame count. `target_count` is the number of
/// targets the owning morph primitive was imported with.
///
/// The texture is `width = ceil(target_count / 4)` texels by
/// `height = frames + 1` rows: one row per frame plus the wrap-around row that
/// repeats frame 0, exactly like the joint-matrix texture.
pub(super) fn bake_weight_track(
    channel: &gltf::animation::Channel<'_>,
    buffers: &[gltf::buffer::Data],
    target_count: usize,
    duration: f32,
    fps: f32,
    mesh: String,
    target: String,
) -> Result<BakedTrack, GltfImportError> {
    use gltf::animation::util::ReadOutputs;

    let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));

    let times: Vec<f32> = reader.read_inputs()
        .ok_or(GltfImportError::MalformedWeightChannel("no keyframe times"))?
        .collect();
    if times.is_empty() {
        return Err(GltfImportError::MalformedWeightChannel("empty keyframe times"));
    }

    let flat: Vec<f32> = match reader.read_outputs() {
        Some(ReadOutputs::MorphTargetWeights(weights)) => weights.into_f32().collect(),
        _ => return Err(GltfImportError::MalformedWeightChannel("outputs are not morph target weights")),
    };

    let interpolation = channel.sampler().interpolation();
    if interpolation == gltf::animation::Interpolation::CubicSpline {
        return Err(GltfImportError::UnsupportedFeature("cubic interpolation"));
    }
    let step = interpolation == gltf::animation::Interpolation::Step;

    let keyframes = split_weights(&flat, target_count);
    let frame_count = frame_count_for(duration, fps, MAX_FRAMES);
    // At least one texel per row, even for a target-less channel.
    let width = target_count.div_ceil(4).max(1) as u32;

    let mut texels = Vec::with_capacity(frame_count as usize * width as usize * 8);
    for frame in 0..frame_count {
        let weights = sample_weights(&times, &keyframes, target_count, frame as f32 / fps, step);
        write_weight_row(&mut texels, &weights, width);
    }
    // Append frame 0 again, so the shader can interpolate into the next loop
    // without a modulo.
    let weights = sample_weights(&times, &keyframes, target_count, 0.0, step);
    write_weight_row(&mut texels, &weights, width);

    let texture = TextureAsset {
        format: TextureFormat::Rgba16Float,
        dimension: D2,
        width: NonZero::new(width).ok_or(GltfImportError::ZeroDimension)?,
        height: Some(NonZero::new(frame_count + 1).ok_or(GltfImportError::ZeroDimension)?),
        depth_or_array_slice: None,
        sampler: Sampler::default(),
        texels: TexelData::Plain(texels.into_boxed_slice()),
    };

    Ok(BakedTrack::MorphWeights { mesh, target, texture })
}

#[cfg(test)]
mod test {
    use super::*;

    fn assert_identity(matrix: &[[f32; 4]; 4]) {
        for column in 0..4 {
            for row in 0..4 {
                let expected = if column == row { 1.0 } else { 0.0 };
                assert!(
                    (matrix[column][row] - expected).abs() < 1e-5,
                    "expected identity, got {matrix:?}"
                );
            }
        }
    }

    #[test]
    fn both_tracks_share_the_frame_count_of_the_longest_channel() {
        // Two channels of one animation ending at different times. The tracks
        // must agree even though neither of them reaches the animation duration
        // on its own.
        let ends = [0.5f32, 2.0];
        let duration = ends.iter().copied().fold(0.0f32, f32::max);

        let skeletal = frame_count_for(duration, BAKE_FPS, MAX_FRAMES);
        let morph = frame_count_for(duration, BAKE_FPS, MAX_FRAMES);

        assert_eq!(skeletal, morph);
        assert_eq!(skeletal, 61, "ceil(2.0 * 30) + 1 rows");
        assert_ne!(
            frame_count_for(ends[0], BAKE_FPS, MAX_FRAMES),
            skeletal,
            "per-track durations would have desynchronized the tracks"
        );
    }

    #[test]
    fn frame_count_is_clamped() {
        assert_eq!(frame_count_for(0.0, BAKE_FPS, MAX_FRAMES), 1);
        assert_eq!(frame_count_for(f32::MAX, BAKE_FPS, MAX_FRAMES), MAX_FRAMES);
    }

    #[test]
    fn split_weights_groups_targets_by_keyframe() {
        let flat = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        assert_eq!(
            split_weights(&flat, 2),
            vec![vec![0.0, 1.0], vec![2.0, 3.0], vec![4.0, 5.0], vec![6.0, 7.0]]
        );
        assert_eq!(
            split_weights(&flat, 4),
            vec![vec![0.0, 1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0, 7.0]]
        );

        // A trailing partial keyframe is dropped instead of panicking.
        assert_eq!(split_weights(&[1.0, 2.0, 3.0], 2), vec![vec![1.0, 2.0]]);
        assert!(split_weights(&flat, 0).is_empty());
    }

    #[test]
    fn classifies_morph_only_and_skeletal_animations() {
        // The cube is driven purely by morph weights: no skin, one morph mesh.
        let (document, buffers, _) = gltf::import("resource/AnimatedMorphCube.glb")
            .expect("the morph cube should load");
        let animation = document.animations().next().expect("the cube has one animation");
        let classification = classify_animation(&animation, &document, &HashMap::new());
        assert_eq!(classification.skin, None, "weights-only animations drive no skin");
        assert_eq!(
            classification.morph_tracks.iter().map(|track| track.mesh).collect::<Vec<_>>(),
            vec![0]
        );
        assert_eq!(classification.scene, Some(0));
        assert!(animation_duration(&animation, &buffers) > 0.0);

        // The fox is driven purely by transforms: one skin, no morph track.
        let (document, buffers, _) = gltf::import("resource/Fox.glb")
            .expect("the fox should load");
        let skin_usage = crate::asset::importer::gltf_importer::GltfImporter::resolve_skin_usage(&document);
        assert!(document.animations().count() > 0, "the fox has baked animations");
        for animation in document.animations() {
            let classification = classify_animation(&animation, &document, &skin_usage);
            assert_eq!(classification.skin, Some(0), "the fox has a single skin");
            assert!(
                classification.morph_tracks.is_empty(),
                "the fox has no morph targets"
            );
            assert!(animation_duration(&animation, &buffers) > 0.0);
        }
    }

    #[test]
    fn segment_clamps_and_interpolates() {
        let times = [0.0, 1.0, 2.0];
        assert_eq!(segment(&times, -1.0), (0, 0, 0.0));
        assert_eq!(segment(&times, 0.0), (0, 0, 0.0));
        let (first, second, amount) = segment(&times, 0.25);
        assert_eq!((first, second), (0, 1));
        assert!((amount - 0.25).abs() < 1e-6);
        assert_eq!(segment(&times, 5.0), (2, 2, 0.0));
    }

    /// Rotation of `degrees` about +Z.
    fn rotation_z(degrees: f32) -> Quaternion<f32> {
        let half = (degrees * 0.5).to_radians();
        Quaternion::from_sv(half.cos(), Vector3::unit_z() * half.sin())
    }

    #[test]
    fn slerp_midpoint_is_half_the_angle() {
        let identity = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let half = rotation_z(90.0);
        let middle = slerp(identity, half, 0.5);
        let expected = rotation_z(45.0);
        assert!((middle.dot(expected).abs() - 1.0).abs() < 1e-5, "got {middle:?}, expected {expected:?}");
    }

    #[test]
    fn rest_pose_bakes_to_identity_matrices() {
        // A two-joint chain whose mesh node sits at the origin, so the rest pose
        // must produce identity skinning matrices.
        let root = Matrix4::from_translation(Vector3::new(1.0, 0.0, 0.0))
            * Matrix4::from_angle_z(cgmath::Deg(30.0));
        let child = Matrix4::from_translation(Vector3::new(0.0, 2.0, 0.0));
        let locals = vec![root, child];
        let parents = vec![None, Some(0)];

        let child_world = root * child;
        let inverse_bind_matrices = [
            *AsRef::<[[f32; 4]; 4]>::as_ref(&root.invert().unwrap()),
            *AsRef::<[[f32; 4]; 4]>::as_ref(&child_world.invert().unwrap()),
        ];

        let matrices = bake_frame(
            &locals,
            &parents,
            &[0, 1],
            &inverse_bind_matrices,
            &Matrix4::from_scale(1.0),
        );

        assert_identity(&matrices[0]);
        assert_identity(&matrices[1]);
    }

    #[test]
    fn mesh_world_is_removed_from_the_result() {
        // Same rest pose, but the mesh node sits at a non-trivial transform:
        // every joint matrix becomes the inverse of that transform.
        let mesh_world = Matrix4::from_translation(Vector3::new(3.0, -1.0, 2.0));
        let joint_world = Matrix4::from_translation(Vector3::new(1.0, 0.0, 0.0));
        let locals = vec![joint_world];
        let parents = vec![None];
        let inverse_bind_matrices = [*AsRef::<[[f32; 4]; 4]>::as_ref(&joint_world.invert().unwrap())];

        let matrices = bake_frame(
            &locals,
            &parents,
            &[0],
            &inverse_bind_matrices,
            &mesh_world.invert().unwrap(),
        );

        let expected = *AsRef::<[[f32; 4]; 4]>::as_ref(&mesh_world.invert().unwrap());
        for column in 0..4 {
            for row in 0..4 {
                assert!(
                    (matrices[0][column][row] - expected[column][row]).abs() < 1e-5,
                    "expected {expected:?}, got {:?}",
                    matrices[0]
                );
            }
        }
    }

    /// Targets beyond the first four share a second texel, so the baker must
    /// not clamp the row to one texel. This is the packing half of the
    /// multi-texel weight path the shader test exercises.
    #[test]
    fn weight_rows_span_multiple_texels() {
        let mut texels = Vec::new();
        write_weight_row(&mut texels, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 2);

        assert_eq!(texels.len(), 2 * 8, "two RGBA16F texels, four lanes each");
        assert!(
            texels[..8].iter().all(|byte| *byte == 0),
            "the first texel only carries targets 0..4"
        );

        // Target 7 is lane 3 of the second texel.
        let offset = 8 + 3 * 2;
        let bits = u16::from_le_bytes([texels[offset], texels[offset + 1]]);
        assert_eq!(bits, f32_to_f16_bits(1.0));
    }

    #[test]
    fn split_weights_handles_more_than_four_targets() {
        let flat: Vec<f32> = (0..16).map(|value| value as f32).collect();
        let keyframes = split_weights(&flat, 8);

        assert_eq!(keyframes.len(), 2);
        assert_eq!(keyframes[0], (0..8).map(|value| value as f32).collect::<Vec<_>>());
        assert_eq!(keyframes[1], (8..16).map(|value| value as f32).collect::<Vec<_>>());
    }
}
