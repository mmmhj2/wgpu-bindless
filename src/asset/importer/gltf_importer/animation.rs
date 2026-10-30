//! Baking of glTF animation channels into a 2D texture of joint matrices.
//!
//! The renderer never evaluates glTF animation curves at runtime. Every frame of
//! an animation is sampled here, the joint hierarchy is resolved on the CPU, and
//! the resulting object-space skinning matrices are packed into an
//! `Rgba16Float` texture: one row per frame, four texels per joint, holding the
//! four columns of that joint's matrix.

use std::num::NonZero;

use cgmath::{InnerSpace, Matrix4, Quaternion, Rotation, SquareMatrix, Vector3};

use crate::asset::asset_types::{
    animation_asset::SkinnedAnimationAsset,
    texture_asset::{Sampler, TexelData, TextureAsset, TextureDimension::D2, TextureFormat},
};

use super::common::GltfImportError;

/// Rate at which animation frames are baked.
pub(super) const BAKE_FPS: f32 = 30.0;

/// Upper bound on baked frames, so a malformed animation cannot allocate an
/// enormous texture.
const MAX_FRAMES: u32 = 4096;

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

/// Read every usable channel of an animation.
///
/// Morph target weights and channels without a target node are skipped.
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
            ReadOutputs::MorphTargetWeights(_) => return Err(GltfImportError::UnsupportedFeature("morph target")),
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

/// Bake one glTF animation into a joint-matrix texture.
///
/// `skin` supplies the joint order and inverse bind matrices; `mesh_node_index`
/// is the node the skinned mesh is attached to, whose *rest* world transform is
/// removed so the result stays in mesh space.
pub(super) fn bake_animation(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    animation: &gltf::Animation,
    skin: &gltf::Skin,
    inverse_bind_matrices: &[[[f32; 4]; 4]],
    mesh_node_index: usize,
    skin_name: String,
    fps: f32,
) -> Result<SkinnedAnimationAsset, GltfImportError> {
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

    let duration = channels
        .iter()
        .filter_map(|channel| channel.times.last().copied())
        .fold(0.0f32, f32::max);
    let mut frame_count = ((duration * fps).ceil() as u32).saturating_add(1);
    if frame_count > MAX_FRAMES {
        log::warn!("Animation '{}' is {duration}s long; clamping the bake to {MAX_FRAMES} frames.", animation.name().unwrap_or("<unnamed>"));
        frame_count = MAX_FRAMES;
    }

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

    let texture = TextureAsset {
        format: TextureFormat::Rgba16Float,
        dimension: D2,
        width: NonZero::new((joint_count * 4) as u32)
            .ok_or(GltfImportError::MissingSkinningAttributes)?,
        height: Some(NonZero::new(frame_count).ok_or(GltfImportError::MissingSkinningAttributes)?),
        depth_or_array_slice: None,
        sampler: Sampler::default(),
        texels: TexelData::Plain(texels.into_boxed_slice()),
    };

    Ok(SkinnedAnimationAsset { skin: skin_name, frames: frame_count, fps, texture })
}

/// Convert an `f32` to IEEE-754 binary16 bits.
///
/// Round-to-nearest (ties away from zero) is used; it is accurate enough for
/// matrix data and avoids pulling in a dependency for half floats.
/// 
/// See https://goals.rust-lang.org/2026/stabilizing-f16.html for more info.
/// Maybe we should extract this into a single utility module.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exponent == 0xFF {
        // Infinity or NaN.
        let payload = if mantissa == 0 { 0 } else { 0x0200 };
        return sign | 0x7C00 | payload;
    }

    // Re-bias from f32 (127) to f16 (15).
    let mut exponent = exponent - 127 + 15;

    if exponent >= 0x1F {
        return sign | 0x7C00;
    }

    if exponent <= 0 {
        // Subnormal half or zero.
        if exponent < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x0080_0000;
        let shift = 14 - exponent;
        let rounded = (mantissa + (1 << (shift - 1))) >> shift;
        return sign | rounded as u16;
    }

    let rounded = mantissa + 0x1000;
    if rounded & 0x0080_0000 != 0 {
        // Rounding carried into the exponent.
        exponent += 1;
        if exponent >= 0x1F {
            return sign | 0x7C00;
        }
        return sign | ((exponent as u16) << 10);
    }
    sign | ((exponent as u16) << 10) | ((rounded >> 13) as u16)
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
    fn f16_encoding_matches_known_values() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3C00);
        assert_eq!(f32_to_f16_bits(-1.0), 0xBC00);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f32_to_f16_bits(2.0), 0x4000);
        assert_eq!(f32_to_f16_bits(-2.0), 0xC000);
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
}
