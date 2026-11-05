//! Morph weight baking: weight texture layout, sampling against a CPU
//! reference, and the clip identity shared with the skeletal track.

#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use rust_renderer::asset::{
        asset_manager::AssetManager,
        asset_types::{AssetData, texture_asset::{TexelData, TextureFormat}},
        importer::gltf_importer::GltfImporter,
    };

    /// `resource/AnimatedMorphCube.glb`: Morph only, no skin
    const SOURCE_GLTF: &str = "resource/AnimatedMorphCube.glb";
    /// `resource/Fox.glb` : Skin only, no morph.
    const FOX_GLTF: &str = "resource/Fox.glb";
    const TARGET_COUNT: usize = 2;

    fn temp_workspace(source: &str, name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rust_renderer_bake_{name}"));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).ok();
        }
        std::fs::create_dir_all(&dir).unwrap();

        let file_name = std::path::Path::new(source)
            .file_name()
            .expect("source should have a file name");
        std::fs::copy(source, dir.join(file_name)).unwrap();
        dir
    }

    fn import(dir: &std::path::Path, file_name: &str) -> (AssetManager, PathBuf) {
        let glb = dir.join(file_name);
        let manager = AssetManager::new();
        manager.import(GltfImporter::new(&glb)).expect("glTF import should succeed");
        (manager, glb)
    }

    /// Decode one IEEE-754 binary16 value.
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = ((bits >> 10) & 0x1F) as i32;
        let mantissa = (bits & 0x3FF) as f32;

        let magnitude = if exponent == 0 {
            mantissa * 2f32.powi(-24)
        } else if exponent == 0x1F {
            if mantissa == 0.0 { f32::INFINITY } else { f32::NAN }
        } else {
            (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15)
        };
        sign * magnitude
    }

    /// Read one texel of a baked texture as four floats.
    fn texel_at(texels: &[u8], width: usize, x: usize, y: usize) -> [f32; 4] {
        let offset = (y * width + x) * 8;
        let mut value = [0.0f32; 4];
        for (channel, slot) in value.iter_mut().enumerate() {
            let bits = u16::from_le_bytes([texels[offset + channel * 2], texels[offset + channel * 2 + 1]]);
            *slot = f16_to_f32(bits);
        }
        value
    }

    /// Keyframe times, per-keyframe target weights and total duration of the
    /// cube's weights channel, read straight from the glTF.
    fn cpu_weights(glb: &std::path::Path) -> (Vec<f32>, Vec<[f32; 2]>, f32) {
        use gltf::animation::util::ReadOutputs;

        let (document, buffers, _images) = gltf::import(glb).expect("the source glTF should load");
        let animation = document.animations().next().expect("the cube has one animation");

        let mut duration = 0.0f32;
        for channel in animation.channels() {
            let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));
            if let Some(last) = reader.read_inputs().and_then(|times| times.last()) {
                duration = duration.max(last);
            }
        }

        let channel = animation
            .channels()
            .find(|channel| matches!(channel.target().property(), gltf::animation::Property::MorphTargetWeights))
            .expect("the cube animation drives morph weights");
        let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));

        let times: Vec<f32> = reader.read_inputs().expect("weights have keyframe times").collect();
        let flat: Vec<f32> = match reader.read_outputs().expect("weights have outputs") {
            ReadOutputs::MorphTargetWeights(weights) => weights.into_f32().collect(),
            _ => panic!("the channel should carry morph target weights"),
        };

        let keyframes = flat
            .chunks_exact(TARGET_COUNT)
            .map(|frame| [frame[0], frame[1]])
            .collect();
        (times, keyframes, duration)
    }

    /// Linearly sample the weights channel at `time`.
    fn cpu_sample(times: &[f32], keyframes: &[[f32; 2]], time: f32) -> [f32; 2] {
        let last = times.len() - 1;
        if time <= times[0] {
            return keyframes[0];
        }
        if time >= times[last] {
            return keyframes[last];
        }

        let second = times.partition_point(|&t| t <= time);
        let first = second - 1;
        let span = times[second] - times[first];
        let amount = if span > f32::EPSILON { (time - times[first]) / span } else { 0.0 };

        [
            keyframes[first][0] + (keyframes[second][0] - keyframes[first][0]) * amount,
            keyframes[first][1] + (keyframes[second][1] - keyframes[first][1]) * amount,
        ]
    }

    #[test]
    fn bakes_morph_weight_texture() {
        let dir = temp_workspace(SOURCE_GLTF, "texture");
        let (manager, glb) = import(&dir, "AnimatedMorphCube.glb");
        let (times, keyframes, duration) = cpu_weights(&glb);

        let name = GltfImporter::morph_animation_subasset_name(&glb, 0, 0, 0, 0)
            .expect("morph animation name should resolve");
        assert!(
            name.ends_with(".morphanim_0_0_0_0"),
            "morph animations must follow `.morphanim_<scene>_<anim>_<mesh>_<prim>`, got '{name}'"
        );

        let asset = manager.get_asset(&name).expect("morph animation should be imported");
        let asset = asset.read().unwrap();
        let AssetData::VertexMorphAnimationAssetType(morph) = asset.get_data() else {
            panic!("'{name}' is not a morph animation asset");
        };

        assert_eq!(
            morph.mesh,
            GltfImporter::skinned_primitive_subasset_name(&glb, 0, 0).unwrap(),
            "the track must reference its deformable mesh"
        );
        assert_eq!(
            morph.target,
            GltfImporter::morph_target_subasset_name(&glb, 0, 0).unwrap(),
            "the track must reference its morph target asset"
        );
        assert_eq!(morph.source, "scene0/anim0", "the clip source identifies the glTF animation");
        assert_eq!(morph.fps, 30.0);

        // The frame count comes from the animation duration, which the skeletal
        // track uses too, so both tracks stay in sync (D12 / I4).
        let expected_frames = ((duration * morph.fps).ceil() as u32) + 1;
        assert_eq!(morph.frames, expected_frames, "frames = ceil(duration * fps) + 1");
        assert!(morph.frames > 2, "the cube animation should span several frames");

        assert_eq!(morph.texture.format, TextureFormat::Rgba16Float);
        assert_eq!(
            morph.texture.width.get(),
            1,
            "two targets must share a single texel per frame"
        );
        assert_eq!(
            morph.texture.height.unwrap().get(),
            morph.frames + 1,
            "the texture needs one wrapping row on top of the baked frames"
        );

        let texels = match &morph.texture.texels {
            TexelData::Plain(texels) => texels,
            _ => panic!("baked weights must be a plain texture"),
        };
        let width = morph.texture.width.get() as usize;
        let frame_bytes = width * 8;
        assert_eq!(texels.len(), frame_bytes * (morph.frames as usize + 1));

        for frame in [0, morph.frames / 2, morph.frames - 1] {
            let time = frame as f32 / morph.fps;
            let expected = cpu_sample(&times, &keyframes, time);
            let texel = texel_at(texels, width, 0, frame as usize);

            assert!(
                (texel[0] - expected[0]).abs() < 1e-3,
                "frame {frame}: r = {} but CPU sampling gives {}",
                texel[0],
                expected[0]
            );
            assert!(
                (texel[1] - expected[1]).abs() < 1e-3,
                "frame {frame}: g = {} but CPU sampling gives {}",
                texel[1],
                expected[1]
            );
            assert_eq!(texel[2], 0.0, "frame {frame}: unused texel channels must be zero");
            assert_eq!(texel[3], 0.0, "frame {frame}: unused texel channels must be zero");
        }

        // The animation must actually animate.
        let first = texel_at(texels, width, 0, 0);
        let middle = texel_at(texels, width, 0, morph.frames as usize / 2);
        assert_ne!(first, middle, "the baked weights look constant");

        // The last row repeats frame 0, so the shader can wrap without a modulo.
        assert_eq!(
            &texels[..frame_bytes],
            &texels[morph.frames as usize * frame_bytes..],
            "the wrap row must repeat frame 0"
        );
    }

    #[test]
    fn weights_only_animation_has_no_skeletal_track() {
        let dir = temp_workspace(SOURCE_GLTF, "no_skeletal");
        let (manager, glb) = import(&dir, "AnimatedMorphCube.glb");

        // `classify_animation` reports no skin for a weights-only animation, so
        // the importer must not bake a joint-matrix track.
        let animation_name = GltfImporter::animation_subasset_name(&glb, 0, 0)
            .expect("animation name should resolve");
        assert!(
            manager.get_asset(&animation_name).is_none(),
            "'{animation_name}' must not exist: the cube animation drives no skin"
        );

        // `mesh.weights` becomes the static base shape of the target asset.
        let target_name = GltfImporter::morph_target_subasset_name(&glb, 0, 0)
            .expect("morph target name should resolve");
        let asset = manager.get_asset(&target_name).expect("morph target should be imported");
        let asset = asset.read().unwrap();
        let AssetData::VertexMorphTargetAssetType(target) = asset.get_data() else {
            panic!("'{target_name}' is not a morph target asset");
        };
        assert_eq!(
            target.default_weights.as_deref(),
            Some([0.0f32, 0.0].as_slice()),
            "the cube's `mesh.weights` must be used as the base shape"
        );
    }

    #[test]
    fn skeletal_tracks_carry_the_clip_source() {
        let dir = temp_workspace(FOX_GLTF, "fox");
        let (manager, glb) = import(&dir, "Fox.glb");

        for animation_index in 0..3 {
            let name = GltfImporter::animation_subasset_name(&glb, 0, animation_index)
                .expect("animation name should resolve");
            let asset = manager.get_asset(&name)
                .unwrap_or_else(|| panic!("animation '{name}' should be imported"));
            let asset = asset.read().unwrap();
            let AssetData::SkinnedAnimationAssetType(animation) = asset.get_data() else {
                panic!("'{name}' is not a skeletal animation asset");
            };
            assert_eq!(
                animation.source,
                format!("scene0/anim{animation_index}"),
                "the skeletal track must carry the clip source"
            );
        }

        // The fox has no morph targets, so no morph weight track may be baked.
        let database = manager.get_database();
        assert!(
            database.iter().all(|(name, _)| !name.contains(".morphanim_")),
            "a mesh without morph targets must not produce a morph animation asset"
        );
    }
}
