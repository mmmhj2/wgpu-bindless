use std::num::{NonZero, NonZeroU32};

use bytemuck::Zeroable;

use crate::{
    asset::asset_types::{
        animation_asset::{MorphTarget, VertexMorphTargetAsset, MAX_MORPH_TARGETS},
        texture_asset::{Sampler, TexelData, TextureAsset, TextureDimension, TextureFormat},
    },
    renderer::upload::{
        context::UploadContext,
        error::UploadError,
        gpu_mesh::GpuMorphTargets,
        gpu_type::GpuAssetType,
    },
    util::f16::f32_to_f16_bits,
};

/// Append one attribute block of a morph target to `deltas`.
///
/// An absent attribute sets no mask bit and leaves its offset at zero; a present
/// one records where its block starts and then appends it.
fn push_attribute<const N: usize>(
    deltas: &mut Vec<f32>,
    mask: &mut u32,
    offset: &mut u32,
    bit: u32,
    values: Option<&[[f32; N]]>,
) {
    let Some(values) = values else { return };
    *mask |= 1 << bit;
    *offset = deltas.len() as u32;
    for value in values {
        deltas.extend_from_slice(value);
    }
}

#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct AttributeOffset {
    pub offsets: [u32; 5]
}

#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct MorphLayout {
    pub channel_count: u32,
    pub attribute_masks: [u32; 8],
    pub attribute_offsets: [AttributeOffset; 8]
}

impl GpuAssetType for VertexMorphTargetAsset {
    type GpuObjectType = GpuMorphTargets;

    /// Pack the morph targets into a layout table, one delta buffer and an
    /// optional base-weight texture.
    ///
    /// Offsets are `f32` indices into the delta buffer, so the whole buffer is
    /// bound at offset 0 and no 256-byte binding alignment applies. Missing
    /// attributes are not stored at all.
    fn upload(
        &self,
        name: &str,
        ctx: &mut UploadContext<'_>,
    ) -> Result<Self::GpuObjectType, UploadError> {
        let channel_count = self.targets.len();
        if !(1..=MAX_MORPH_TARGETS).contains(&channel_count) {
            return Err(UploadError::MorphTargetCount {
                name: name.to_string(),
                count: channel_count,
                max: MAX_MORPH_TARGETS,
            });
        }
        let channel_count = channel_count as u32;

        // `[0]` channel count, `[1..9]` masks, `[9..49]` offsets.
        let mut layout = MorphLayout::zeroed();
        layout.channel_count = channel_count;

        let mut deltas: Vec<f32> = Vec::new();
        for (channel, target) in self.targets.iter().enumerate() {
            let mut mask = 0u32;
            let mut offsets = [0u32; MorphTarget::CHANNEL_COUNT];

            push_attribute(&mut deltas, &mut mask, &mut offsets[0], 0, target.position.as_deref());
            push_attribute(&mut deltas, &mut mask, &mut offsets[1], 1, target.normal.as_deref());
            push_attribute(&mut deltas, &mut mask, &mut offsets[2], 2, target.tangent.as_deref());
            push_attribute(&mut deltas, &mut mask, &mut offsets[3], 3, target.texcoord0.as_deref());
            push_attribute(&mut deltas, &mut mask, &mut offsets[4], 4, target.color0.as_deref());

            layout.attribute_masks[channel] = mask;
            layout.attribute_offsets[channel].offsets
                .copy_from_slice(&offsets);
        }

        if deltas.is_empty() {
            log::warn!("zero sized delta buffer encountered for morph target {name}");
            deltas.push(0.0);
        }

        let layout_buffer = ctx.upload_buffer(
            &format!("{name}.morph.layout"),
            wgpu::BufferUsages::STORAGE,
            bytemuck::bytes_of(&layout),
        )?;
        let delta_buffer = ctx.upload_buffer(
            &format!("{name}.morph.deltas"),
            wgpu::BufferUsages::STORAGE,
            deltas.as_slice(),
        )?;

        let base_weights = self.default_weights.as_deref()
            .filter(|weights| weights.iter().any(|weight| *weight != 0.0));
        let base_weights = match base_weights {
            Some(weights) => {
                let texture = base_weight_texture(weights, channel_count as usize);
                Some(ctx.upload_texture(&format!("{name}.morph.base_weights"), &texture)?.handle)
            }
            None => None,
        };
        let base_channels = base_weights.and_then(|_| NonZeroU32::new(channel_count));

        Ok(GpuMorphTargets {
            layout: layout_buffer,
            deltas: delta_buffer,
            base_weights,
            channel_count,
            base_channels,
        })
    }
}

/// Build the one-row RGBA16F texture holding the base morph weights.
///
/// Weights of up to four targets share a texel, exactly like the baked weight
/// textures, so the morph shader can sample the base at `frame = 0`.
fn base_weight_texture(weights: &[f32], channel_count: usize) -> TextureAsset {
    let width = channel_count.div_ceil(4).max(1) as u32;
    let mut texels = Vec::with_capacity(width as usize * 8);
    for x in 0..width as usize {
        for channel in 0..4usize {
            let value = weights.get(x * 4 + channel).copied().unwrap_or(0.0);
            texels.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
        }
    }

    TextureAsset {
        format: TextureFormat::Rgba16Float,
        dimension: TextureDimension::D2,
        // `width` is at least one texel, so neither dimension can be zero.
        width: NonZero::new(width).expect("base weight texture is at least one texel wide"),
        height: NonZero::new(1),
        depth_or_array_slice: None,
        sampler: Sampler::default(),
        texels: TexelData::Plain(texels.into_boxed_slice()),
    }
}
