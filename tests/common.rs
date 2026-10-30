//! Helpers shared by the skeletal animation integration tests.

#![allow(dead_code)]

use std::num::NonZero;

use rust_renderer::asset::asset_types::texture_asset::{
    Sampler, TexelData, TextureAsset, TextureDimension, TextureFormat,
};

/// Encode a value as IEEE-754 binary16.
///
/// Only the exact values the tests need are supported; that keeps the helper
/// free of rounding logic while still producing bit-exact input.
pub fn f16_bits(value: f32) -> u16 {
    match value {
        0.0 => 0x0000,
        1.0 => 0x3C00,
        -1.0 => 0xBC00,
        other => panic!("this helper only encodes 0/±1, got {other}"),
    }
}

/// Build a one-frame animation texture from column-major joint matrices.
///
/// `columns` holds four columns per joint, laid out exactly like the baked
/// animation asset: `width = joint_count * 4`, `height = 1`.
pub fn joint_matrix_texture(columns: &[[f32; 4]]) -> TextureAsset {
    assert_eq!(columns.len() % 4, 0, "every joint needs four columns");
    let joint_count = columns.len() / 4;

    let mut texels = Vec::with_capacity(columns.len() * 4 * 2);
    for column in columns {
        for channel in *column {
            texels.extend_from_slice(&f16_bits(channel).to_le_bytes());
        }
    }

    TextureAsset {
        format: TextureFormat::Rgba16Float,
        dimension: TextureDimension::D2,
        width: NonZero::new(joint_count as u32 * 4).unwrap(),
        height: Some(NonZero::new(1).unwrap()),
        depth_or_array_slice: None,
        sampler: Sampler::default(),
        texels: TexelData::Plain(texels.into_boxed_slice()),
    }
}

/// Column-major identity matrix.
pub const IDENTITY_COLUMNS: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// One frame of `joint_count` identity joint matrices, i.e. the bind pose.
pub fn rest_pose_texture(joint_count: usize) -> TextureAsset {
    let columns: Vec<[f32; 4]> = (0..joint_count)
        .flat_map(|_| IDENTITY_COLUMNS)
        .collect();
    joint_matrix_texture(&columns)
}
