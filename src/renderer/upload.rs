//! Single entry point for turning CPU-side asset data into GPU resources.

pub mod buffer;
pub mod context;
pub mod error;
pub mod gpu_mesh;
pub mod gpu_type;
pub mod texture;

/// Maps the serialized texture enums onto `wgpu` enums.
/// Privated as only the impls are of interest, and they are visible crate-wide.
mod format;
mod impl_animation;
mod impl_mesh;
mod impl_texture;

pub use buffer::upload_buffer;
pub use context::UploadContext;
pub use crate::asset::asset_types::animation_asset::MAX_MORPH_TARGETS;
pub use error::UploadError;
pub use gpu_mesh::{GpuMorphTargets, GpuSkinnedMesh, GpuStaticMesh, MAX_MORPH_CHANNELS, MORPH_LAYOUT_WORDS};
pub use gpu_type::GpuAssetType;
pub use texture::{TextureBinding, create_texture};
