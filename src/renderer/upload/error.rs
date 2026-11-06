use crate::{
    asset::asset_types::AssetKind,
    renderer::resources::bindless::BindlessError,
};

#[derive(Debug, thiserror::Error)]
pub enum UploadError {
    #[error("asset '{0}' is not loaded")]
    MissingAsset(String),

    #[error("asset '{name}' holds {actual:?}, but {expected:?} was expected")]
    WrongKind {
        name: String,
        actual: AssetKind,
        expected: AssetKind,
    },

    #[error("texture '{name}': payload is {actual} bytes, but the layout needs {expected}")]
    TexelSize {
        name: String,
        expected: usize,
        actual: usize,
    },

    #[error("buffer '{label}' needs {needed} bytes, exceeding max_buffer_size ({limit})")]
    BufferTooLarge {
        label: String,
        needed: u64,
        limit: u64,
    },

    #[error("buffer '{label}' needs {needed} bytes, exceeding max_storage_buffer_binding_size ({limit})")]
    StorageBindingTooLarge {
        label: String,
        needed: u64,
        limit: u64,
    },

    #[error("buffer '{0}' has no data")]
    EmptyBuffer(String),

    #[error("mesh batch has {count} instances, which does not fit in the u32 instance range")]
    TooManyInstances { count: usize },

    #[error("asset '{0}' has a poisoned lock")]
    AssetLockPoisoned(String),

    #[error("morph target asset '{name}' has {count} targets, but the supported range is 1..={max}")]
    MorphTargetCount {
        name: String,
        count: usize,
        max: usize,
    },

    #[error("mesh '{name}' has {positions} positions but {attributes} vertex attributes")]
    VertexCountMismatch {
        name: String,
        positions: usize,
        attributes: usize,
    },

    #[error("skinned mesh '{name}' has {vertices} vertices but {skinning} skinning entries")]
    SkinningCountMismatch {
        name: String,
        vertices: usize,
        skinning: usize,
    },

    #[error("texture '{name}' declares more mip levels than its size allows (mip {level})")]
    MipChainTooLong { name: String, level: u32 },

    #[error(transparent)]
    Bindless(#[from] BindlessError),
}
