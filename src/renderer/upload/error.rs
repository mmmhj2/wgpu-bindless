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

    #[error("buffer '{0}' has no data")]
    EmptyBuffer(String),

    #[error(transparent)]
    Bindless(#[from] BindlessError),
}
