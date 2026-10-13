use std::path::PathBuf;

use thiserror::Error;

/// Error produced by an [`AssetImporter`](crate::asset::importer::AssetImporter).
pub type ImportError = Box<dyn std::error::Error>;

/// Errors from reading or writing an asset container and its manifest entries.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AssetIoError {
    #[error("cannot open '{path}'")]
    Open {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("cannot create '{path}'")]
    Create {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("cannot read '{path}'")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("cannot write '{path}'")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("'{path}' is not a valid asset container")]
    Container {
        path: PathBuf,
        #[source]
        source: zip::result::ZipError,
    },

    #[error("asset container '{path}' has no usable entry '{entry}'")]
    Entry {
        path: PathBuf,
        entry: String,
        #[source]
        source: zip::result::ZipError,
    },

    #[error("cannot read entry '{entry}' from '{path}'")]
    EntryRead {
        path: PathBuf,
        entry: String,
        #[source]
        source: std::io::Error,
    },

    #[error("cannot write entry '{entry}' to '{path}'")]
    EntryWrite {
        path: PathBuf,
        entry: &'static str,
        #[source]
        source: std::io::Error,
    },

    #[error("cannot finalize asset container '{path}'")]
    Finalize {
        path: PathBuf,
        #[source]
        source: zip::result::ZipError,
    },

    #[error("cannot serialize {what}")]
    Serialize {
        what: &'static str,
        #[source]
        source: flexbuffers::SerializationError,
    },

    #[error("cannot deserialize {what} from '{path}'")]
    Deserialize {
        what: &'static str,
        path: PathBuf,
        #[source]
        source: flexbuffers::DeserializationError,
    },

    #[error("source file '{path}' is unreadable")]
    Source {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("asset '{0}' has no file path and cannot be persisted")]
    MissingFilePath(String),
}

/// Errors from high-level asset operations: importing, loading and persisting.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AssetError {
    #[error("asset file IO error")]
    Io(#[from] AssetIoError),

    #[error("import failed")]
    Import(#[from] ImportError),

    /// Note: This should be extremely rare, and is usually indicative of a bug in the engine or a plugin.
    #[error("a lock was poisoned by a previous panic")]
    LockPoisoned,
}
