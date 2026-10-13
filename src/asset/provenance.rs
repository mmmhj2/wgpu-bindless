use std::{fs::File, io::Read, path::Path, time::{SystemTime, UNIX_EPOCH}};

use serde::{Deserialize, Serialize};

use crate::asset::error::AssetIoError;

/// A content fingerprint of a single source file used to produce an asset.
///
/// The file size is stored as a cheap fast-path check and the BLAKE3 hash is
/// authoritative.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct SourceFingerprint {
    /// Path relative to the resource root (absolute if outside the root).
    pub path: std::path::PathBuf,
    pub size: u64,
    pub hash: [u8; 32],
}

impl SourceFingerprint {
    /// Compute the fingerprint of a file on disk.
    pub fn of(path: impl AsRef<Path>) -> Result<Self, AssetIoError> {
        let path = path.as_ref();
        let metadata = std::fs::metadata(path)
            .map_err(|source| AssetIoError::Source { path: path.to_path_buf(), source })?;

        let mut hasher = blake3::Hasher::new();
        let mut file = File::open(path)
            .map_err(|source| AssetIoError::Source { path: path.to_path_buf(), source })?;

        let mut buffer = [0u8; 64 * 1024];
        loop {
            let read = file.read(&mut buffer)
                .map_err(|source| AssetIoError::Source { path: path.to_path_buf(), source })?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }

        Ok(Self {
            path: path.to_path_buf(),
            size: metadata.len(),
            hash: *hasher.finalize().as_bytes(),
        })
    }

    pub fn size_matches(&self) -> bool {
        std::fs::metadata(&self.path).map(|m| m.len() == self.size).unwrap_or(false)
    }

    pub fn matches(&self) -> bool {
        if !self.size_matches() {
            return false;
        }
        Self::of(&self.path).map(|fingerprint| fingerprint.hash == self.hash).unwrap_or(false)
    }
}

/// How an asset was produced, recorded inside the asset file itself.
///
/// This lets the loader decide whether a serialized asset can be reused or must
/// be re-imported because its sources or the importer changed.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ImportProvenance {
    /// Stable identifier of the importer kind, e.g. `"gltf"`.
    pub importer_id: String,
    /// Version of the importer logic that produced this asset.
    pub importer_version: u32,
    /// Wall-clock time of the import, for diagnostics only.
    pub imported_unix_ms: i64,
    /// The source files this asset was derived from.
    pub sources: Vec<SourceFingerprint>,
}

impl ImportProvenance {
    /// Whether this provenance was produced by the given importer kind/version.
    pub fn matches_importer(&self, importer_id: &str, importer_version: u32) -> bool {
        self.importer_id == importer_id && self.importer_version == importer_version
    }

    /// Freshness check: every recorded source is unchanged.
    pub fn is_fresh(&self) -> bool {
        !self.sources.is_empty() && self.sources.iter().all(SourceFingerprint::matches)
    }
}

/// Fingerprint a batch of source paths.
pub fn fingerprint_all(
    paths: impl IntoIterator<Item = std::path::PathBuf>,
) -> Result<Vec<SourceFingerprint>, AssetIoError> {
    paths.into_iter().map(SourceFingerprint::of).collect()
}

/// Current wall-clock time in milliseconds since the Unix epoch.
pub fn now_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
