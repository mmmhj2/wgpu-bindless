use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::asset::{
    error::AssetIoError,
    importer::{
        gltf_importer::GltfImporter,
        image_file_importer::{ColorSpace, ImageArrayImporter, ImageFileImporter},
        AssetImporter,
    },
};

/// A serializable import recipe: how to (re)produce one batch of assets.
///
/// Import recipes are what make re-import possible when an asset is missing or
/// stale. The objects here record only the configuration needed to rebuild a
/// concrete [`AssetImporter`], not the runtime importer state.
/// 
/// *TODO:* Is it really necessary to create separate structs?
/// Maybe we can just store the importer object.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ImportJob {
    Gltf { path: PathBuf },
    Image { source: PathBuf, color_space: ColorSpace },
    ImageArray { name: Option<String>, sources: Vec<PathBuf>, color_space: ColorSpace },
}

impl ImportJob {
    pub fn gltf(path: impl Into<PathBuf>) -> Self {
        ImportJob::Gltf { path: path.into() }
    }

    pub fn image(source: impl Into<PathBuf>, color_space: ColorSpace) -> Self {
        ImportJob::Image { source: source.into(), color_space }
    }

    pub fn image_array<P: Into<PathBuf>>(
        name: Option<String>,
        sources: impl IntoIterator<Item = P>,
        color_space: ColorSpace,
    ) -> Self {
        ImportJob::ImageArray { name, sources: sources.into_iter().map(Into::into).collect(), color_space }
    }

    /// Stable identifier of the importer kind this job dispatches to.
    pub fn importer_id(&self) -> &'static str {
        match self {
            ImportJob::Gltf { .. } => <GltfImporter as AssetImporter>::IMPORTER_ID,
            ImportJob::Image { .. } => <ImageFileImporter as AssetImporter>::IMPORTER_ID,
            ImportJob::ImageArray { .. } => <ImageArrayImporter as AssetImporter>::IMPORTER_ID,
        }
    }

    /// Version of the importer logic; must match the produced assets' provenance.
    pub fn importer_version(&self) -> u32 {
        match self {
            ImportJob::Gltf { .. } => <GltfImporter as AssetImporter>::IMPORTER_VERSION,
            ImportJob::Image { .. } => <ImageFileImporter as AssetImporter>::IMPORTER_VERSION,
            ImportJob::ImageArray { .. } => <ImageArrayImporter as AssetImporter>::IMPORTER_VERSION,
        }
    }

    /// The main source file of this job, if it has one.
    pub fn primary_source(&self) -> Option<&Path> {
        match self {
            ImportJob::Gltf { path } => Some(path),
            ImportJob::Image { source, .. } => Some(source),
            ImportJob::ImageArray { sources, .. } => sources.first().map(PathBuf::as_path),
        }
    }

    /// Directory that this job's sources live in.
    pub fn resource_root(&self) -> PathBuf {
        self.primary_source()
            .and_then(Path::parent)
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    }

    /// Stable identity string used to key a job (e.g. for deduplication).
    pub fn job_id(&self) -> String {
        let source = self.primary_source().map(Path::display).map(|d| d.to_string()).unwrap_or_default();
        format!("{}:{}", self.importer_id(), source)
    }
}

/// A single entry of an [`ImportManifest`].
///
/// The `job` is the declarative part; `outputs` is the generated state: the
/// list of `.asset` files this job produced last time.
#[derive(Serialize, Deserialize, Debug)]
pub struct ManifestEntry {
    pub job: ImportJob,
    pub outputs: Vec<PathBuf>,
}

/// The merged manifest/catalog: the single file describing which import jobs
/// exist and what each one last produced.
///
/// Provenance is *not* duplicated here; it lives inside each asset container and
/// is read via `AssetMetadata::peek`.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct ImportManifest {
    pub entries: Vec<ManifestEntry>,
}

impl ImportManifest {
    /// Load a manifest, returning an empty one if the file does not exist.
    pub fn load(path: &Path) -> Result<Self, AssetIoError> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let bytes = std::fs::read(path)
            .map_err(|source| AssetIoError::Read { path: path.to_path_buf(), source })?;
        flexbuffers::from_slice(&bytes)
            .map_err(|source| AssetIoError::Deserialize { what: "import manifest", path: path.to_path_buf(), source })
    }

    /// Persist the manifest (including generated `outputs`).
    pub fn save(&self, path: &Path) -> Result<(), AssetIoError> {
        let bytes = flexbuffers::to_vec(self)
            .map_err(|source| AssetIoError::Serialize { what: "import manifest", source })?;
        std::fs::write(path, bytes)
            .map_err(|source| AssetIoError::Write { path: path.to_path_buf(), source })
    }
}

/// Outcome of ensuring a single manifest entry.
#[derive(Debug, PartialEq, Eq)]
pub enum EnsureOutcome {
    /// The existing assets were fresh and only loaded.
    Fresh,
    /// The assets were missing or stale and got re-imported.
    Reimported,
}

/// Summary of a directory load / manifest refresh.
#[derive(Debug, Default)]
pub struct LoadReport {
    pub loaded: usize,
    pub reimported: usize,
    /// Files that could not be read during a directory scan.
    pub failed: usize,
}

/// Recursively collect every `*.asset` file under `root` (sorted for determinism).
pub fn walk_asset_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    walk_asset_files_into(root, &mut files);
    files.sort();
    files
}

fn walk_asset_files_into(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_asset_files_into(&path, files);
        } else if path.extension().and_then(|e| e.to_str()) == Some("asset") {
            files.push(path);
        }
    }
}
