use std::{collections::HashMap, path::{Path, PathBuf}, sync::{Arc, RwLock}};

use crate::asset::{asset_types::Asset, error::ImportError};

pub mod gltf_importer;
mod gltf_mrao;
pub mod image_file_importer;
pub mod vertex_reconditioner;

/// Sink that collects assets produced by a single import operation.
///
/// The source(s) to import are owned by the importer itself; this context only
/// gathers the resulting assets so the `AssetManager` can merge them. Importers
/// also register every source file they actually read via [`record_source`], so
/// the manager can fingerprint them into the asset provenance.
///
/// [`record_source`]: ImporterContext::record_source
pub struct ImporterContext {
    pub asset_batch: RwLock<HashMap<String, Arc<RwLock<Asset>>>>,
    recorded_sources: RwLock<Vec<PathBuf>>,
}

impl ImporterContext {
    pub fn new() -> Self {
        Self { asset_batch: HashMap::new().into(), recorded_sources: Vec::new().into() }
    }
    /// Register a source file that this import read from.
    pub fn record_source(&self, path: impl AsRef<Path>) {
        self.recorded_sources.write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(path.as_ref().to_path_buf());
    }
    /// Take the recorded source paths, leaving the context empty.
    pub fn take_sources(&mut self) -> Vec<PathBuf> {
        std::mem::take(
            self.recorded_sources.write()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .as_mut(),
        )
    }
}

impl Default for ImporterContext {
    fn default() -> Self {
        Self::new()
    }
}

pub trait AssetImporter {
    /// Stable identifier of this importer kind, stored in asset provenance.
    const IMPORTER_ID: &'static str;

    /// Version of the import logic. 
    /// 
    /// Bump whenever the produced output changes for unchanged sources, so the
    /// loader re-imports existing assets.
    const IMPORTER_VERSION: u32;

    fn import(self, context: &ImporterContext) -> Result<(), ImportError>;
}
