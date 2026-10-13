use std::{collections::HashMap, path::{Path, PathBuf}, sync::{Arc, RwLock, RwLockReadGuard}};

use crate::asset::{
    asset_io_error::AssetIoError,
    asset_store::{walk_asset_files, EnsureOutcome, ImportJob, ImportManifest, LoadReport, ManifestEntry},
    asset_types::{self, Asset, AssetMetadata},
    importer::{gltf_importer::GltfImporter, image_file_importer::{ImageArrayImporter, ImageFileImporter}, AssetImporter, ImporterContext},
    provenance::{fingerprint_all, now_unix_ms, ImportProvenance},
};

pub struct AssetManager {
    asset_database : RwLock<HashMap<String, Arc<RwLock<Asset>>>>
}

impl AssetManager {
    pub fn new() -> Self {
        Self {
            asset_database: HashMap::new().into()
        }
    }

    /// Import through a concrete importer, stamp provenance, persist, register.
    pub fn import<T: AssetImporter>(&self, importer: T) {
        self.run_importer(importer);
    }

    /// Shared import path: run the importer, stamp provenance, save every
    /// produced asset to disk, and return the output paths.
    fn run_importer<T: AssetImporter>(&self, importer: T) -> Vec<PathBuf> {
        let mut context = ImporterContext::new();
        importer.import(&context);

        // Stamp every produced asset with provenance built from the sources the
        // importer recorded, then persist the container so it can be reused.
        let provenance = ImportProvenance {
            importer_id: T::IMPORTER_ID.to_string(),
            importer_version: T::IMPORTER_VERSION,
            imported_unix_ms: now_unix_ms(),
            sources: fingerprint_all(context.take_sources()),
        };

        let pending_database = context.asset_batch.into_inner().expect("Failed to consume pending asset database.");
        let mut outputs = Vec::with_capacity(pending_database.len());
        for asset in pending_database.values() {
            let mut asset = asset.write().expect("Failed to acquire write lock for asset.");
            asset.metadata_mut().provenance = Some(provenance.clone());
            asset.save_to_disk().expect("Failed to save imported asset to disk.");
            outputs.push(asset.get_asset_file_path().expect("imported asset should have a file path").to_path_buf());
        }

        let mut database = self.asset_database.write().expect("Failed to acquire write lock for asset database");
        database.extend(pending_database);
        outputs
    }

    /// Execute an import recipe, registering and persisting all produced assets.
    pub fn reimport_job(&self, job: ImportJob) -> Vec<PathBuf> {
        match job {
            ImportJob::Gltf { path } => self.run_importer(GltfImporter::new(path)),
            ImportJob::Image { source, color_space } => self.run_importer(ImageFileImporter::new(source, color_space)),
            ImportJob::ImageArray { name, sources, color_space } => self.run_importer(ImageArrayImporter::new(name, sources, color_space)),
        }
    }

    /// Load one serialized asset container from disk and register it.
    pub fn load_asset_file(&self, path: &Path) -> Arc<RwLock<Asset>> {
        self.try_load_asset_file(path).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Fallible variant of [`load_asset_file`] for tolerant bulk loading.
    ///
    /// [`load_asset_file`]: AssetManager::load_asset_file
    pub fn try_load_asset_file(&self, path: &Path) -> Result<Arc<RwLock<Asset>>, AssetIoError> {
        let asset = Asset::try_load_from_disk(path)?;
        let name = asset.metadata().asset_name.clone();
        let asset = Arc::new(RwLock::new(asset));
        self.asset_database.write().expect("Failed to acquire write lock for asset database").insert(name, asset.clone());
        Ok(asset)
    }

    /// Recursively load every `*.asset` under `root` without importing.
    ///
    /// Files that cannot be read (corrupt or pre-container legacy assets) are
    /// counted in [`LoadReport::failed`] rather than aborting the scan.
    pub fn load_directory(&self, root: &Path) -> LoadReport {
        let mut report = LoadReport::default();
        for path in walk_asset_files(root) {
            match self.try_load_asset_file(&path) {
                Ok(_) => report.loaded += 1,
                Err(e) => {
                    log::warn!("skipping unreadable asset '{}': {e}", path.display());
                    report.failed += 1;
                }
            }
        }
        report
    }

    /// Ensure a single manifest entry is present and fresh.
    ///
    /// Loads the recorded outputs when their provenance matches the recipe and
    /// every source is unchanged; otherwise re-imports the whole job.
    pub fn ensure_entry(&self, entry: &mut ManifestEntry) -> EnsureOutcome {
        if !entry.outputs.is_empty() && entry.outputs.iter().all(|p| p.exists()) {
            if let Ok(metadata) = AssetMetadata::try_peek(&entry.outputs[0]) {
                if let Some(provenance) = metadata.provenance.as_ref() {
                    if provenance.matches_importer(entry.job.importer_id(), entry.job.importer_version())
                        && provenance.is_fresh()
                    {
                        for path in &entry.outputs {
                            self.load_asset_file(path);
                        }
                        return EnsureOutcome::Fresh;
                    }
                }
            }
        }

        entry.outputs = self.reimport_job(entry.job.clone());
        EnsureOutcome::Reimported
    }

    /// Load a manifest and refresh every entry, re-importing missing/stale ones.
    ///
    /// The manifest is written back only when something was re-imported.
    pub fn import_manifest(&self, manifest_path: &Path) -> LoadReport {
        let mut manifest = ImportManifest::load(manifest_path);
        let mut report = LoadReport::default();
        let mut changed = false;

        for entry in manifest.entries.iter_mut() {
            match self.ensure_entry(entry) {
                EnsureOutcome::Fresh => report.loaded += entry.outputs.len(),
                EnsureOutcome::Reimported => {
                    report.reimported += entry.outputs.len();
                    changed = true;
                }
            }
        }

        if changed {
            manifest.save(manifest_path);
        }
        report
    }

    pub fn save_to_disk(&self) -> std::result::Result<(), ()> {
        let database = self.asset_database.read().expect("Failed to acquire read lock for asset database");
        for (_k, v) in database.iter() {
            let r = v.read().expect("Failed to acquire read lock for asset.");
            r.save_to_disk().expect("Failed to save asset to disk.");
        }
        Ok(())
    }

    pub fn get_database(&self) -> std::sync::RwLockReadGuard<'_, HashMap<std::string::String, Arc<std::sync::RwLock<asset_types::Asset>>>> {
        self.asset_database.read().expect("failed to acquire read lock for asset database")
    }

    /// Look up a single asset by its name.
    pub fn get_asset(&self, name: &str) -> Option<Arc<RwLock<Asset>>> {
        self.asset_database.read().expect("failed to acquire read lock for asset database").get(name).cloned()
    }
}
