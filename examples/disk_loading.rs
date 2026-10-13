//! Headless demo of the disk-driven asset pipeline.
//!
//! Run with `cargo run --example disk_loading`.
//!
//! The first run imports the configured jobs and persists them to disk; later
//! runs load the serialized containers instead of re-importing, and only
//! re-import when a source file or the importer version changes.

use std::path::Path;

use rust_renderer::asset::{
    asset_manager::AssetManager,
    asset_store::{ImportJob, ImportManifest, ManifestEntry},
};

const GLTF_PATH: &str = "resource/test_two_cubes.glb";
const MANIFEST_PATH: &str = "resource/import_manifest.ff";

fn main() {
    env_logger::init();

    // Seed the manifest on first run; afterwards it just records outputs.
    let manifest_path = Path::new(MANIFEST_PATH);
    let mut manifest = ImportManifest::load(manifest_path);
    if manifest.entries.is_empty() {
        manifest.entries = vec![
            ManifestEntry { job: ImportJob::gltf(GLTF_PATH), outputs: Vec::new() },
        ];
        manifest.save(manifest_path);
    }

    // `import_manifest` loads fresh assets and re-imports stale/missing ones.
    let manager = AssetManager::new();
    let report = manager.import_manifest(manifest_path);
    println!("manifest: loaded {} assets, re-imported {}", report.loaded, report.reimported);

    // Independently of any manifest, every serialized asset can be scanned and
    // loaded straight from a directory.
    let loader = AssetManager::new();
    let scan = loader.load_directory(Path::new("resource"));
    println!("directory scan: loaded {} assets (skipped {})", scan.loaded, scan.failed);
}
