#[cfg(test)]
mod test {
    use std::path::{Path, PathBuf};

    use rust_renderer::asset::{
        asset_manager::AssetManager,
        asset_store::{EnsureOutcome, ImportJob, ImportManifest, ManifestEntry},
        importer::gltf_importer::GltfImporter,
    };

    const SOURCE_GLTF: &str = "resource/test_two_cubes.glb";

    /// Create an isolated temp workspace containing a copy of the test glTF.
    ///
    /// Importers write assets next to their source, so copying the source keeps
    /// generated `*.asset` files out of `resource/` (and avoids loading the
    /// large skybox assets).
    fn temp_workspace(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rust_renderer_manifest_{name}"));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).ok();
        }
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(SOURCE_GLTF, dir.join("test_two_cubes.glb")).unwrap();
        dir
    }

    #[test]
    fn load_directory_loads_persisted_assets() {
        let dir = temp_workspace("load_directory");
        let glb = dir.join("test_two_cubes.glb");

        // First manager imports, which persists the assets to the workspace.
        {
            let manager = AssetManager::new();
            manager.import(GltfImporter::new(&glb));
        }

        // A fresh manager loads everything from disk without importing.
        let loader = AssetManager::new();
        let report = loader.load_directory(&dir);
        assert!(report.loaded > 0, "load_directory should find persisted assets");
        assert_eq!(report.reimported, 0, "load_directory never imports");

        let scene_name = GltfImporter::scene_subasset_name(&glb, 0);
        assert!(loader.get_asset(&scene_name).is_some(), "scene asset should be loaded from disk");
    }

    #[test]
    fn ensure_entry_reimports_then_is_fresh() {
        let dir = temp_workspace("ensure_fresh");
        let glb = dir.join("test_two_cubes.glb");
        let manager = AssetManager::new();

        let mut entry = ManifestEntry { job: ImportJob::gltf(&glb), outputs: Vec::new() };

        assert_eq!(manager.ensure_entry(&mut entry), EnsureOutcome::Reimported);
        assert!(!entry.outputs.is_empty(), "re-import should record outputs");
        assert!(entry.outputs.iter().all(|p| p.exists()));

        let outputs = entry.outputs.clone();
        assert_eq!(manager.ensure_entry(&mut entry), EnsureOutcome::Fresh);
        assert_eq!(entry.outputs, outputs, "a fresh entry keeps the same outputs");
    }

    #[test]
    fn ensure_entry_reimports_when_output_missing() {
        let dir = temp_workspace("ensure_missing");
        let glb = dir.join("test_two_cubes.glb");
        let manager = AssetManager::new();

        let mut entry = ManifestEntry { job: ImportJob::gltf(&glb), outputs: Vec::new() };
        assert_eq!(manager.ensure_entry(&mut entry), EnsureOutcome::Reimported);

        std::fs::remove_file(&entry.outputs[0]).unwrap();
        assert_eq!(manager.ensure_entry(&mut entry), EnsureOutcome::Reimported);
        assert!(entry.outputs.iter().all(|p| p.exists()), "missing output should be regenerated");
    }

    #[test]
    fn manifest_roundtrip_records_outputs_and_refreshes() {
        let dir = temp_workspace("manifest");
        let glb = dir.join("test_two_cubes.glb");
        let manifest_path = dir.join("import_manifest.ff");

        ImportManifest {
            entries: vec![ManifestEntry { job: ImportJob::gltf(&glb), outputs: Vec::new() }],
        }.save(&manifest_path);

        // First run re-imports and writes the produced outputs back to disk.
        let first = AssetManager::new().import_manifest(&manifest_path);
        assert!(first.reimported > 0);
        assert_eq!(first.loaded, 0);

        let reloaded = ImportManifest::load(&manifest_path);
        assert!(!reloaded.entries[0].outputs.is_empty(), "manifest should persist outputs");

        // Second run finds everything fresh and only loads.
        let second = AssetManager::new().import_manifest(&manifest_path);
        assert_eq!(second.reimported, 0);
        assert!(second.loaded > 0);
    }

    #[test]
    fn load_directory_skips_unreadable_files() {
        let dir = temp_workspace("load_dir_corrupt");
        let glb = dir.join("test_two_cubes.glb");
        {
            let manager = AssetManager::new();
            manager.import(GltfImporter::new(&glb));
        }

        // A leftover pre-container (or otherwise corrupt) asset must not abort the scan.
        std::fs::write(dir.join("legacy.asset"), b"not a zip container").unwrap();

        let loader = AssetManager::new();
        let report = loader.load_directory(&dir);
        assert!(report.loaded > 0, "valid assets should still load");
        assert_eq!(report.failed, 1, "the corrupt file should be reported, not fatal");
    }

    #[test]
    fn manifest_load_of_missing_file_is_empty() {
        let path = std::env::temp_dir().join("rust_renderer_manifest_does_not_exist.ff");
        std::fs::remove_file(&path).ok();
        let manifest = ImportManifest::load(Path::new(&path));
        assert!(manifest.entries.is_empty());
    }
}
