#[cfg(test)]
mod test {
    use std::path::{Path, PathBuf};

    use rust_renderer::asset::{
        asset_manager::AssetManager,
        asset_types::{Asset, AssetMetadata},
        importer::{
            gltf_importer::GltfImporter,
            image_file_importer::{ColorSpace, ImageFileImporter},
        },
        provenance::SourceFingerprint,
    };

    const GLTF_PATH: &str = "resource/test_two_cubes.glb";
    const IMAGE_PATH: &str = "resource/skybox/back.jpg";

    /// Reads any asset's name, on-disk path and provenance out of the manager,
    /// then verifies the container it wrote can be peeked and fully re-loaded.
    #[test]
    fn imported_assets_are_persisted_as_zip_containers_with_provenance() {
        let manager = AssetManager::new();
        manager.import(GltfImporter::new(GLTF_PATH)).expect("import should succeed");

        let (name, path, provenance) = {
            let database = manager.get_database();
            let (name, arc) = database.iter().next().expect("at least one asset should be imported");
            let asset = arc.read().expect("failed to read imported asset");
            let path = asset.get_asset_file_path().expect("imported assets have a file path").to_path_buf();
            let provenance = asset.metadata().provenance.clone().expect("import should stamp provenance");
            (name.clone(), path, provenance)
        };

        assert!(path.exists(), "asset container should exist on disk: {}", path.display());

        // The importer identity is recorded.
        assert_eq!(provenance.importer_id, "gltf");
        assert_eq!(provenance.importer_version, 1);

        // The glTF file is recorded as a source and is unchanged, so it is fresh.
        assert!(!provenance.sources.is_empty(), "provenance should record source files");
        assert!(
            provenance.sources.iter().any(|s| s.path.as_path() == Path::new(GLTF_PATH)),
            "provenance should reference the glTF source"
        );
        assert!(provenance.is_fresh(), "an unchanged source should be fresh");

        // `peek` reads only metadata; the payload is never opened.
        let peeked = AssetMetadata::peek(&path).expect("metadata should peek");
        assert_eq!(peeked.asset_name, name);
        assert!(peeked.provenance.is_some(), "peeked metadata should carry provenance");

        // The full container round-trips.
        let loaded = Asset::load_from_disk(&path).expect("asset should load from its container");
        assert_eq!(loaded.metadata().asset_name, name);
    }

    #[test]
    fn image_importer_records_its_own_identity() {
        let manager = AssetManager::new();
        manager.import(ImageFileImporter::new(IMAGE_PATH, ColorSpace::Srgb)).expect("import should succeed");

        let database = manager.get_database();
        let asset = database.values().next().expect("one texture asset should be imported").read().unwrap();
        let provenance = asset.metadata().provenance.clone().expect("import should stamp provenance");

        assert_eq!(provenance.importer_id, "image_file");
        assert_eq!(provenance.importer_version, 1);
        assert!(provenance.sources.iter().any(|s| s.path.as_path() == Path::new(IMAGE_PATH)));
    }

    /// Fast path is size-only; the authoritative check is the BLAKE3 hash.
    #[test]
    fn fingerprint_detects_size_and_same_size_content_changes() {
        let dir = std::env::temp_dir().join("rust_renderer_phase1_fingerprint");
        std::fs::create_dir_all(&dir).unwrap();
        let path: PathBuf = dir.join("source.bin");

        std::fs::write(&path, b"hello world").unwrap();
        let fingerprint = SourceFingerprint::of(&path).expect("fingerprint should succeed");
        assert_eq!(fingerprint.size, 11);
        assert!(fingerprint.size_matches());
        assert!(fingerprint.matches());

        // Same size, different content: size fast path is fooled, hash catches it.
        std::fs::write(&path, b"HELLO WORLD").unwrap();
        assert!(fingerprint.size_matches());
        assert!(!fingerprint.matches());

        // Different size: both checks fail.
        std::fs::write(&path, b"hello world!").unwrap();
        assert!(!fingerprint.size_matches());
        assert!(!fingerprint.matches());

        // Missing file: both checks fail.
        std::fs::remove_file(&path).unwrap();
        assert!(!fingerprint.size_matches());
        assert!(!fingerprint.matches());

        std::fs::remove_dir_all(&dir).ok();
    }
}
