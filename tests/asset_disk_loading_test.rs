#[cfg(test)]
mod test {
    use std::path::{Path, PathBuf};

    use rust_renderer::asset::{
        asset_manager::AssetManager,
        asset_types::{Asset, AssetKind, AssetMetadata},
        importer::{
            AssetImporter,
            gltf_importer::GltfImporter,
            image_file_importer::{ColorSpace, ImageFileImporter},
        },
        provenance::{AssetTypeStamp, ImportProvenance, SourceFingerprint},
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
        assert_eq!(provenance.importer_id, GltfImporter::IMPORTER_ID);
        assert_eq!(provenance.importer_version, GltfImporter::IMPORTER_VERSION);

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

        assert_eq!(provenance.importer_id, ImageFileImporter::IMPORTER_ID);
        assert_eq!(provenance.importer_version, ImageFileImporter::IMPORTER_VERSION);
        assert!(provenance.sources.iter().any(|s| s.path.as_path() == Path::new(IMAGE_PATH)));
    }

    /// A single import may produce several asset kinds; all of them must be
    /// stamped with their current schema version so a shared-type change can
    /// invalidate the batch regardless of which importer wrote it.
    #[test]
    fn gltf_provenance_stamps_all_produced_asset_types() {
        let manager = AssetManager::new();
        manager.import(GltfImporter::new(GLTF_PATH)).expect("import should succeed");

        let database = manager.get_database();
        let provenance = database.values().next().expect("assets should be imported")
            .read().unwrap().metadata().provenance.clone().expect("provenance should be stamped");

        let mut kinds: Vec<AssetKind> = provenance.asset_types.iter().map(|s| s.kind).collect();
        kinds.sort_by_key(|kind| *kind as u32);
        assert_eq!(kinds, vec![AssetKind::StaticMesh, AssetKind::Texture, AssetKind::Scene]);

        for stamp in &provenance.asset_types {
            assert_eq!(stamp.version, stamp.kind.current_schema_version());
            assert!(stamp.is_current());
        }
        assert!(provenance.is_fresh(), "unchanged sources with current schemas are fresh");
    }

    /// A schema bump for any produced kind invalidates the batch, even when the
    /// importer id/version and sources are unchanged.
    #[test]
    fn stale_when_asset_type_schema_version_mismatches() {
        let dir = std::env::temp_dir().join("rust_renderer_type_schema");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("source.bin");
        std::fs::write(&path, b"data").unwrap();
        let source = SourceFingerprint::of(&path).expect("fingerprint should succeed");

        let provenance_with = |asset_types: Vec<AssetTypeStamp>| ImportProvenance {
            importer_id: "test".into(),
            importer_version: 1,
            imported_unix_ms: 0,
            sources: vec![source.clone()],
            asset_types,
        };

        // No type stamps (e.g. legacy provenance) -> stale.
        assert!(!provenance_with(Vec::new()).is_fresh());

        // A produced kind recorded with a non-current version -> stale.
        let bump = AssetKind::Texture.current_schema_version() + 1;
        assert!(!provenance_with(vec![AssetTypeStamp { kind: AssetKind::Texture, version: bump }]).is_fresh());

        // Current version -> fresh.
        assert!(provenance_with(vec![AssetTypeStamp {
            kind: AssetKind::Texture,
            version: AssetKind::Texture.current_schema_version(),
        }]).is_fresh());

        std::fs::remove_dir_all(&dir).ok();
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
