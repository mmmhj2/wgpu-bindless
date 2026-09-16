#[cfg(test)]
mod test {
    use std::path::Path;

use rust_renderer::asset::{asset_manager::AssetManager, importer::gltf_importer::GltfImporter};

    #[test]
    fn import_gltf_test() {
        let asset_manager = AssetManager::new();
        let mut gltf_importer = GltfImporter::new();

        asset_manager.import(Path::new("resource/test_two_cubes.glb"), &mut gltf_importer);
        asset_manager.save_to_disk().unwrap();
    }
}
