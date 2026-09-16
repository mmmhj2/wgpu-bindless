#[cfg(test)]
mod test {
    use std::path::{Path, PathBuf};

use rust_renderer::asset::{asset_manager::AssetManager, asset_types::Asset, importer::gltf_importer::GltfImporter};

    #[test]
    fn import_gltf_test() {
        let test_filename: PathBuf;
        {
            let asset_manager = AssetManager::new();
            asset_manager.import(Path::new("resource/test_two_cubes.glb"), GltfImporter::new());
            asset_manager.save_to_disk().unwrap();
            {
                let db = asset_manager.get_database();
                let a = db.iter().next().expect("some asset should present");
                test_filename = a.1.as_ref().read().unwrap().get_asset_file_path().expect("asset path should present").into();
            }
        }

        Asset::load_from_disk(&test_filename).unwrap();
    }
}
