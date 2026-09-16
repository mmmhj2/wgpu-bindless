pub mod static_mesh_asset;
pub mod texture_asset;

use std::fs::File;

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::asset::{asset_types::{static_mesh_asset::StaticMeshAsset, texture_asset::TextureAsset}, importer::ImporterContext};

#[derive(Serialize, Deserialize, Debug)]
pub struct AssetMetadata {
    pub asset_name: String,
    pub asset_file_path: Option<Box<std::path::Path>>
}

#[derive(Serialize, Deserialize, Debug)]
pub enum AssetData {
    StaticMeshAsset(StaticMeshAsset),
    TextureAsset(TextureAsset)
}

trait ConcreteAssetType: Serialize + DeserializeOwned {
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Asset {
    metadata: AssetMetadata,
    data    : AssetData
}

impl Asset {
    pub fn new(metadata: AssetMetadata, data: AssetData) -> Self {
        Self { metadata, data }
    }
    
    pub fn get_asset_file_path(&self) -> Option<&std::path::Path> { 
        match &self.metadata.asset_file_path {
            Some(p) => Some(p.as_ref()),
            None => None
        }
    }

    pub fn get_data(&self) -> &AssetData { &self.data }
    pub fn get_data_mut(&mut self) -> &mut AssetData { &mut self.data }
    
    pub fn save_to_disk(&self) -> Result<(), ()> {
        let disk_path = self.metadata.asset_file_path.as_ref().expect("This asset cannot be saved to disk.");
        serde_json::to_writer(File::create(disk_path).expect(""), self).expect("failed to serialize asset");
        Ok(())
    }

    pub fn load_from_disk(path: &std::path::Path) -> Result<Self, ()> {
        let reader = File::open(path).expect("cannot open asset file");
        Ok(serde_json::from_reader(reader).expect("invalid asset file"))
    }
}
