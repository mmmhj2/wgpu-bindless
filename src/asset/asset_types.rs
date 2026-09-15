pub mod static_mesh_asset;

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::asset::asset_types::static_mesh_asset::StaticMeshAsset;

struct AssetMetadata {
    pub asset_name: String,
    pub asset_file_path: Option<Box<std::path::Path>>
}

pub enum AssetData {
    StaticMeshAsset(StaticMeshAsset)
}

trait ConcreteAssetType: Serialize + DeserializeOwned {
    fn import(path: &std::path::Path) -> Self;
}

pub struct Asset {
    metadata: AssetMetadata,
    data    : AssetData
}

impl Asset {
    pub fn get_asset_file_path(&self) -> Option<&std::path::Path> { 
        match &self.metadata.asset_file_path {
            Some(p) => Some(p.as_ref()),
            None => None
        }
    }

    pub fn get_data(&self) -> &AssetData { &self.data }
    pub fn get_data_mut(&mut self) -> &mut AssetData { &mut self.data }
    
    pub fn save_to_disk(&self) {
        todo!()
    }
}
