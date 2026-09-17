pub mod static_mesh_asset;
pub mod texture_asset;

use std::{fs::File, io::{Read, Write}};

use image::EncodableLayout;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::asset::{asset_types::{static_mesh_asset::StaticMeshAsset, texture_asset::TextureAsset}, importer::ImporterContext};

#[derive(Serialize, Deserialize, Debug)]
pub struct AssetMetadata {
    pub asset_name: String,
    pub asset_file_path: Option<Box<std::path::Path>>
}

#[derive(Serialize, Deserialize, Debug)]
pub enum AssetData {
    StaticMeshAssetType(StaticMeshAsset),
    TextureAssetType(TextureAsset)
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
        let b = flexbuffers::to_vec(self).unwrap();
        File::create(disk_path).unwrap().write_all(b.as_bytes()).unwrap();
        Ok(())
    }

    pub fn load_from_disk(path: &std::path::Path) -> Result<Self, ()> {
        let mut buf = Vec::new();
        File::open(path).expect("cannot open asset file").read_to_end(&mut buf).unwrap();
        Ok(flexbuffers::from_buffer(&buf.as_slice()).unwrap())
    }
}
