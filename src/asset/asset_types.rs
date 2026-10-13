pub mod scene_asset;
pub mod static_mesh_asset;
pub mod texture_asset;
pub mod vertex;

use std::{fs::File, io::{Read, Write}};

use image::EncodableLayout;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::asset::{asset_io_error::AssetIoError, asset_types::{scene_asset::SceneAsset, static_mesh_asset::StaticMeshAsset, texture_asset::TextureAsset}, provenance::ImportProvenance};

/// Name of the ZIP entry holding the serialized [`AssetMetadata`].
const METADATA_ENTRY: &str = "metadata";
/// Name of the ZIP entry holding the serialized [`AssetData`].
const PAYLOAD_ENTRY: &str = "payload";

#[derive(Serialize, Deserialize, Debug)]
pub struct AssetMetadata {
    pub asset_name: String,
    pub asset_file_path: Option<Box<std::path::Path>>,
    /// How this asset was produced. `None` for legacy or hand-built assets,
    /// which the loader always treats as stale.
    #[serde(default)]
    pub provenance: Option<ImportProvenance>
}

impl AssetMetadata {
    pub fn new(asset_name: String, asset_file_path: Option<Box<std::path::Path>>) -> Self {
        Self { asset_name, asset_file_path, provenance: None }
    }

    /// Read only the metadata entry of an on-disk asset container.
    ///
    /// The payload entry is never opened, so this stays cheap regardless of how
    /// large the asset is.
    pub fn peek(path: &std::path::Path) -> Self {
        Self::try_peek(path).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Fallible variant of [`peek`] for callers that tolerate corrupt or legacy files.
    pub fn try_peek(path: &std::path::Path) -> Result<Self, AssetIoError> {
        let file = File::open(path)
            .map_err(|e| AssetIoError(format!("cannot open asset file '{}': {e}", path.display())))?;
        let mut archive = zip::ZipArchive::new(file)
            .map_err(|e| AssetIoError(format!("asset file '{}' is not a valid archive: {e}", path.display())))?;
        read_asset_entry(&mut archive, METADATA_ENTRY)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub enum AssetData {
    StaticMeshAssetType(StaticMeshAsset),
    TextureAssetType(TextureAsset),
    SceneAssetType(SceneAsset)
}

pub trait ConcreteAssetType: Serialize + DeserializeOwned {
}

pub trait GpuAssetType: ConcreteAssetType {
    type GpuObjectType;
    fn upload(&self, name: &str, device: &wgpu::Device, queue: &wgpu::Queue) -> Self::GpuObjectType;
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

    pub fn metadata(&self) -> &AssetMetadata { &self.metadata }
    pub fn metadata_mut(&mut self) -> &mut AssetMetadata { &mut self.metadata }

    pub fn get_asset_file_path(&self) -> Option<&std::path::Path> { 
        match &self.metadata.asset_file_path {
            Some(p) => Some(p.as_ref()),
            None => None
        }
    }

    pub fn get_data(&self) -> &AssetData { &self.data }
    pub fn get_data_mut(&mut self) -> &mut AssetData { &mut self.data }
    
    /// Serialize the asset into its on-disk ZIP container.
    ///
    /// Metadata and payload are stored as two separate, uncompressed entries so
    /// that metadata can later be read without touching the payload.
    pub fn save_to_disk(&self) -> Result<(), ()> {
        let disk_path = self.metadata.asset_file_path.as_ref().expect("This asset cannot be saved to disk.").as_ref();
        write_asset_container(disk_path, self);
        Ok(())
    }

    pub fn load_from_disk(path: &std::path::Path) -> Self {
        read_asset_container(path).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Fallible load for callers that must tolerate corrupt or legacy files
    /// (e.g. directory scans).
    pub fn try_load_from_disk(path: &std::path::Path) -> Result<Self, AssetIoError> {
        read_asset_container(path)
    }
}

/// Write an asset to `path` as a ZIP container with `metadata` / `payload` entries.
fn write_asset_container(path: &std::path::Path, asset: &Asset) {
    let file = File::create(path)
        .unwrap_or_else(|e| panic!("cannot create asset file '{}': {e}", path.display()));
    let mut archive = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored);

    let metadata = flexbuffers::to_vec(&asset.metadata).expect("failed to serialize asset metadata");
    archive.start_file(METADATA_ENTRY, options).expect("failed to start asset metadata entry");
    archive.write_all(&metadata).expect("failed to write asset metadata");

    let payload = flexbuffers::to_vec(&asset.data).expect("failed to serialize asset payload");
    archive.start_file(PAYLOAD_ENTRY, options).expect("failed to start asset payload entry");
    archive.write_all(&payload).expect("failed to write asset payload");

    archive.finish().expect("failed to finalize asset container");
}

/// Read and deserialize a single named entry from an asset container.
fn read_asset_entry<T: DeserializeOwned>(archive: &mut zip::ZipArchive<File>, name: &str) -> Result<T, AssetIoError> {
    let mut entry = archive.by_name(name)
        .map_err(|e| AssetIoError(format!("asset container is missing a required entry '{name}': {e}")))?;
    let mut buffer = Vec::new();
    entry.read_to_end(&mut buffer)
        .map_err(|e| AssetIoError(format!("failed to read asset entry '{name}': {e}")))?;
    flexbuffers::from_slice(buffer.as_slice())
        .map_err(|e| AssetIoError(format!("failed to deserialize asset entry '{name}': {e}")))
}

/// Load a full asset from its on-disk ZIP container.
fn read_asset_container(path: &std::path::Path) -> Result<Asset, AssetIoError> {
    let file = File::open(path)
        .map_err(|e| AssetIoError(format!("cannot open asset file '{}': {e}", path.display())))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| AssetIoError(format!("asset file '{}' is not a valid archive: {e}", path.display())))?;
    let metadata: AssetMetadata = read_asset_entry(&mut archive, METADATA_ENTRY)?;
    let data: AssetData = read_asset_entry(&mut archive, PAYLOAD_ENTRY)?;
    Ok(Asset::new(metadata, data))
}
