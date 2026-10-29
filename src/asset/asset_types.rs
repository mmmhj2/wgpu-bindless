pub mod animation_asset;
pub mod scene_asset;
pub mod static_mesh_asset;
pub mod skinned_mesh_asset;
pub mod texture_asset;
pub mod vertex;

use std::{fs::File, io::{Read, Write}};

use image::EncodableLayout;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::asset::{asset_types::{scene_asset::SceneAsset, skinned_mesh_asset::SkinnedMeshAsset, static_mesh_asset::StaticMeshAsset, texture_asset::TextureAsset}, error::AssetIoError, provenance::ImportProvenance};

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
    /// large the asset is. Callers that tolerate corrupt or legacy files can
    /// ignore the error.
    pub fn peek(path: &std::path::Path) -> Result<Self, AssetIoError> {
        let file = File::open(path).map_err(|source| AssetIoError::Open { path: path.to_path_buf(), source })?;
        let mut archive = zip::ZipArchive::new(file)
            .map_err(|source| AssetIoError::Container { path: path.to_path_buf(), source })?;
        read_asset_entry(&mut archive, path, METADATA_ENTRY)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub enum AssetData {
    StaticMeshAssetType(StaticMeshAsset),
    TextureAssetType(TextureAsset),
    SceneAssetType(SceneAsset),
    SkinnedMeshAssetType(SkinnedMeshAsset),
    SkinnedAnimationAssetType(())
}

impl AssetData {
    /// The concrete kind of asset this payload holds.
    pub fn kind(&self) -> AssetKind {
        match self {
            AssetData::StaticMeshAssetType(_) => AssetKind::StaticMesh,
            AssetData::TextureAssetType(_) => AssetKind::Texture,
            AssetData::SceneAssetType(_) => AssetKind::Scene,
            AssetData::SkinnedMeshAssetType(_) => AssetKind::SkinnedMesh,
            AssetData::SkinnedAnimationAssetType(_) => AssetKind::SkinnedAnimation,
        }
    }
}

/// Discriminates the concrete asset type held by an [`AssetData`].
///
/// Needed to stamp provenance and detect schema changes.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssetKind {
    StaticMesh,
    Texture,
    Scene,
    SkinnedMesh,
    SkinnedAnimation
}

impl AssetKind {
    /// The schema version currently compiled for this asset kind.
    pub fn current_schema_version(self) -> u32 {
        match self {
            AssetKind::StaticMesh => <StaticMeshAsset as ConcreteAssetType>::SCHEMA_VERSION,
            AssetKind::Texture => <TextureAsset as ConcreteAssetType>::SCHEMA_VERSION,
            AssetKind::Scene => <SceneAsset as ConcreteAssetType>::SCHEMA_VERSION,
            AssetKind::SkinnedMesh => todo!(),
            AssetKind::SkinnedAnimation => todo!()
        }
    }
}

pub trait ConcreteAssetType: Serialize + DeserializeOwned {
    /// Version of this asset type's serialized layout and semantics.
    ///
    /// Bump whenever the layout or meaning of the type changes; every serialized
    /// asset of this kind then becomes stale and is re-imported.
    const SCHEMA_VERSION: u32;
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
    pub fn save_to_disk(&self) -> Result<(), AssetIoError> {
        let disk_path = self.metadata.asset_file_path.as_deref()
            .ok_or_else(|| AssetIoError::MissingFilePath(self.metadata.asset_name.clone()))?;
        write_asset_container(disk_path, self)
    }

    /// Load a full asset from its on-disk ZIP container.
    ///
    /// Callers that must tolerate corrupt or legacy files (e.g. directory
    /// scans) can handle the error themselves.
    pub fn load_from_disk(path: &std::path::Path) -> Result<Self, AssetIoError> {
        read_asset_container(path)
    }
}

/// Write an asset to `path` as a ZIP container with `metadata` / `payload` entries.
fn write_asset_container(path: &std::path::Path, asset: &Asset) -> Result<(), AssetIoError> {
    let file = File::create(path).map_err(|source| AssetIoError::Create { path: path.to_path_buf(), source })?;
    let mut archive = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored);

    let metadata = flexbuffers::to_vec(&asset.metadata)
        .map_err(|source| AssetIoError::Serialize { what: "asset metadata", source })?;
    archive.start_file(METADATA_ENTRY, options)
        .map_err(|source| AssetIoError::Entry { path: path.to_path_buf(), entry: METADATA_ENTRY.to_string(), source })?;
    archive.write_all(&metadata)
        .map_err(|source| AssetIoError::EntryWrite { path: path.to_path_buf(), entry: METADATA_ENTRY, source })?;

    let payload = flexbuffers::to_vec(&asset.data)
        .map_err(|source| AssetIoError::Serialize { what: "asset payload", source })?;
    archive.start_file(PAYLOAD_ENTRY, options)
        .map_err(|source| AssetIoError::Entry { path: path.to_path_buf(), entry: PAYLOAD_ENTRY.to_string(), source })?;
    archive.write_all(&payload)
        .map_err(|source| AssetIoError::EntryWrite { path: path.to_path_buf(), entry: PAYLOAD_ENTRY, source })?;

    archive.finish()
        .map_err(|source| AssetIoError::Finalize { path: path.to_path_buf(), source })?;
    Ok(())
}

/// Read and deserialize a single named entry from an asset container.
fn read_asset_entry<T: DeserializeOwned>(
    archive: &mut zip::ZipArchive<File>,
    path: &std::path::Path,
    name: &'static str,
) -> Result<T, AssetIoError> {
    let mut entry = archive.by_name(name)
        .map_err(|source| AssetIoError::Entry { path: path.to_path_buf(), entry: name.to_string(), source })?;
    let mut buffer = Vec::new();
    entry.read_to_end(&mut buffer)
        .map_err(|source| AssetIoError::EntryRead { path: path.to_path_buf(), entry: name.to_string(), source })?;
    flexbuffers::from_slice(buffer.as_slice())
        .map_err(|source| AssetIoError::Deserialize { what: name, path: path.to_path_buf(), source })
}

/// Load a full asset from its on-disk ZIP container.
fn read_asset_container(path: &std::path::Path) -> Result<Asset, AssetIoError> {
    let file = File::open(path).map_err(|source| AssetIoError::Open { path: path.to_path_buf(), source })?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|source| AssetIoError::Container { path: path.to_path_buf(), source })?;
    let metadata: AssetMetadata = read_asset_entry(&mut archive, path, METADATA_ENTRY)?;
    let data: AssetData = read_asset_entry(&mut archive, path, PAYLOAD_ENTRY)?;
    Ok(Asset::new(metadata, data))
}
