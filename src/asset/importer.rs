use std::{collections::HashMap, sync::{Arc, RwLock}};

use crate::asset::asset_types::Asset;

pub mod gltf_importer;
pub mod image_file_importer;
pub mod vertex_reconditioner;

/// Sink that collects assets produced by a single import operation.
///
/// The source(s) to import are owned by the importer itself; this context only
/// gathers the resulting assets so the `AssetManager` can merge them.
pub struct ImporterContext {
    pub asset_batch: RwLock<HashMap<String, Arc<RwLock<Asset>>>>,
}

impl ImporterContext {
    pub fn new() -> Self {
        Self { asset_batch: HashMap::new().into() }
    }
}

impl Default for ImporterContext {
    fn default() -> Self {
        Self::new()
    }
}

pub trait AssetImporter {
    fn import(self, context: &ImporterContext);
}
