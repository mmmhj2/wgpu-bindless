use std::{collections::HashMap, sync::{Arc, RwLock}};

use crate::asset::asset_types::Asset;

pub mod gltf_importer;
pub mod image_file_importer;
pub mod vertex_reconditioner;

pub struct ImporterContext {
    pub asset_batch: RwLock<HashMap<String, Arc<RwLock<Asset>>>>,
    pub imported_file_path: Box<std::path::Path>
}

impl ImporterContext {
    pub fn new(path: &std::path::Path) -> Self {
        Self { asset_batch: HashMap::new().into(), imported_file_path: Box::from(path) }
    }
}

pub trait AssetImporter {
    fn import(self, context: &ImporterContext);
}
