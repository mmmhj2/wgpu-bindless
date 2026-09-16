use std::{collections::HashMap, fs::File, sync::{Arc, RwLock}};

use crate::asset::{asset_types::{self, Asset}, importer::{AssetImporter, ImporterContext}};

pub struct AssetManager {
    asset_database : RwLock<HashMap<String, Arc<RwLock<Asset>>>>
}

impl AssetManager {
    pub fn new() -> Self {
        Self {
            asset_database: HashMap::new().into()
        }
    }

    pub fn import<T: AssetImporter>(&self, path: &std::path::Path, importer: &mut T) {
        let context = ImporterContext::new(path);
        importer.import(&context);
        
        let mut database = self.asset_database.write().expect("Failed to acquire write lock for asset database");
        let pending_database = context.asset_batch.into_inner().expect("Failed to consume pending asset database.");
        database.extend(pending_database);
    }

    pub fn save_to_disk(&self) -> std::result::Result<(), ()> {
        let database = self.asset_database.read().expect("Failed to acquire read lock for asset database");
        for (_k, v) in database.iter() {
            let r = v.read().expect("Failed to acquire read lock for asset.");
            r.save_to_disk().expect("Failed to save asset to disk.");
        }
        Ok(())
    }
}
