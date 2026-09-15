use std::collections::HashMap;

use crate::asset::asset_types::Asset;

pub struct AssetManager {
    asset_database : HashMap<String, Asset>
}

impl AssetManager {
    pub fn new() -> Self {
        Self {
            asset_database: HashMap::new()
        }
    }

    pub fn save_to_disk(&self) {
        for (_k, v) in self.asset_database.iter() {
            v.save_to_disk();
        }
    }
}
