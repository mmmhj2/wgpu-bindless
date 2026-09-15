use std::collections::HashMap;

use crate::asset::asset_types::Asset;

pub mod gltf_importer;
pub mod vertex_reconditioner;

pub(crate) struct ImporterContext {
    pub asset_batch: HashMap<String, Asset>,
    pub imported_file_path: Box<std::path::Path>
}
