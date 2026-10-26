use std::collections::HashMap;

use crate::renderer::{
    mesh::vertex_types::VertexType,
    pipeline::key::VertexLayoutKey,
};

/// Owned vertex-buffer layout registered with the pipeline library.
#[derive(Clone)]
pub struct VertexLayout {
    buffers: Vec<wgpu::VertexBufferLayout<'static>>,
}

impl VertexLayout {
    pub fn new(buffers: Vec<wgpu::VertexBufferLayout<'static>>) -> Self {
        Self { buffers }
    }

    pub fn buffers(&self) -> &[wgpu::VertexBufferLayout<'static>] { &self.buffers }
}

/// Maps stable [`VertexLayoutKey`] values to owned vertex layouts.
pub struct VertexLayoutRegistry {
    layouts: HashMap<VertexLayoutKey, VertexLayout>,
}

impl VertexLayoutRegistry {
    pub fn new() -> Self {
        let mut ret = Self { layouts: HashMap::new() };
        ret.register(
            VertexLayoutKey::BASIC,
            VertexLayout::new(VertexType::get_vertex_buffer_layout(VertexType::Basic).to_vec()),
        );
        ret.register(
            VertexLayoutKey::POSITION_ONLY,
            VertexLayout::new(VertexType::get_vertex_buffer_layout(VertexType::PositionOnly).to_vec()),
        );
        ret.register(VertexLayoutKey::EMPTY, VertexLayout::new(Vec::new()));
        ret
    }

    pub fn register(&mut self, key: VertexLayoutKey, layout: VertexLayout) {
        self.layouts.insert(key, layout);
    }

    pub fn get(&self, key: VertexLayoutKey) -> &VertexLayout {
        self.layouts
            .get(&key)
            .unwrap_or_else(|| panic!("vertex layout {:?} is not registered", key))
    }

    pub fn try_get(&self, key: VertexLayoutKey) -> Option<&VertexLayout> {
        self.layouts.get(&key)
    }
}

impl Default for VertexLayoutRegistry {
    fn default() -> Self { Self::new() }
}
