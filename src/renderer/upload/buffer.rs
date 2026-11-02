use crate::renderer::upload::error::UploadError;

/// Create a GPU buffer and fill it with `data` in one go.
///
/// Basically a wrapper around buffer creation (`create_buffer`) and
/// submission (`write_buffer`).
///
/// intented to handle asset-related upload-time buffers. Uniform buffers for
/// rendering (lighting, cameras, etc) are not concerned.
pub fn upload_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    usage: wgpu::BufferUsages,
    data: &[T],
) -> Result<wgpu::Buffer, UploadError> {
    if data.is_empty() {
        return Err(UploadError::EmptyBuffer(label.to_string()));
    }

    let size = std::mem::size_of_val(data) as u64;
    let limit = device.limits().max_buffer_size;
    if size > limit {
        return Err(UploadError::BufferTooLarge {
            label: label.to_string(),
            needed: size,
            limit,
        });
    }

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));

    Ok(buffer)
}
