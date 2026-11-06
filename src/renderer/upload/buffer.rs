use crate::renderer::upload::error::UploadError;

/// Validate a buffer size against the device limits that apply to its usage.
fn validate_size(
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
    max_buffer_size: u64,
    max_storage_binding_size: u64,
) -> Result<(), UploadError> {
    if size > max_buffer_size {
        return Err(UploadError::BufferTooLarge {
            label: label.to_string(),
            needed: size,
            limit: max_buffer_size,
        });
    }

    if usage.contains(wgpu::BufferUsages::STORAGE) && size > max_storage_binding_size {
        return Err(UploadError::StorageBindingTooLarge {
            label: label.to_string(),
            needed: size,
            limit: max_storage_binding_size,
        });
    }

    Ok(())
}

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
    validate_size(
        label,
        size,
        usage,
        device.limits().max_buffer_size,
        u64::from(device.limits().max_storage_buffer_binding_size),
    )?;

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));

    Ok(buffer)
}
