use std::num::NonZero;

use bytemuck::Zeroable;

use crate::renderer::device_interface::DeviceInterface;

/// Maximum number of lights that can be uploaded to the GPU at once.
pub const MAX_LIGHTS: usize = 64;

/// Discriminator stored in the GPU light struct.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightKind {
    Directional = 0,
    Point = 1,
    Spot = 2,
}

/// CPU-side description of an analytic light source in world space.
/// 
/// It will be converted internally to a GPU UBO array element before submitting
/// to the device.
#[derive(Clone, Debug)]
pub struct Light {
    pub kind: LightKind,
    pub color: [f32; 3],
    pub intensity: f32,
    /// World-space position, used by point and spot lights.
    pub position: [f32; 3],
    /// World-space direction (pointing away from the light), used by directional and spot lights.
    pub direction: [f32; 3],
    /// Cut-off distance for point/spot lights; a non-positive value means no range limit.
    pub range: f32,
    pub inner_cone_radians: f32,
    pub outer_cone_radians: f32,
}

impl Light {
    pub fn directional(direction: [f32; 3], color: [f32; 3], intensity: f32) -> Self {
        Self {
            kind: LightKind::Directional,
            color,
            intensity,
            position: [0.0; 3],
            direction,
            range: 0.0,
            inner_cone_radians: 0.0,
            outer_cone_radians: 0.0,
        }
    }

    pub fn point(position: [f32; 3], color: [f32; 3], intensity: f32, range: f32) -> Self {
        Self {
            kind: LightKind::Point,
            color,
            intensity,
            position,
            direction: [0.0, -1.0, 0.0],
            range,
            inner_cone_radians: 0.0,
            outer_cone_radians: 0.0,
        }
    }

    pub fn spot(
        position: [f32; 3],
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        range: f32,
        inner_cone_radians: f32,
        outer_cone_radians: f32,
    ) -> Self {
        Self {
            kind: LightKind::Spot,
            color,
            intensity,
            position,
            direction,
            range,
            inner_cone_radians,
            outer_cone_radians,
        }
    }

    /// Convert into the packed GPU representation.
    ///
    /// Intensity is baked into the colour (i.e. the stored value is the emitted
    /// radiance), and the spot cone angles are pre-converted into cosines so the
    /// shader does not have to evaluate them per fragment.
    pub(crate) fn to_gpu(&self) -> GpuLight {
        let radiance = [
            self.color[0] * self.intensity,
            self.color[1] * self.intensity,
            self.color[2] * self.intensity,
        ];
        let (cos_inner, cos_outer) = match self.kind {
            LightKind::Spot => (
                self.inner_cone_radians.cos(),
                self.outer_cone_radians.cos(),
            ),
            _ => (1.0, 0.0),
        };

        GpuLight {
            color_kind: [radiance[0], radiance[1], radiance[2], self.kind as u32 as f32],
            position_range: [
                self.position[0],
                self.position[1],
                self.position[2],
                self.range,
            ],
            direction: [self.direction[0], self.direction[1], self.direction[2], 0.0],
            cones: [cos_inner, cos_outer, 0.0, 0.0],
        }
    }
}

/// GPU-side light layout. Must stay in sync with `GpuLight` in `bindless_header.wgsl`.
///
/// Every field is a `vec4` so that the std140/uniform layout matches the Rust
/// layout without any padding surprises.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuLight {
    /// `rgb` = emitted radiance (colour × intensity); `a` = [`LightKind`] as `f32`.
    pub color_kind: [f32; 4],
    /// `xyz` = world-space position; `a` = range.
    pub position_range: [f32; 4],
    /// `xyz` = world-space direction; `a` reserved.
    pub direction: [f32; 4],
    /// `x` = cosine of the inner cone angle; `y` = cosine of the outer cone angle;
    /// `zw` reserved for the future shadow-map parameters.
    pub cones: [f32; 4],
}

/// Uniform "header" block uploaded to the GPU.
/// 
/// Contains header information such as ambient term and light count and also 
/// the light array.
/// Must stay in sync with `Lighting` in `bindless_header.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LightingUniform {
    pub ambient: [f32; 3],
    pub light_count: u32,
    pub lights: [GpuLight; MAX_LIGHTS],
}

/// Owns the scene lighting uniform buffer and its bind group.
///
/// The buffer is persistent: lights are uploaded by rewriting it in place, so
/// the bind group can be created once and reused every frame.
/// 
/// TODO: Investigate sychronization implementation of WGPU. Maybe separate
/// buffer per frame in rotation can improve performance.
pub struct LightingManager {
    uniform_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    ambient: [f32; 3],
}

impl LightingManager {
    const BGL_LIGHTING: [wgpu::BindGroupLayoutEntry; 1] = [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZero::new(std::mem::size_of::<LightingUniform>() as u64),
            },
            count: None,
        },
    ];

    const BGLD_LIGHTING: wgpu::BindGroupLayoutDescriptor<'static> = wgpu::BindGroupLayoutDescriptor {
        label: Some("Lighting descriptor set layout"),
        entries: &Self::BGL_LIGHTING,
    };

    pub fn new(di: &DeviceInterface) -> Self {
        let device = di.get_device();

        let bind_group_layout = device.create_bind_group_layout(&Self::BGLD_LIGHTING);
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lighting uniform buffer"),
            size: std::mem::size_of::<LightingUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lighting bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            uniform_buffer,
            bind_group_layout,
            bind_group,
            ambient: [0.0; 3],
        }
    }

    pub fn get_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    pub fn get_bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn set_ambient(&mut self, ambient: [f32; 3]) {
        self.ambient = ambient;
    }

    /// Upload the current frame's lights, rewriting the whole uniform buffer.
    ///
    /// Panics if more than [`MAX_LIGHTS`] lights are supplied.
    pub fn upload_lights(&self, queue: &wgpu::Queue, lights: &[Light]) {
        assert!(
            lights.len() <= MAX_LIGHTS,
            "too many lights: {} supplied, at most {} supported",
            lights.len(),
            MAX_LIGHTS
        );

        let mut uniform = LightingUniform {
            ambient: self.ambient,
            light_count: lights.len() as u32,
            lights: [GpuLight::zeroed(); MAX_LIGHTS],
        };
        for (slot, light) in uniform.lights.iter_mut().zip(lights) {
            *slot = light.to_gpu();
        }

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn gpu_light_has_expected_layout() {
        assert_eq!(std::mem::size_of::<GpuLight>(), 64);
        assert_eq!(std::mem::size_of::<LightingUniform>(), 16 + 64 * MAX_LIGHTS);
    }

    #[test]
    fn directional_light_bakes_intensity_into_radiance() {
        let light = Light::directional([0.0, -1.0, 0.0], [0.5, 0.25, 0.125], 4.0);
        let gpu = light.to_gpu();

        assert_eq!(gpu.color_kind, [2.0, 1.0, 0.5, 0.0]);
        assert_eq!(gpu.direction, [0.0, -1.0, 0.0, 0.0]);
        assert_eq!(gpu.position_range, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn spot_light_precomputes_cone_cosines() {
        let light = Light::spot(
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            1.0,
            10.0,
            0.0,
            std::f32::consts::FRAC_PI_2,
        );
        let gpu = light.to_gpu();

        assert_eq!(gpu.color_kind[3], 2.0);
        assert_eq!(gpu.position_range, [1.0, 2.0, 3.0, 10.0]);
        assert_eq!(gpu.cones[0], 1.0);
        assert!(gpu.cones[1].abs() < 1e-6);
    }
}
