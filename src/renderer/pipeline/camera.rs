use std::num::NonZero;

use cgmath::Point3;
use wgpu::BufferBinding;

use crate::renderer::device_interface::DeviceInterface;


const CAMERA_UBO_SIZE : usize = std::mem::size_of::<[[f32; 4]; 4]>() * 3;

pub trait HasViewMatrix {
    fn get_view_matrix (&self) -> cgmath::Matrix4<f32>;
}

pub trait HasViewProjectionMatrix : HasViewMatrix {
    const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::from_cols(
        cgmath::Vector4::new(1.0, 0.0, 0.0, 0.0),
        cgmath::Vector4::new(0.0, 1.0, 0.0, 0.0),
        cgmath::Vector4::new(0.0, 0.0, 0.5, 0.0),
        cgmath::Vector4::new(0.0, 0.0, 0.5, 1.0),
    );

    fn get_projection_matrix (&self) -> cgmath::Matrix4<f32>;

    /// Acquire the view-projection matrix that applys view and projection transform accordingly.
    /// 
    /// This function automatically applys correct transform from OpenGL NDC space of `cgmath` to the NDC space of WGSL. 
    fn get_vp_matrix (&self) -> cgmath::Matrix4<f32> {
        Self::OPENGL_TO_WGPU_MATRIX * self.get_projection_matrix() * self.get_view_matrix()
    }

    /// Get the u8 array containing the matrices that should be submitted to the GPU as UBO.
    fn get_ubo_content(&self) -> [u8; CAMERA_UBO_SIZE] {
        let arr: [[[f32; 4]; 4]; 3] = [
            self.get_view_matrix().into(),
            self.get_projection_matrix().into(),
            self.get_vp_matrix().into()
        ];
        bytemuck::cast(arr)
    }
}

/// Camera base class.
/// 
/// Contains spatial information for the view matrix.
#[derive(Clone, Debug)]
pub struct Camera {
    origin  : cgmath::Point3<f32>,
    target  : cgmath::Point3<f32>,
    up      : cgmath::Vector3<f32>
}

impl Default for Camera {
    /// Construct a default camera that is positioned at origin, looking at +z, and up vector pointing to +y.
    /// 
    /// This coordinate system is therefore the expected one recommended by [the GLTF specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#coordinate-system-and-units).
    fn default() -> Self {
        Self { 
            origin: cgmath::point3(0.0, 0.0, 0.0),
            target: cgmath::point3(0.0, 0.0, 1.0),
            up: cgmath::vec3(0.0, 1.0, 0.0)
        }
    }
}

impl Camera {
    pub fn new() -> Self {
        Self {
            origin: (-1.0, 0.0, 0.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: (0.0, 1.0, 0.0).into()
        }
    }

    pub fn get_origin(&self) -> cgmath::Point3<f32> { self.origin }
    pub fn get_target(&self) -> cgmath::Point3<f32> { self.target }
    pub fn get_up(&self) -> cgmath::Vector3<f32> { self.up }
}

impl HasViewMatrix for Camera {
    fn get_view_matrix (&self) -> cgmath::Matrix4<f32> {
        cgmath::Matrix4::look_at_rh(self.origin, self.target, self.up)
    }
}

/// Camera class for perspective cameras.
/// 
/// Contains additional information for the projection matrix
#[derive(Clone, Debug)]
pub struct CameraPerspective {
    camera              : Camera,
    fovy                : cgmath::Deg<f32>,
    aspect              : f32,
    near_clip_distance  : f32,
    far_clip_distance   : f32
}

impl CameraPerspective {
    pub fn get_underlying_camera(&self) -> &Camera { &self.camera }
}

impl Default for CameraPerspective {
    /// Constructa default perspective camera that:
    /// - has a vertical FoV of 90 degrees;
    /// - has an aspect ratio of 4:3;
    /// - has a clip space range of \[0.1, 1e5\].
    fn default() -> Self {
        Self {
            camera: Default::default(),
            fovy: cgmath::Deg(90.0),
            aspect: 800.0 / 600.0,
            near_clip_distance: 1e-1,
            far_clip_distance: 1e5
        }
    }
}

impl HasViewMatrix for CameraPerspective {
    fn get_view_matrix (&self) -> cgmath::Matrix4<f32> {
        self.camera.get_view_matrix()
    }
}

impl HasViewProjectionMatrix for CameraPerspective {
    fn get_projection_matrix (&self) -> cgmath::Matrix4<f32> {
        cgmath::perspective(self.fovy, self.aspect, self.near_clip_distance, self.far_clip_distance)
    }
}

pub struct CameraPerspectiveBuilder {
    camera: CameraPerspective
}

impl CameraPerspectiveBuilder {
    pub fn new() -> Self { 
        Self { camera: Default::default() } 
    }

    pub fn set_origin(&mut self, v: cgmath::Point3<f32>) -> &mut Self { self.camera.camera.origin = v; self }
    pub fn set_target(&mut self, v: cgmath::Point3<f32>) -> &mut Self { self.camera.camera.target = v; self }
    pub fn set_up(&mut self, v: cgmath::Vector3<f32>) -> &mut Self { self.camera.camera.up = v; self }
    pub fn set_vertical_fov(&mut self, fov: cgmath::Deg<f32>) -> &mut Self { self.camera.fovy = fov; self }
    pub fn set_aspect(&mut self, aspect: f32) -> &mut Self { self.camera.aspect = aspect; self }
    pub fn set_clip_distance(&mut self, near: f32, far: f32) -> &mut Self { self.camera.near_clip_distance = near; self.camera.far_clip_distance = far; self }
    pub fn build(&mut self) -> CameraPerspective { self.camera.clone() }
}


pub struct CameraManager {
    active_camera       : CameraPerspective,
    bind_group_layout   : wgpu::BindGroupLayout,
    ubo                 : wgpu::Buffer,
    bind_group          : wgpu::BindGroup
}

impl CameraManager {
    const BGL_CAMERA : [wgpu::BindGroupLayoutEntry; 1] = [
        wgpu::BindGroupLayoutEntry{
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(NonZero::new(CAMERA_UBO_SIZE as u64).unwrap())
            },
            count: None,
        }
    ];

    const BGLD_CAMERA: wgpu::BindGroupLayoutDescriptor<'static> = wgpu::BindGroupLayoutDescriptor{
        label: Some("Camera descriptor set layout"),
        entries: &Self::BGL_CAMERA,
    };

    /// Create a new camera manager with a default perspective camera.
    pub fn new(di: &DeviceInterface) -> Self {
        let bind_group_layout = di.get_device().create_bind_group_layout(&Self::BGLD_CAMERA);
        let buffer = di.get_device().create_buffer(&wgpu::BufferDescriptor{
            label: Some("Camera UBO"),
            size: CAMERA_UBO_SIZE as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let bind_group = di.get_device().create_bind_group(&wgpu::BindGroupDescriptor{
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(BufferBinding{
                        buffer: &buffer,
                        offset: 0,
                        size: Some((CAMERA_UBO_SIZE as u64).try_into().unwrap())
                    })
                }
            ]
        });

        Self {
            active_camera: Default::default(),
            bind_group_layout,
            bind_group,
            ubo: buffer
        }
    }

    pub fn get_bind_group_layout(&self) -> &wgpu::BindGroupLayout { &self.bind_group_layout }

    /// Prepare for rendering, and return the bind group pointing to the camera uniform buffer.
    pub fn prepare(&self, di: &DeviceInterface) -> &wgpu::BindGroup {
        di.get_queue().write_buffer(&self.ubo, 0, &self.active_camera.get_ubo_content());
        &self.bind_group
    }

    pub fn set_active_camera(&mut self, camera: CameraPerspective) { self.active_camera = camera }
    pub fn get_active_camera(&self) -> &CameraPerspective { &self.active_camera }
}
