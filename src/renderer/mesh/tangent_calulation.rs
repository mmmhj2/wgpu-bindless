use crate::renderer::mesh::vertex_types::{VertexBufferOthers, VertexBufferPosition};

mod private {
    use cgmath::Zero;

use crate::renderer::mesh::immediate_mesh::ImmediateMeshBuilder;
    pub trait TangentRecalculatorImpl : super::CanRecaluclateTangent {
        fn renormalize_and_write(&mut self, tangents1: Vec<cgmath::Vector3<f32>>, tangents2: Vec<cgmath::Vector3<f32>>) {
            let attribute = self.get_attribute_buffer_tgt_mut();
            for i in 0..attribute.len()
            {
                let n = attribute[i].normal.into();
                let t = tangents1[i];
                
                // Gram-Schmidt orthogonalize
                let tgt: [f32; 3] = cgmath::InnerSpace::normalize(t - n * cgmath::dot(n, t) ).into();
                let w = if cgmath::dot(n.cross(t), tangents2[i]) < 0.0 { -1.0 } else { 1.0 };
                // Calculate handedness
                attribute[i].tangent = [tgt[0], tgt[1], tgt[2], w];
            }
        }

        fn solve_direction(
            v: [cgmath::Vector3<f32>; 3],
            w: [cgmath::Vector2<f32>; 3]
        ) -> (cgmath::Vector3<f32>, cgmath::Vector3<f32>) {
            let x1 = v[1].x - v[0].x;
            let x2 = v[2].x - v[0].x;
            let y1 = v[1].y - v[0].y;
            let y2 = v[2].y - v[0].y;
            let z1 = v[1].z - v[0].z;
            let z2 = v[2].z - v[0].z;
            
            let s1 = w[1].x - w[0].x;
            let s2 = w[2].x - w[0].x;
            let t1 = w[1].y - w[0].y;
            let t2 = w[2].y - w[0].y;
            
            let r = 1.0f32 / (s1 * t2 - s2 * t1);
            let sdir = cgmath::vec3((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r,
                    (t2 * z1 - t1 * z2) * r);
            let tdir = cgmath::vec3((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r,
                    (s1 * z2 - s2 * z1) * r);

            (sdir, tdir)
        }

        fn recalculate_tangents_unindexed(&mut self) {
            assert!(self.get_index_buffer_tgt().is_none());

            assert!(self.get_position_buffer_tgt().len() % 3 == 0, "Triangluar mesh is not closed");

            let position = self.get_position_buffer_tgt();
            let attribute = self.get_attribute_buffer_tgt();

            let mut tangents1 = Vec::new();
            tangents1.resize(position.len(), cgmath::Vector3::<f32>::zero());
            let mut tangents2 = Vec::new();
            tangents2.resize(position.len(), cgmath::Vector3::<f32>::zero());

            for i in (0..position.len()).step_by(3) {
                let v: [cgmath::Vector3<f32>; 3] = [
                    position[i].position.into(),
                    position[i + 1].position.into(),
                    position[i + 2].position.into()
                ];
                let w: [cgmath::Vector2<f32>; 3] = [
                    attribute[i].uv0.into(),
                    attribute[i + 1].uv0.into(),
                    attribute[i + 2].uv0.into()
                ];

                let (sdir, tdir) = Self::solve_direction(v, w);

                tangents1[i] = sdir;
                tangents1[i + 1] = sdir;
                tangents1[i + 2] = sdir;

                tangents2[i] = tdir;
                tangents2[i] = tdir;
                tangents2[i] = tdir;
            }

            Self::renormalize_and_write(self, tangents1, tangents2);
        }

        fn recalculate_tangents_indexed(&mut self) {
            let index_buffer = self.get_index_buffer_tgt().unwrap();

            assert!(index_buffer.len() % 3 == 0, "Triangluar mesh is not closed");

            let position = self.get_position_buffer_tgt();
            let attribute = self.get_attribute_buffer_tgt();

            let mut tangents1 = Vec::new();
            tangents1.resize(position.len(), cgmath::Vector3::<f32>::zero());
            let mut tangents2 = Vec::new();
            tangents2.resize(position.len(), cgmath::Vector3::<f32>::zero());

            for i in (0..index_buffer.len()).step_by(3) {
                let v: [cgmath::Vector3<f32>; 3] = [
                    position[index_buffer[i] as usize].position.into(),
                    position[index_buffer[i + 1] as usize].position.into(),
                    position[index_buffer[i + 2] as usize].position.into()
                ];
                let w: [cgmath::Vector2<f32>; 3] = [
                    attribute[index_buffer[i] as usize].uv0.into(),
                    attribute[index_buffer[i + 1] as usize].uv0.into(),
                    attribute[index_buffer[i + 2] as usize].uv0.into()
                ];

                let (sdir, tdir) = Self::solve_direction(v, w);
                
                tangents1[i] += sdir;
                tangents1[i + 1] += sdir;
                tangents1[i + 2] += sdir;

                tangents2[i] += tdir;
                tangents2[i] += tdir;
                tangents2[i] += tdir;
            }

            Self::renormalize_and_write(self, tangents1, tangents2);
        }
    }

    impl TangentRecalculatorImpl for ImmediateMeshBuilder {}
}

pub trait CanRecaluclateTangent {
    fn get_position_buffer_tgt(&self) -> &Vec<VertexBufferPosition>;
    fn get_attribute_buffer_tgt(&self) -> &Vec<VertexBufferOthers>;
    fn get_attribute_buffer_tgt_mut(&mut self) -> &mut Vec<VertexBufferOthers>;
    fn get_index_buffer_tgt(&self) -> Option<&Vec<u32>>;
}

pub trait TangentRecalculator : private::TangentRecalculatorImpl {
    /// Clear and recalculate tangents of all vertices.
    /// 
    /// This method relies on correct normals and texcoords of vertices, asserting that the triangles are wound counterclockwise.
    /// Uses the algorithm presented in
    /// https://terathon.com/blog/tangent-space.html
    fn recalculate_tangents(&mut self) -> () {
        match self.get_index_buffer_tgt() {
            Some(_) => self.recalculate_tangents_indexed(),
            None => self.recalculate_tangents_unindexed()
        }
    }
}

#[cfg(test)]
mod test {
    use crate::renderer::mesh::vertex_types::{VertexBufferOthers, VertexBufferPosition};
    use super::*;

    struct Dummy {
        vp: Vec<VertexBufferPosition>,
        va: Vec<VertexBufferOthers>,
        vi: Option<Vec<u32>>
    }

    impl CanRecaluclateTangent for Dummy {
        fn get_position_buffer_tgt(&self) -> &Vec<VertexBufferPosition> {
            &self.vp
        }
    
        fn get_attribute_buffer_tgt(&self) -> &Vec<VertexBufferOthers> {
            &self.va
        }
    
        fn get_attribute_buffer_tgt_mut (&mut self) -> &mut Vec<VertexBufferOthers> {
            &mut self.va
        }
    
        fn get_index_buffer_tgt(&self) -> Option<&Vec<u32>> {
            self.vi.as_ref()
        }
    }
    impl private::TangentRecalculatorImpl for Dummy {}
    impl TangentRecalculator for Dummy {}

    #[test]
    fn test_tangent_recalculator_unindexed () {
        let mut d = Dummy{
            vp: vec![
                VertexBufferPosition{ position: [0.0, 0.0, 0.0] },
                VertexBufferPosition{ position: [0.0, 1.0, 0.0] },
                VertexBufferPosition{ position: [0.0, 0.0, 1.0] }
            ],
            va: vec![
                VertexBufferOthers { color: [0.0, 0.0, 0.0, 1.0], normal: [1.0, 0.0, 0.0], tangent: [0.0, 0.0, 0.0, 0.0], uv0: [0.0, 0.0] },
                VertexBufferOthers { color: [0.0, 0.0, 0.0, 1.0], normal: [1.0, 0.0, 0.0], tangent: [0.0, 0.0, 0.0, 0.0], uv0: [1.0, 0.0] },
                VertexBufferOthers { color: [0.0, 0.0, 0.0, 1.0], normal: [1.0, 0.0, 0.0], tangent: [0.0, 0.0, 0.0, 0.0], uv0: [0.0, 1.0] },
            ],
            vi: None
        };

        d.recalculate_tangents();
        // Floating point comparison is tricky, but for testing it is good enough.
        assert_eq!(d.va[0].tangent, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(d.va[1].tangent, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(d.va[2].tangent, [0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_tangent_recalculator_indexed() {
        let mut d = Dummy{
            vp: vec![
                VertexBufferPosition{ position: [0.0, 0.0, 0.0] },
                VertexBufferPosition{ position: [0.0, 1.0, 0.0] },
                VertexBufferPosition{ position: [0.0, 0.0, 1.0] }
            ],
            va: vec![
                VertexBufferOthers { color: [0.0, 0.0, 0.0, 1.0], normal: [1.0, 0.0, 0.0], tangent: [0.0, 0.0, 0.0, 0.0], uv0: [0.0, 0.0] },
                VertexBufferOthers { color: [0.0, 0.0, 0.0, 1.0], normal: [1.0, 0.0, 0.0], tangent: [0.0, 0.0, 0.0, 0.0], uv0: [1.0, 0.0] },
                VertexBufferOthers { color: [0.0, 0.0, 0.0, 1.0], normal: [1.0, 0.0, 0.0], tangent: [0.0, 0.0, 0.0, 0.0], uv0: [0.0, 1.0] },
            ],
            vi: Some(vec![0, 1, 2])
        };

        d.recalculate_tangents();
        assert_eq!(d.va[0].tangent, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(d.va[1].tangent, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(d.va[2].tangent, [0.0, 1.0, 0.0, 1.0]);
    }
}
