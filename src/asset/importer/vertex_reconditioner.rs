use cgmath::{InnerSpace, Zero};
    use crate::asset::asset_types::vertex::{VertexBufferOthers, VertexBufferPosition};

/// Errors from recalculating vertex tangents.
#[derive(Debug, thiserror::Error)]
pub enum ReconditionerError {
    #[error("position buffer has {0} vertices, which is not a multiple of 3")]
    NonTriangularPositions(usize),
    #[error("index buffer has {0} indices, which is not a multiple of 3")]
    NonTriangularIndices(usize),
}

/// Mark that a mesh could be reconditioned
pub trait VertexReconditionable {
    fn get_position_buffer(&self) -> &Vec<VertexBufferPosition>;
    fn get_attribute_buffer(&self) -> &Vec<VertexBufferOthers>;
    fn get_index_buffer(&self) -> Option<&Vec<u32>>;
}

/// Mark that a mesh can be reconditioned, and its attribute buffer can be written.
pub trait VertexReconditionableAttributeWrite {
    fn get_attribute_buffer_mut(&mut self) -> &mut Vec<VertexBufferOthers>;
}

/// Mark that a mesh can be reconditioned, and its position buffer can be written.
pub trait VertexReconditionablePositionWrite {
    fn get_position_buffer_mut(&mut self) -> &mut Vec<VertexBufferPosition>;
}

pub trait TangentRecalculator : VertexReconditionable + VertexReconditionableAttributeWrite {
    /// Performs Gram-Schmidt normalization and write tangent attribute 
    fn renormalize_and_write(&mut self, tangents1: Vec<cgmath::Vector3<f32>>, tangents2: Vec<cgmath::Vector3<f32>>) {
        let attribute = self.get_attribute_buffer_mut();
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

    fn recalculate_tangents_unindexed(&self) -> Result<(Vec<cgmath::Vector3<f32>>, Vec<cgmath::Vector3<f32>>), ReconditionerError>{
        let position = self.get_position_buffer();
        let attribute = self.get_attribute_buffer();

        if position.len() % 3 != 0 {
            return Err(ReconditionerError::NonTriangularPositions(position.len()));
        }

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

            tangents1[i + 0] = sdir;
            tangents1[i + 1] = sdir;
            tangents1[i + 2] = sdir;

            tangents2[i + 0] = tdir;
            tangents2[i + 1] = tdir;
            tangents2[i + 2] = tdir;
        }

        return Ok((tangents1, tangents2));
    }

    fn recalculate_tangents_indexed(&self)  -> Result<(Vec<cgmath::Vector3<f32>>, Vec<cgmath::Vector3<f32>>), ReconditionerError> {
        let index_buffer = self.get_index_buffer()
            .ok_or(ReconditionerError::NonTriangularIndices(0))?;

        if index_buffer.len() % 3 != 0 {
            return Err(ReconditionerError::NonTriangularIndices(index_buffer.len()));
        }

        let position = self.get_position_buffer();
        let attribute = self.get_attribute_buffer();

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
            
            tangents1[index_buffer[i + 0] as usize] += sdir;
            tangents1[index_buffer[i + 1] as usize] += sdir;
            tangents1[index_buffer[i + 2] as usize] += sdir;

            tangents2[index_buffer[i + 0] as usize] += tdir;
            tangents2[index_buffer[i + 1] as usize] += tdir;
            tangents2[index_buffer[i + 2] as usize] += tdir;
        }

        return Ok((tangents1, tangents2));
    }

    /// Clear and recalculate tangents of all vertices.
    /// 
    /// This method relies on correct normals and texcoords of vertices, asserting that the triangles are wound counterclockwise.
    /// Uses the algorithm presented in
    /// https://terathon.com/blog/tangent-space.html
    fn recalculate_tangents(&mut self) -> Result<(), ReconditionerError> {
        let (t1, t2) = match self.get_index_buffer() {
            Some(_) => self.recalculate_tangents_indexed()?,
            None => self.recalculate_tangents_unindexed()?
        };
        self.renormalize_and_write(t1, t2);
        Ok(())
    }
}

pub trait VertexColorApplyScale : VertexReconditionable + VertexReconditionableAttributeWrite {
    fn rescale_vertex_color(&mut self, scale: [f32; 4]) {
        for v in self.get_attribute_buffer_mut() {
            for i in 0..4 {
                v.color[i] *= scale[i]
            }
        }
    }
}

pub trait NormalRecalculator : VertexReconditionable + VertexReconditionableAttributeWrite {
    /// Recompute normalized vertex normals from the triangle winding.
    ///
    /// Face normals are weighted by their triangle surface.
    /// Vertices that no triangle references keep their existing normal.
    ///
    /// Triangles are assumed to be wound counterclockwise.
    fn recalculate_normals(&mut self) -> Result<(), ReconditionerError> {
        let positions: Vec<cgmath::Vector3<f32>> = self
            .get_position_buffer()
            .iter()
            .map(|vertex| vertex.position.into())
            .collect();

        let triangles: Vec<[usize; 3]> = match self.get_index_buffer() {
            Some(indices) => {
                if indices.len() % 3 != 0 {
                    return Err(ReconditionerError::NonTriangularIndices(indices.len()));
                }
                indices
                    .chunks_exact(3)
                    .map(|triangle| [triangle[0] as usize, triangle[1] as usize, triangle[2] as usize])
                    .collect()
            }
            None => {
                if positions.len() % 3 != 0 {
                    return Err(ReconditionerError::NonTriangularPositions(positions.len()));
                }
                (0..positions.len())
                    .collect::<Vec<_>>()
                    .chunks_exact(3)
                    .map(|triangle| [triangle[0], triangle[1], triangle[2]])
                    .collect()
            }
        };

        let mut accumulated = vec![cgmath::Vector3::<f32>::zero(); positions.len()];
        for [a, b, c] in triangles {
            let face = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
            accumulated[a] += face;
            accumulated[b] += face;
            accumulated[c] += face;
        }

        for (attribute, normal) in self.get_attribute_buffer_mut().iter_mut().zip(accumulated) {
            if normal.magnitude2() > 0.0 {
                attribute.normal = normal.normalize().into();
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
use crate::asset::asset_types::vertex::{VertexBufferOthers, VertexBufferPosition};
    use super::*;

    struct Dummy {
        vp: Vec<VertexBufferPosition>,
        va: Vec<VertexBufferOthers>,
        vi: Option<Vec<u32>>
    }

    impl VertexReconditionable for Dummy {
        fn get_position_buffer(&self) -> &Vec<VertexBufferPosition> {
            &self.vp
        }
        fn get_attribute_buffer(&self) -> &Vec<VertexBufferOthers> {
            &self.va
        }
        fn get_index_buffer(&self) -> Option<&Vec<u32>> {
            self.vi.as_ref()
        }
    }

    impl VertexReconditionableAttributeWrite for Dummy {
        fn get_attribute_buffer_mut(&mut self) -> &mut Vec<VertexBufferOthers> {
            &mut self.va
        }
    }
    impl TangentRecalculator for Dummy {}
    impl NormalRecalculator for Dummy {}

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

        d.recalculate_tangents().expect("tangent recalculation should succeed");
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

        d.recalculate_tangents().expect("tangent recalculation should succeed");
        assert_eq!(d.va[0].tangent, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(d.va[1].tangent, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(d.va[2].tangent, [0.0, 1.0, 0.0, 1.0]);
    }

    fn dummy_triangle(vi: Option<Vec<u32>>) -> Dummy {
        Dummy {
            vp: vec![
                VertexBufferPosition { position: [0.0, 0.0, 0.0] },
                VertexBufferPosition { position: [1.0, 0.0, 0.0] },
                VertexBufferPosition { position: [0.0, 1.0, 0.0] },
            ],
            va: vec![
                VertexBufferOthers { color: [1.0; 4], normal: [0.0; 3], tangent: [0.0; 4], uv0: [0.0, 0.0] },
                VertexBufferOthers { color: [1.0; 4], normal: [0.0; 3], tangent: [0.0; 4], uv0: [1.0, 0.0] },
                VertexBufferOthers { color: [1.0; 4], normal: [0.0; 3], tangent: [0.0; 4], uv0: [0.0, 1.0] },
            ],
            vi,
        }
    }

    #[test]
    fn test_normal_recalculator_unindexed() {
        let mut d = dummy_triangle(None);
        d.recalculate_normals().expect("normal recalculation should succeed");

        // Counterclockwise in the XY plane faces +Z.
        for vertex in &d.va {
            assert_eq!(vertex.normal, [0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn test_normal_recalculator_indexed() {
        let mut d = dummy_triangle(Some(vec![0, 1, 2]));
        d.recalculate_normals().expect("normal recalculation should succeed");

        for vertex in &d.va {
            assert_eq!(vertex.normal, [0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn test_normal_recalculator_weights_by_area() {
        // Two triangles sharing vertex 0: one large (+Z), one small (-Z).
        // The shared vertex must keep the +Z normal because the large face wins.
        let mut d = Dummy {
            vp: vec![
                VertexBufferPosition { position: [0.0, 0.0, 0.0] },
                VertexBufferPosition { position: [2.0, 0.0, 0.0] },
                VertexBufferPosition { position: [0.0, 2.0, 0.0] },
                VertexBufferPosition { position: [0.5, 0.0, 0.0] },
                VertexBufferPosition { position: [0.0, 0.5, 0.0] },
            ],
            va: vec![
                VertexBufferOthers { color: [1.0; 4], normal: [0.0; 3], tangent: [0.0; 4], uv0: [0.0, 0.0] },
                VertexBufferOthers { color: [1.0; 4], normal: [0.0; 3], tangent: [0.0; 4], uv0: [0.0, 0.0] },
                VertexBufferOthers { color: [1.0; 4], normal: [0.0; 3], tangent: [0.0; 4], uv0: [0.0, 0.0] },
                VertexBufferOthers { color: [1.0; 4], normal: [0.0; 3], tangent: [0.0; 4], uv0: [0.0, 0.0] },
                VertexBufferOthers { color: [1.0; 4], normal: [0.0; 3], tangent: [0.0; 4], uv0: [0.0, 0.0] },
            ],
            vi: Some(vec![0, 1, 2, 0, 4, 3]),
        };

        d.recalculate_normals().expect("normal recalculation should succeed");

        assert_eq!(d.va[0].normal, [0.0, 0.0, 1.0], "the large face must dominate the shared vertex");
        assert_eq!(d.va[3].normal, [0.0, 0.0, -1.0], "the small face still faces -Z");
    }
}
