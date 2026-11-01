//! Directed paths: sub-paths made of line and cubic segments, as a font or
//! vector file stores them. Direction matters for the non-zero rule, so
//! segment order is always preserved.
use crate::{CubicBezier, Error, Point2, Polyline};

/// One piece of a sub-path; its start is the previous segment's end.
#[derive(Clone, Debug, PartialEq)]
pub enum Segment {
    LineTo(Point2),
    CubicTo {
        control1: Point2,
        control2: Point2,
        end: Point2,
    },
}
/// One contour: a start point followed by connected segments.
#[derive(Clone, Debug, PartialEq)]
pub struct Subpath {
    start: Point2,
    segments: Vec<Segment>,
    closed: bool,
}
impl Subpath {
    pub fn new(start: Point2) -> Self {
        Self {
            start,
            segments: Vec::new(),
            closed: false,
        }
    }
    pub fn start(&self) -> Point2 {
        self.start
    }
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }
    pub fn is_closed(&self) -> bool {
        self.closed
    }
    pub fn line_to(&mut self, end: Point2) -> &mut Self {
        self.segments.push(Segment::LineTo(end));
        self
    }
    pub fn cubic_to(&mut self, control1: Point2, control2: Point2, end: Point2) -> &mut Self {
        self.segments.push(Segment::CubicTo {
            control1,
            control2,
            end,
        });
        self
    }
    pub fn close(&mut self) -> &mut Self {
        self.closed = true;
        self
    }
    /// The same contour traversed backwards (Part 2-3). No point moves: the
    /// segments are visited in reverse order, each cubic swaps its two
    /// control points, and every segment ends where it used to start.
    pub fn reversed(&self) -> Self {
        // Record each segment's start point; reversing needs them as ends.
        let mut current = self.start;
        let mut starts = Vec::with_capacity(self.segments.len());
        for segment in &self.segments {
            starts.push(current);
            current = match *segment {
                Segment::LineTo(end) | Segment::CubicTo { end, .. } => end,
            };
        }
        let mut result = Self::new(current);
        for (segment, start) in self.segments.iter().zip(starts).rev() {
            match *segment {
                Segment::LineTo(_) => {
                    result.line_to(start);
                }
                Segment::CubicTo {
                    control1, control2, ..
                } => {
                    result.cubic_to(control2, control1, start);
                }
            }
        }
        result.closed = self.closed;
        result
    }
    /// Applies a point transform to every start, control and end point.
    /// Valid for affine transforms, which map Bézier curves to Bézier curves.
    fn transformed(&self, transform: &impl Fn(Point2) -> Point2) -> Self {
        let mut result = Self::new(transform(self.start));
        for segment in &self.segments {
            match *segment {
                Segment::LineTo(end) => {
                    result.line_to(transform(end));
                }
                Segment::CubicTo {
                    control1,
                    control2,
                    end,
                } => {
                    result.cubic_to(transform(control1), transform(control2), transform(end));
                }
            }
        }
        result.closed = self.closed;
        result
    }
    /// Converts to a polyline: lines are kept, cubics are approximated by
    /// the Part 1 mid-point construction at the given depth.
    fn flatten(&self, depth: usize) -> Result<Polyline, Error> {
        if !self.start.is_finite() {
            return Err(Error::NonFinite);
        }
        let mut vertices = Vec::new();
        vertices
            .try_reserve_exact(1)
            .map_err(|_| Error::Allocation)?;
        vertices.push(self.start);
        let mut current = self.start;
        for segment in &self.segments {
            match *segment {
                Segment::LineTo(end) => {
                    if !end.is_finite() {
                        return Err(Error::NonFinite);
                    }
                    vertices.try_reserve(1).map_err(|_| Error::Allocation)?;
                    vertices.push(end);
                    current = end;
                }
                Segment::CubicTo {
                    control1,
                    control2,
                    end,
                } => {
                    let curve = CubicBezier::new([current, control1, control2, end]);
                    let flat = curve.flatten(depth)?;
                    // Skip the first vertex: it equals the current point,
                    // which is already in the list.
                    vertices
                        .try_reserve(flat.vertices.len() - 1)
                        .map_err(|_| Error::Allocation)?;
                    vertices.extend_from_slice(&flat.vertices[1..]);
                    current = end;
                }
            }
        }
        Ok(Polyline {
            vertices,
            closed: self.closed,
        })
    }
}
/// A shape made of one or more sub-paths (e.g. the two circles of an "O",
/// or the 96 wedges of the wheel), filled together as one path.
#[derive(Clone, Debug, PartialEq)]
pub struct Path {
    subpaths: Vec<Subpath>,
}
impl Path {
    pub fn new(subpaths: Vec<Subpath>) -> Self {
        Self { subpaths }
    }
    pub fn subpaths(&self) -> &[Subpath] {
        &self.subpaths
    }
    pub fn reversed(&self) -> Self {
        Self::new(self.subpaths.iter().map(Subpath::reversed).collect())
    }
    pub fn rotated_about(&self, center: Point2, angle: f64) -> Self {
        Self::new(
            self.subpaths
                .iter()
                .map(|sub| sub.transformed(&|p| p.rotated_about(center, angle)))
                .collect(),
        )
    }
    pub fn scaled_about(&self, center: Point2, factor: f64) -> Self {
        Self::new(
            self.subpaths
                .iter()
                .map(|sub| sub.transformed(&|p| p.scaled_about(center, factor)))
                .collect(),
        )
    }
    pub fn flatten(&self, depth: usize) -> Result<FlattenedPath, Error> {
        if depth > crate::geometry::MAX_SUBDIVISION_DEPTH {
            return Err(Error::SizeOverflow);
        }
        let mut subpaths = Vec::new();
        subpaths
            .try_reserve_exact(self.subpaths.len())
            .map_err(|_| Error::Allocation)?;
        for sub in &self.subpaths {
            subpaths.push(sub.flatten(depth)?);
        }
        Ok(FlattenedPath { subpaths })
    }
}
/// A path after flattening: every sub-path is a polyline, ready to fill.
#[derive(Clone, Debug, PartialEq)]
pub struct FlattenedPath {
    pub subpaths: Vec<Polyline>,
}
impl FlattenedPath {
    pub fn scaled_about(&self, center: Point2, factor: f64) -> Self {
        self.try_scaled_about(center, factor)
            .expect("geometry allocation failed")
    }
    // Fallible consumers keep allocation errors in their Result. The public
    // infallible transform retains ordinary Rust allocation semantics.
    pub(crate) fn try_scaled_about(&self, center: Point2, factor: f64) -> Result<Self, Error> {
        let mut subpaths = Vec::new();
        subpaths
            .try_reserve_exact(self.subpaths.len())
            .map_err(|_| Error::Allocation)?;
        for sub in &self.subpaths {
            let mut vertices = Vec::new();
            vertices
                .try_reserve_exact(sub.vertices.len())
                .map_err(|_| Error::Allocation)?;
            vertices.extend(sub.vertices.iter().map(|p| p.scaled_about(center, factor)));
            subpaths.push(Polyline {
                vertices,
                closed: sub.closed,
            });
        }
        Ok(Self { subpaths })
    }
}
