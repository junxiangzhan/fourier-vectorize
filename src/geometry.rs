//! Points, polylines and the two Part 1 constructions of a cubic Bézier:
//! A. the cubic formula, evaluated as a matrix product (`CubicBezier::eval`);
//! B. recursive mid-point subdivision (`CubicBezier::flatten`).
//! The two share only the control points; neither calls the other.
use crate::{Error, Matrix, Vector};
use std::ops::{Add, Sub};

// Resource guard: at most 2^20 segments in one flattened cubic.
pub const MAX_SUBDIVISION_DEPTH: usize = 20;
const MAX_SAMPLES: usize = (1 << 20) + 1;

/// A position on the canvas, in pixels; x right, y down.
/// Position - position gives a displacement vector; position + vector gives
/// a position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point2 {
    coords: Vector<2>,
}
impl Point2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            coords: Vector::from_array([x, y]),
        }
    }
    pub fn x(self) -> f64 {
        self.coords[(0, 0)]
    }
    pub fn y(self) -> f64 {
        self.coords[(1, 0)]
    }
    pub fn is_finite(self) -> bool {
        self.x().is_finite() && self.y().is_finite()
    }
    /// Linear interpolation: self at t = 0, other at t = 1.
    pub fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            coords: self.coords * (1.0 - t) + other.coords * t,
        }
    }
    pub fn midpoint(self, other: Self) -> Self {
        self.lerp(other, 0.5)
    }
    pub fn distance(self, other: Self) -> f64 {
        (self - other).norm()
    }
    /// Rotation about an arbitrary point: translate to the origin, rotate,
    /// translate back. With y down, a positive angle looks clockwise.
    pub fn rotated_about(self, center: Self, angle: f64) -> Self {
        center + Matrix::rotation(angle) * (self - center)
    }
    pub fn scaled_about(self, center: Self, factor: f64) -> Self {
        center + (self - center) * factor
    }
}
impl Sub<Point2> for Point2 {
    type Output = Vector<2>;
    fn sub(self, rhs: Point2) -> Self::Output {
        self.coords - rhs.coords
    }
}
impl Add<Vector<2>> for Point2 {
    type Output = Point2;
    fn add(self, rhs: Vector<2>) -> Point2 {
        Point2 {
            coords: self.coords + rhs,
        }
    }
}
impl Sub<Vector<2>> for Point2 {
    type Output = Point2;
    fn sub(self, rhs: Vector<2>) -> Point2 {
        Point2 {
            coords: self.coords - rhs,
        }
    }
}

/// Shortest distance from `point` to the segment start..end (Part 1-2 gap
/// measurement).
pub fn distance_to_segment(point: Point2, start: Point2, end: Point2) -> f64 {
    let delta = end - start;
    let length_squared = delta.norm_squared();
    // A zero-length segment is just a point.
    if length_squared == 0.0 {
        return point.distance(start);
    }
    // Project onto the infinite line, then clamp the foot of the
    // perpendicular to the segment so endpoints are handled correctly.
    let t = ((point - start).dot(delta) / length_squared).clamp(0.0, 1.0);
    point.distance(start + delta * t)
}

/// Straight-line approximation of a curve or contour. `closed` adds an
/// implicit edge from the last vertex back to the first.
#[derive(Clone, Debug, PartialEq)]
pub struct Polyline {
    pub vertices: Vec<Point2>,
    pub closed: bool,
}
impl Polyline {
    /// Distance from `point` to the nearest segment of this polyline.
    pub fn distance_to(&self, point: Point2) -> Result<f64, Error> {
        if !point.is_finite() || self.vertices.iter().any(|p| !p.is_finite()) {
            return Err(Error::NonFinite);
        }
        let first = *self
            .vertices
            .first()
            .ok_or(Error::InvalidParameter("empty polyline"))?;
        let mut distance = point.distance(first);
        for edge in self.vertices.windows(2) {
            distance = distance.min(distance_to_segment(point, edge[0], edge[1]));
        }
        if self.closed {
            distance = distance.min(distance_to_segment(
                point,
                *self.vertices.last().unwrap(),
                first,
            ));
        }
        if !distance.is_finite() {
            return Err(Error::NonFinite);
        }
        Ok(distance)
    }
}

/// Cubic Bézier curve with control points b0..b3.
#[derive(Clone, Debug)]
pub struct CubicBezier {
    controls: [Point2; 4],
    /// Power-basis coefficients: row i multiplies t^i.
    coefficients: Matrix<4, 2>,
}
impl CubicBezier {
    pub fn new(controls: [Point2; 4]) -> Self {
        // Expanding C(t) = (1-t)^3 b0 + 3t(1-t)^2 b1 + 3t^2(1-t) b2 + t^3 b3
        // in powers of t gives C(t) = [1 t t^2 t^3] * basis * [b0 b1 b2 b3]^T.
        // Row i of `basis` holds the t^i coefficient of each control point.
        let basis = Matrix::from_rows([
            [1.0, 0.0, 0.0, 0.0],
            [-3.0, 3.0, 0.0, 0.0],
            [3.0, -6.0, 3.0, 0.0],
            [-1.0, 3.0, -3.0, 1.0],
        ]);
        let points = Matrix::from_rows(controls.map(|p| [p.x(), p.y()]));
        Self {
            controls,
            coefficients: basis * points,
        }
    }
    pub fn controls(&self) -> &[Point2; 4] {
        &self.controls
    }
    pub fn validate(&self) -> Result<(), Error> {
        if self.controls.iter().any(|p| !p.is_finite())
            || self
                .coefficients
                .rows()
                .iter()
                .flatten()
                .any(|v| !v.is_finite())
        {
            Err(Error::NonFinite)
        } else {
            Ok(())
        }
    }
    /// Construction A: evaluate the cubic formula at parameter t in [0, 1].
    /// Polynomial evaluation; valid geometry consumers validate finite inputs.
    pub fn eval(&self, t: f64) -> Point2 {
        let parameter = Vector::from_array([1.0, t, t * t, t * t * t]).transpose();
        let result = parameter * self.coefficients;
        Point2::new(result[(0, 0)], result[(0, 1)])
    }
    /// Construction B, one step: split the curve at t = 0.5 using only
    /// mid-points of the control polygon (de Casteljau with t = 0.5).
    pub fn split_midpoint(&self) -> (Self, Self) {
        let [p0, p1, p2, p3] = self.controls;
        // First level: mid-points of the three control-polygon sides.
        let a = p0.midpoint(p1);
        let b = p1.midpoint(p2);
        let c = p2.midpoint(p3);
        // Second level: mid-points of those.
        let d = a.midpoint(b);
        let e = b.midpoint(c);
        // Third level: f lies on the curve, at C(0.5).
        let f = d.midpoint(e);
        // Each half is again a cubic Bézier with its own control points.
        (Self::new([p0, a, d, f]), Self::new([f, e, c, p3]))
    }
    /// Construction B: subdivide recursively to `depth` and return the
    /// polyline through the end points of the 2^depth sub-curves.
    /// Depth 0 is the single chord b0 -> b3.
    pub fn flatten(&self, depth: usize) -> Result<Polyline, Error> {
        self.validate()?;
        if depth > MAX_SUBDIVISION_DEPTH {
            return Err(Error::SizeOverflow);
        }
        let mut vertices = Vec::new();
        vertices
            .try_reserve_exact((1usize << depth) + 1)
            .map_err(|_| Error::Allocation)?;
        vertices.push(self.controls[0]);
        self.append_endpoints(depth, &mut vertices);
        Ok(Polyline {
            vertices,
            closed: false,
        })
    }
    /// Depth-first, left half before right half, so end points arrive in
    /// curve order. The shared start point was pushed by the caller.
    fn append_endpoints(&self, depth: usize, output: &mut Vec<Point2>) {
        if depth == 0 {
            output.push(self.controls[3]);
        } else {
            let (left, right) = self.split_midpoint();
            left.append_endpoints(depth - 1, output);
            right.append_endpoints(depth - 1, output);
        }
    }
    /// Construction A sampled at evenly spaced t, finely enough that adjacent
    /// samples are closer than `max_spacing` pixels (Part 1-2 uses 0.1 px).
    ///
    /// Derivative lies in the convex hull of its three control differences.
    /// A derivative bound controls parameter spacing, with a 10% safety margin.
    pub fn sample_dense(&self, max_spacing: f64) -> Result<Vec<Point2>, Error> {
        self.validate()?;
        if !max_spacing.is_finite() || max_spacing <= 0.0 {
            return Err(Error::InvalidParameter(
                "sampling spacing must be finite and positive",
            ));
        }
        let bound = self
            .controls
            .windows(2)
            .map(|p| ((p[1] - p[0]) * 3.0).norm())
            .fold(0.0_f64, f64::max);
        let steps = (bound / (max_spacing * 0.9)).ceil().max(1.0);
        if !steps.is_finite() || steps >= MAX_SAMPLES as f64 {
            return Err(Error::SizeOverflow);
        }
        let steps = steps as usize;
        let mut points = Vec::new();
        points
            .try_reserve_exact(steps + 1)
            .map_err(|_| Error::Allocation)?;
        for i in 0..=steps {
            let point = self.eval(i as f64 / steps as f64);
            if !point.is_finite() {
                return Err(Error::NonFinite);
            }
            points.push(point);
        }
        // Verify the spacing guarantee on the actual samples, not just in theory.
        if points
            .windows(2)
            .any(|p| p[0].distance(p[1]) >= max_spacing)
        {
            return Err(Error::InvalidParameter(
                "sampling spacing lost to floating-point precision",
            ));
        }
        Ok(points)
    }
}
