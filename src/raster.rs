//! Part 2: scanline fill with the non-zero winding rule, plus supersampling
//! (Part 3) built on the same fill.
//!
//! Sign convention, fixed once for the whole program: screen coordinates have
//! y pointing down. Walking a row from left to right, an edge that passes
//! through the row going UP (b.y < a.y) contributes +1 and an edge going DOWN
//! contributes -1. Hence a screen-clockwise contour has winding +1 inside.
use crate::{Error, FlattenedPath, Image, Point2};

/// One straight edge of a flattened contour, kept in travel order a -> b so
/// that its direction (and therefore its winding sign) is preserved.
#[derive(Clone, Copy)]
struct Edge {
    a: Point2,
    b: Point2,
}
impl Edge {
    /// Where this edge crosses the horizontal line at `y`, with its signed
    /// winding contribution, or None if the edge does not span `y`.
    fn crossing(self, y: f64) -> Option<(f64, i32)> {
        let low = self.a.y().min(self.b.y());
        let high = self.a.y().max(self.b.y());
        // Half-open span [low, high): a vertex shared by two edges is counted
        // by exactly one of them, so it is never double-counted.
        if y < low || y >= high {
            return None;
        }
        // Canonical low-to-high interpolation gives identical intersections
        // when a contour is reversed and avoids drift on simple diagonals.
        let (a, b) = if self.a.y() < self.b.y() {
            (self.a, self.b)
        } else {
            (self.b, self.a)
        };
        let t = (y - a.y()) / (b.y() - a.y());
        let x = a.x() + (b.x() - a.x()) * t;
        // Scanning left to right: an upward (negative dy) edge enters CW fill.
        Some((x, if self.b.y() < self.a.y() { 1 } else { -1 }))
    }
}
/// Collects the edges of every closed sub-path into one list. Sub-paths are
/// not filled separately: their crossings are summed together, which is what
/// lets an inner contour add to or cancel the outer one.
fn edges(path: &FlattenedPath) -> Result<Vec<Edge>, Error> {
    let len = path.subpaths.iter().try_fold(0usize, |sum, sub| {
        sum.checked_add(sub.vertices.len())
            .ok_or(Error::SizeOverflow)
    })?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(len)
        .map_err(|_| Error::Allocation)?;
    for sub in &path.subpaths {
        if !sub.closed {
            return Err(Error::OpenPath);
        }
        if sub.vertices.iter().any(|p| !p.is_finite()) {
            return Err(Error::NonFinite);
        }
        let n = sub.vertices.len();
        // (i + 1) % n adds the closing edge from the last vertex to the first.
        for i in 0..n {
            let a = sub.vertices[i];
            let b = sub.vertices[(i + 1) % n];
            if !(b.x() - a.x()).is_finite() || !(b.y() - a.y()).is_finite() {
                return Err(Error::NonFinite);
            }
            // A horizontal edge never crosses a row through its interior, so
            // it cannot change the winding number and is skipped.
            if a.y() != b.y() {
                result.push(Edge { a, b });
            }
        }
    }
    Ok(result)
}

/// Signed winding for a rightward ray; screen-clockwise contours are positive.
///
/// This is the single-point version of the rule (used to report the winding
/// number at a star or ring center). A closed path crosses any full line a net
/// zero times, so the crossings to the right of the point are the negation of
/// the scanline's running total to its left; subtracting them gives the same
/// sign as `Rasterizer::fill`.
pub fn winding_at(path: &FlattenedPath, point: Point2) -> Result<i32, Error> {
    if !point.is_finite() {
        return Err(Error::NonFinite);
    }
    let mut winding = 0i32;
    for edge in edges(path)? {
        if let Some((x, sign)) = edge.crossing(point.y()) {
            if !x.is_finite() {
                return Err(Error::NonFinite);
            }
            if x > point.x() {
                winding = winding.checked_sub(sign).ok_or(Error::SizeOverflow)?;
            }
        }
    }
    Ok(winding)
}

/// Fills flattened paths onto a fixed-size coverage canvas (0.0 = empty,
/// 1.0 = fully covered).
pub struct Rasterizer {
    width: usize,
    height: usize,
}
impl Rasterizer {
    pub fn new(width: usize, height: usize) -> Result<Self, Error> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions);
        }
        width
            .checked_mul(height)
            .and_then(|len| len.checked_mul(std::mem::size_of::<f64>()))
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or(Error::SizeOverflow)?;
        Ok(Self { width, height })
    }
    /// Part 3-I supersampling: fill the shape on a canvas k times larger in
    /// each direction (shape scaled by k too), then average each k x k block
    /// into one output pixel. The result approximates "brightness proportional
    /// to area of intersection" with k*k sub-positions per pixel.
    pub fn render(&self, path: &FlattenedPath, k: usize) -> Result<Image<f64>, Error> {
        if k == 0 {
            return Err(Error::InvalidFactor);
        }
        // k = 1 is the plain binary fill: one sample per pixel, no averaging.
        if k == 1 {
            return self.fill(path);
        }
        let width = self.width.checked_mul(k).ok_or(Error::SizeOverflow)?;
        let height = self.height.checked_mul(k).ok_or(Error::SizeOverflow)?;
        let large = Self::new(width, height)?;
        let scaled = path.try_scaled_about(Point2::new(0.0, 0.0), k as f64)?;
        large.fill(&scaled)?.mean_downsample(k)
    }
    /// Scanline non-zero fill, one pixel row at a time:
    /// 1. find where every edge crosses the row's center line y + 0.5;
    /// 2. sort those crossings by x;
    /// 3. walk left to right, adding each crossing's +1/-1 to a running total
    ///    (the winding number) and filling every span where it is non-zero.
    ///
    /// Filling alternate spans instead would be the even-odd rule.
    fn fill(&self, path: &FlattenedPath) -> Result<Image<f64>, Error> {
        let edges = edges(path)?;
        let mut image = Image::new(self.width, self.height, 0.0)?;
        let mut crossings = Vec::new();
        crossings
            .try_reserve_exact(edges.len())
            .map_err(|_| Error::Allocation)?;
        // Active-edge list: each edge waits in the bucket of the row where it
        // starts (one row early, so rounding can never make it late) and
        // leaves the active list once the rows pass its upper end. Rows only
        // test the edges that can reach them; `crossing` still decides
        // exactly, so the crossings per row are the same as testing every
        // edge, only cheaper when a path has many edges.
        let mut starts: Vec<Vec<usize>> = Vec::new();
        starts
            .try_reserve_exact(self.height)
            .map_err(|_| Error::Allocation)?;
        starts.resize(self.height, Vec::new());
        let mut ends = Vec::new();
        ends.try_reserve_exact(edges.len())
            .map_err(|_| Error::Allocation)?;
        for (i, edge) in edges.iter().enumerate() {
            let (low, high) = (edge.a.y().min(edge.b.y()), edge.a.y().max(edge.b.y()));
            ends.push(high);
            let first = ((low - 0.5).ceil() - 1.0).max(0.0);
            // Edges entirely above or below the canvas never cross a row.
            if high > 0.5 && first < self.height as f64 {
                starts[first as usize].push(i);
            }
        }
        let mut active: Vec<usize> = Vec::new();
        for (y, starting) in starts.iter().enumerate() {
            crossings.clear();
            active.extend(starting.iter().copied());
            // Sample through pixel centers. Edges outside the canvas still
            // contribute so the running total is correct when it enters.
            let centre = y as f64 + 0.5;
            let mut failed = false;
            active.retain(|&i| {
                if let Some(crossing) = edges[i].crossing(centre) {
                    failed |= !crossing.0.is_finite();
                    crossings.push(crossing);
                }
                centre < ends[i]
            });
            if failed {
                return Err(Error::NonFinite);
            }
            // Equal crossings are grouped by x; their internal order is irrelevant.
            // Unstable sorting uses no auxiliary allocation outside our Result.
            crossings.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            let row = image.row_mut(y).unwrap();
            let mut winding = 0i32;
            let mut previous: Option<f64> = None;
            let mut i = 0;
            while i < crossings.len() {
                let x = crossings[i].0;
                // The span between the previous crossing and this one has the
                // winding accumulated so far; non-zero means inside.
                if winding != 0 {
                    if let Some(left) = previous {
                        // Pixel x is filled when its center x + 0.5 lies in
                        // [left, x): left edge included, right edge excluded.
                        // Clamping clips spans that extend past the canvas.
                        let start =
                            ((left - 0.5_f64).ceil()).clamp(0.0, self.width as f64) as usize;
                        let end = ((x - 0.5).ceil()).clamp(0.0, self.width as f64) as usize;
                        if start < end {
                            row[start..end].fill(1.0);
                        }
                    }
                }
                // Coincident crossings change winding together; no spurious span.
                while i < crossings.len() && crossings[i].0 == x {
                    winding = winding
                        .checked_add(crossings[i].1)
                        .ok_or(Error::SizeOverflow)?;
                    i += 1;
                }
                previous = Some(x);
            }
        }
        Ok(image)
    }
}
