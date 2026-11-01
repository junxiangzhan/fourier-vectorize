//! Least-squares fits of simple primitives to 2D points (as complex
//! numbers x + iy): straight lines, circles, and cubic Bézier pieces, plus
//! the intersections needed to rebuild sharp corners between them.
use crate::Complex;

/// An infinite line through `point` with unit `direction`.
#[derive(Clone, Copy, Debug)]
pub struct Line {
    pub point: Complex,
    pub direction: Complex,
}
impl Line {
    pub fn distance(&self, p: Complex) -> f64 {
        let d = p - self.point;
        // |cross(direction, d)|: perpendicular distance.
        (self.direction.re * d.im - self.direction.im * d.re).abs()
    }
    /// Signed position of p's foot along the line.
    pub fn project(&self, p: Complex) -> f64 {
        let d = p - self.point;
        d.re * self.direction.re + d.im * self.direction.im
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Circle {
    pub center: Complex,
    pub radius: f64,
}
impl Circle {
    pub fn distance(&self, p: Complex) -> f64 {
        ((p - self.center).abs() - self.radius).abs()
    }
}

fn centroid(points: &[Complex]) -> Complex {
    points
        .iter()
        .fold(Complex::ZERO, |s, p| s + *p)
        .scale(1.0 / points.len() as f64)
}

/// Total least squares: the line through the centroid along the direction
/// of greatest spread (principal axis of the 2x2 covariance matrix), which
/// minimizes perpendicular rather than vertical distances.
pub fn fit_line(points: &[Complex]) -> Option<Line> {
    if points.len() < 2 {
        return None;
    }
    let c = centroid(points);
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for p in points {
        let d = *p - c;
        sxx += d.re * d.re;
        sxy += d.re * d.im;
        syy += d.im * d.im;
    }
    if sxx + syy == 0.0 {
        return None;
    }
    // Angle of the principal axis of [[sxx, sxy], [sxy, syy]].
    let angle = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    Some(Line {
        point: c,
        direction: Complex::from_angle(angle),
    })
}

/// Solves a small dense linear system by Gaussian elimination with partial
/// pivoting; None if it is (numerically) singular.
fn solve<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let pivot = (col..N).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        // Subtract multiples of the pivot row to zero the column below it.
        let pivot_row = a[col];
        for row in col + 1..N {
            let f = a[row][col] / pivot_row[col];
            for (x, p) in a[row][col..].iter_mut().zip(&pivot_row[col..]) {
                *x -= f * p;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; N];
    for row in (0..N).rev() {
        let rest: f64 = (row + 1..N).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - rest) / a[row][row];
    }
    Some(x)
}

/// Algebraic (Kåsa) circle fit: minimizes sum (x² + y² + Dx + Ey + F)²,
/// a linear problem; then center = (-D/2, -E/2), r² = |center|² - F.
/// Points are shifted to their centroid first for numerical stability.
pub fn fit_circle(points: &[Complex]) -> Option<Circle> {
    if points.len() < 3 {
        return None;
    }
    let c = centroid(points);
    let mut a = [[0.0; 3]; 3];
    let mut b = [0.0; 3];
    for p in points {
        let d = *p - c;
        let row = [d.re, d.im, 1.0];
        let rhs = -(d.re * d.re + d.im * d.im);
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] += row[i] * row[j];
            }
            b[i] += row[i] * rhs;
        }
    }
    let [dd, ee, ff] = solve(a, b)?;
    let center = Complex::new(-dd / 2.0, -ee / 2.0);
    let r2 = center.norm_squared() - ff;
    (r2 > 0.0 && r2.is_finite()).then(|| Circle {
        center: center + c,
        radius: r2.sqrt(),
    })
}

/// Point on the cubic Bézier with controls q at parameter t.
pub fn cubic_eval(q: &[Complex; 4], t: f64) -> Complex {
    let s = 1.0 - t;
    q[0].scale(s * s * s)
        + q[1].scale(3.0 * s * s * t)
        + q[2].scale(3.0 * s * t * t)
        + q[3].scale(t * t * t)
}

/// First and second derivatives of the cubic at t.
fn cubic_derivatives(q: &[Complex; 4], t: f64) -> (Complex, Complex) {
    let s = 1.0 - t;
    let d1 = (q[1] - q[0]).scale(3.0 * s * s)
        + (q[2] - q[1]).scale(6.0 * s * t)
        + (q[3] - q[2]).scale(3.0 * t * t);
    let d2 = (q[2] - q[1].scale(2.0) + q[0]).scale(6.0 * s)
        + (q[3] - q[2].scale(2.0) + q[1]).scale(6.0 * t);
    (d1, d2)
}

/// Rounds of parameter correction in `fit_cubic`.
const REPARAMETERIZE: usize = 4;

/// Cubic Bézier from `start` to `end` (fixed) whose two inner control
/// points minimize the squared distance to `points` (Schneider's method).
/// Parameters t_i start as chord lengths; after each least-squares solve,
/// one Newton step moves every t_i toward the closest point of the new
/// curve, and the controls are solved again.
pub fn fit_cubic(points: &[Complex], start: Complex, end: Complex) -> Option<[Complex; 4]> {
    if points.len() < 2 {
        return None;
    }
    let mut lengths = vec![0.0];
    for pair in points.windows(2) {
        lengths.push(lengths.last().unwrap() + (pair[1] - pair[0]).abs());
    }
    let total = *lengths.last().unwrap();
    if total == 0.0 {
        return None;
    }
    let mut ts: Vec<f64> = lengths.iter().map(|l| l / total).collect();
    let mut q = solve_controls(points, &ts, start, end)?;
    for _ in 0..REPARAMETERIZE {
        for (t, p) in ts.iter_mut().zip(points) {
            // Newton step on f(t) = (B(t) - p) . B'(t) = 0.
            let (d1, d2) = cubic_derivatives(&q, *t);
            let e = cubic_eval(&q, *t) - *p;
            let f = e.re * d1.re + e.im * d1.im;
            let df = d1.norm_squared() + e.re * d2.re + e.im * d2.im;
            if df.abs() > 1e-12 {
                *t = (*t - f / df).clamp(0.0, 1.0);
            }
        }
        q = solve_controls(points, &ts, start, end)?;
    }
    Some(q)
}

/// Least-squares inner controls for fixed parameters: the normal equations
/// are one 2 x 2 system per axis.
fn solve_controls(
    points: &[Complex],
    ts: &[f64],
    start: Complex,
    end: Complex,
) -> Option<[Complex; 4]> {
    let mut a = [[0.0; 2]; 2];
    let (mut bx, mut by) = ([0.0; 2], [0.0; 2]);
    for (p, &t) in points.iter().zip(ts) {
        let s = 1.0 - t;
        let basis = [3.0 * s * s * t, 3.0 * s * t * t];
        // What the two free controls must supply at this t.
        let rest = *p - start.scale(s * s * s) - end.scale(t * t * t);
        for i in 0..2 {
            for j in 0..2 {
                a[i][j] += basis[i] * basis[j];
            }
            bx[i] += basis[i] * rest.re;
            by[i] += basis[i] * rest.im;
        }
    }
    let x = solve(a, bx)?;
    let y = solve(a, by)?;
    Some([
        start,
        Complex::new(x[0], y[0]),
        Complex::new(x[1], y[1]),
        end,
    ])
}

/// Where two lines cross; None if (nearly) parallel.
pub fn intersect_lines(a: &Line, b: &Line) -> Option<Complex> {
    let cross = a.direction.re * b.direction.im - a.direction.im * b.direction.re;
    if cross.abs() < 1e-9 {
        return None;
    }
    let d = b.point - a.point;
    let t = (d.re * b.direction.im - d.im * b.direction.re) / cross;
    Some(a.point + a.direction.scale(t))
}

/// Where a line meets a circle (zero, one or two points).
pub fn intersect_line_circle(line: &Line, circle: &Circle) -> Vec<Complex> {
    // Foot of the perpendicular from the center, then step along the line.
    let foot = line.point + line.direction.scale(line.project(circle.center));
    let h2 = circle.radius * circle.radius - (circle.center - foot).norm_squared();
    if h2 < 0.0 {
        return Vec::new();
    }
    let h = h2.sqrt();
    vec![
        foot - line.direction.scale(h),
        foot + line.direction.scale(h),
    ]
}
