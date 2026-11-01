//! Part 4 appendix, method B carried to a full vectorizer on a scene with
//! several shapes:
//! 1. sub-pixel edge points of the bitmap (unordered);
//! 2. DBSCAN groups them into outlines, so the number of shapes is found
//!    rather than given;
//! 3. each outline gets its own Fourier curve by gradient descent;
//! 4. the smooth Fourier curve is cut at corners (sharp turns) and every
//!    piece is replaced by the simplest primitive that fits it: a line, a
//!    circular arc, or else a cubic Bézier. Neighbouring line/arc fits are
//!    intersected to rebuild the sharp corners the Fourier series rounded.
//!
//! The recovered primitives are written as an SVG path (the vector file) and
//! filled back with the Part 2 rasterizer for comparison.
use crate::{
    experiments::{circle, save_coverage, write_coverage},
    vectorize::{
        backdrop, descend, distance_to, edge_points, initial_circle, misclassified, plot,
        plot_line, rms, BLUE, B_HARMONICS, DENSE, K, RED, SIZE,
    },
};
use multimedia::{
    cluster::dbscan,
    fit::{
        cubic_eval, fit_circle, fit_cubic, fit_line, intersect_line_circle, intersect_lines,
        Circle, Line,
    },
    Color, Complex, Error, Path as ShapePath, Point2, Rasterizer, Subpath,
};
use std::{
    f64::consts::{FRAC_PI_2, PI, TAU},
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

/// DBSCAN radius: edge points along one outline are at most ~1.5 px apart
/// (diagonal edges), so 3 px always links 3+ of them; separate shapes are
/// kept > 10 px apart.
const EPS: f64 = 3.0;
const MIN_POINTS: usize = 3;
/// Arc length (px) looked back and ahead when measuring how sharply the
/// curve turns at a point.
const WINDOW: f64 = 5.0;
const CORNER_DEGREES: f64 = 35.0;
/// A piece that is neither a line nor an arc is cut again at its sharpest
/// interior turn if that turn exceeds this (split-and-fit).
const SPLIT_DEGREES: f64 = 15.0;
/// Near a corner the Fourier curve is rounded; fits ignore this much (px).
const TRIM: f64 = 6.0;
/// A primitive is accepted when no point of its piece is farther than this.
const TOLERANCE: f64 = 0.8;
const GREEN: Color = Color::rgb(0.1, 0.6, 0.25);
const PURPLE: Color = Color::rgb(0.55, 0.2, 0.7);
const BLACK: Color = Color::rgb(0.0, 0.0, 0.0);

fn c(x: f64, y: f64) -> Complex {
    Complex::new(x, y)
}
fn p(z: Complex) -> Point2 {
    Point2::new(z.re, z.im)
}

/// What each shape really is, for measuring the recovery.
struct Truth {
    name: &'static str,
    center: Complex,
    corners: Vec<Complex>,
    circle: Option<Circle>,
}

/// Three separate shapes: straight edges only (star), curve only (circle),
/// and both (a "D": one straight side and a half circle).
fn scene() -> (ShapePath, Vec<Truth>) {
    // Star: five vertices visited every other one, as in Part 2.
    let (star_center, outer) = (c(78.0, 82.0), 56.0);
    let spin = 5.0_f64.to_radians();
    let at = |r: f64, a: f64| star_center + Complex::from_angle(a).scale(r);
    let tip = |i: usize| at(outer, -FRAC_PI_2 + i as f64 * TAU / 5.0 + spin);
    let mut star = Subpath::new(p(tip(0)));
    for i in [2, 4, 1, 3] {
        star.line_to(p(tip(i)));
    }
    star.close();
    let inner = outer * (2.0 * PI / 5.0).cos() / (PI / 5.0).cos();
    let star_corners = (0..5)
        .flat_map(|i| {
            [
                tip(i),
                at(inner, -FRAC_PI_2 + i as f64 * TAU / 5.0 + PI / 5.0 + spin),
            ]
        })
        .collect();
    // Circle: four cubic quarter arcs, as in Part 2-3.
    let (ring_center, ring_radius) = (c(192.0, 70.0), 44.0);
    let ring = circle(p(ring_center), ring_radius);
    // D: straight side x = 110, half circle of radius 42 on the right.
    let (dx, dy, r) = (110.0, 195.0, 42.0);
    let h = 0.5523 * r;
    let mut d = Subpath::new(Point2::new(dx, dy - r));
    d.cubic_to(
        Point2::new(dx + h, dy - r),
        Point2::new(dx + r, dy - h),
        Point2::new(dx + r, dy),
    );
    d.cubic_to(
        Point2::new(dx + r, dy + h),
        Point2::new(dx + h, dy + r),
        Point2::new(dx, dy + r),
    );
    d.line_to(Point2::new(dx, dy - r));
    d.close();
    let truths = vec![
        Truth {
            name: "star",
            center: star_center,
            corners: star_corners,
            circle: None,
        },
        Truth {
            name: "circle",
            center: ring_center,
            corners: Vec::new(),
            circle: Some(Circle {
                center: ring_center,
                radius: ring_radius,
            }),
        },
        Truth {
            name: "d",
            center: c(dx + 0.4 * r, dy),
            corners: vec![c(dx, dy - r), c(dx, dy + r)],
            circle: Some(Circle {
                center: c(dx, dy),
                radius: r,
            }),
        },
    ];
    (ShapePath::new(vec![star, ring, d]), truths)
}

// ------------------------------------------------- corners and pieces ----

/// Index reached by walking `distance` px along the closed curve from `i`.
fn walk(z: &[Complex], i: usize, distance: f64, forward: bool) -> usize {
    let n = z.len();
    let (mut j, mut walked) = (i, 0.0);
    while walked < distance {
        let next = if forward {
            (j + 1) % n
        } else {
            (j + n - 1) % n
        };
        walked += (z[next] - z[j]).abs();
        j = next;
        if j == i {
            break;
        }
    }
    j
}

/// How much the curve turns at each point, in degrees: the angle between
/// the chord arriving from WINDOW px back and the chord leaving to WINDOW
/// px ahead. A smooth bend of radius r turns about 2*WINDOW/r radians.
fn turning(z: &[Complex]) -> Vec<f64> {
    (0..z.len())
        .map(|i| {
            let back = z[i] - z[walk(z, i, WINDOW, false)];
            let ahead = z[walk(z, i, WINDOW, true)] - z[i];
            let rel = ahead * back.conj();
            rel.im.atan2(rel.re).abs().to_degrees()
        })
        .collect()
}

/// Corner indices: turns above CORNER_DEGREES, strongest first, keeping
/// only one per 2*WINDOW px (non-maximum suppression). Sorted by index.
fn corners(z: &[Complex], turn: &[f64]) -> Vec<usize> {
    let mut candidates: Vec<usize> = (0..z.len())
        .filter(|&i| turn[i] >= CORNER_DEGREES)
        .collect();
    candidates.sort_by(|&a, &b| turn[b].total_cmp(&turn[a]).then(a.cmp(&b)));
    let mut kept: Vec<usize> = Vec::new();
    for i in candidates {
        if kept.iter().all(|&j| (z[i] - z[j]).abs() > 2.0 * WINDOW) {
            kept.push(i);
        }
    }
    kept.sort_unstable();
    kept
}

/// Drops the rounded ends of a piece next to sharp corners.
fn trimmed(points: &[Complex], start_sharp: bool, end_sharp: bool) -> Vec<Complex> {
    let mut from_start = vec![0.0];
    for pair in points.windows(2) {
        from_start.push(from_start.last().unwrap() + (pair[1] - pair[0]).abs());
    }
    let total = *from_start.last().unwrap();
    let kept: Vec<Complex> = points
        .iter()
        .zip(&from_start)
        .filter(|(_, s)| (!start_sharp || **s >= TRIM) && (!end_sharp || total - **s >= TRIM))
        .map(|(q, _)| *q)
        .collect();
    if kept.len() >= 6 {
        kept
    } else {
        points.to_vec()
    }
}

/// Which simple geometry supports a piece.
#[derive(Clone, Copy)]
enum Support {
    Line(Line),
    Arc(Circle),
    Free,
}

fn classify(points: &[Complex]) -> Support {
    if let Some(line) = fit_line(points) {
        if points.iter().all(|q| line.distance(*q) <= TOLERANCE) {
            return Support::Line(line);
        }
    }
    if let Some(circle) = fit_circle(points) {
        if circle.radius < 1000.0 && points.iter().all(|q| circle.distance(*q) <= TOLERANCE) {
            return Support::Arc(circle);
        }
    }
    Support::Free
}

/// Sharp corner rebuilt from its neighbours' fits: two lines meet at their
/// intersection, a line and an arc where they cross. Falls back to the
/// curve point if no intersection lies near it.
fn rebuild_corner(before: Support, after: Support, detected: Complex) -> (Complex, &'static str) {
    let near = |q: &Complex| (*q - detected).abs() < 3.0 * TRIM;
    let nearest = |qs: Vec<Complex>| {
        qs.into_iter()
            .filter(near)
            .min_by(|a, b| (*a - detected).abs().total_cmp(&(*b - detected).abs()))
    };
    let found = match (before, after) {
        (Support::Line(a), Support::Line(b)) => intersect_lines(&a, &b)
            .filter(near)
            .map(|q| (q, "line-line")),
        (Support::Line(l), Support::Arc(a)) | (Support::Arc(a), Support::Line(l)) => {
            nearest(intersect_line_circle(&l, &a)).map(|q| (q, "line-arc"))
        }
        _ => None,
    };
    found.unwrap_or((detected, "curve"))
}

// ---------------------------------------------------------- primitives ----

#[derive(Clone, Copy)]
pub(crate) enum Primitive {
    Line(Complex, Complex),
    /// Arc of `circle` from `from` to `to`, turning `sweep` radians.
    Arc {
        circle: Circle,
        from: Complex,
        to: Complex,
        sweep: f64,
    },
    Cubic([Complex; 4]),
}

fn angle(z: Complex) -> f64 {
    z.im.atan2(z.re)
}

/// Signed sweep from `from` to `to` around `center` that passes `mid`.
fn sweep(center: Complex, from: Complex, to: Complex, mid: Complex) -> f64 {
    let a = angle(from - center);
    let forward = (angle(to - center) - a).rem_euclid(TAU);
    if (angle(mid - center) - a).rem_euclid(TAU) < forward {
        forward
    } else {
        forward - TAU
    }
}

/// Signed sweep of an arc fitted to `points`. The middle point decides the
/// direction, except when that arc would be far longer than the points
/// themselves: when the corners were moved by intersecting the neighbouring
/// fits, the middle test can pick the long way and turn a short arc into
/// almost a full circle. Near a half circle both ways are about as long, so
/// length alone cannot decide; it only vetoes the clearly wrong choice.
fn arc_sweep(circle: Circle, from: Complex, to: Complex, points: &[Complex], mid: Complex) -> f64 {
    let by_middle = sweep(circle.center, from, to, mid);
    if points.len() < 2 {
        return by_middle;
    }
    let length: f64 = points.windows(2).map(|p| (p[1] - p[0]).abs()).sum::<f64>()
        + (points[0] - from).abs()
        + (to - points[points.len() - 1]).abs();
    let other = if by_middle > 0.0 {
        by_middle - TAU
    } else {
        by_middle + TAU
    };
    let (chosen, alternative) = (circle.radius * by_middle.abs(), circle.radius * other.abs());
    if chosen > 1.5 * length + 2.0 && (alternative - length).abs() < (chosen - length).abs() {
        other
    } else {
        by_middle
    }
}

/// Largest distance from the piece's points to a cubic.
fn cubic_error(q: &[Complex; 4], points: &[Complex]) -> (f64, usize) {
    let curve: Vec<Complex> = (0..=200).map(|i| cubic_eval(q, i as f64 / 200.0)).collect();
    points
        .iter()
        .enumerate()
        .map(|(i, x)| (distance_to(&curve, *x), i))
        .fold((0.0, 0), |m, e| if e.0 > m.0 { e } else { m })
}

/// Cubic Bézier pieces through the points, split at the worst point until
/// every piece is within TOLERANCE (at most 3 levels deep).
fn cubics(points: &[Complex], from: Complex, to: Complex, depth: usize) -> Vec<Primitive> {
    let Some(q) = fit_cubic(points, from, to) else {
        return vec![Primitive::Line(from, to)];
    };
    let (error, worst) = cubic_error(&q, points);
    if error <= TOLERANCE || depth == 3 || worst < 3 || worst + 3 > points.len() {
        return vec![Primitive::Cubic(q)];
    }
    let split = points[worst];
    let mut result = cubics(&points[..=worst], from, split, depth + 1);
    result.extend(cubics(&points[worst..], split, to, depth + 1));
    result
}

pub(crate) struct Recovered {
    pub(crate) primitives: Vec<Primitive>,
    pub(crate) corners: Vec<(Complex, bool, &'static str)>,
}

/// Turns one closed Fourier curve into primitives.
/// Turns one closed Fourier curve `z` into primitives fitted to the
/// cluster's own edge points `targets`. The curve only orders the points
/// (each takes the parameter of its nearest curve point) and locates the
/// corners; the fits use the measured points, so the curve's ringing near
/// sharp corners does not leak into the primitives.
pub(crate) fn recover(z: &[Complex], targets: &[Complex]) -> Recovered {
    // Curve parameter (index into z) of every edge point.
    let params: Vec<usize> = targets
        .iter()
        .map(|q| {
            (0..z.len())
                .min_by(|&a, &b| {
                    (z[a] - *q)
                        .norm_squared()
                        .total_cmp(&(z[b] - *q).norm_squared())
                })
                .unwrap()
        })
        .collect();
    recover_with_params(z, targets, &params)
}

/// As `recover`, when each edge point's curve parameter is already known
/// (ordered outlines, where point j of n sits at parameter j/n).
pub(crate) fn recover_with_params(
    z: &[Complex],
    targets: &[Complex],
    params: &[usize],
) -> Recovered {
    let turn = turning(z);
    let sharp = corners(z, &turn);
    // No corner and one circle fits every edge point: a full circle.
    if sharp.is_empty() {
        if let Some(whole) = fit_circle(targets) {
            if targets.iter().all(|q| whole.distance(*q) <= TOLERANCE) {
                let start = whole.center
                    + (z[0] - whole.center).scale(whole.radius / (z[0] - whole.center).abs());
                let area: f64 = (0..z.len())
                    .map(|i| {
                        let (a, b) = (z[i], z[(i + 1) % z.len()]);
                        a.re * b.im - a.im * b.re
                    })
                    .sum();
                return Recovered {
                    primitives: vec![Primitive::Arc {
                        circle: whole,
                        from: start,
                        to: start,
                        sweep: TAU.copysign(area),
                    }],
                    corners: Vec::new(),
                };
            }
        }
    }
    // Breakpoints: the corners, or four evenly spaced smooth cuts if none.
    let mut breaks: Vec<(usize, bool)> = if sharp.is_empty() {
        (0..4).map(|i| (i * z.len() / 4, false)).collect()
    } else {
        sharp.iter().map(|&i| (i, true)).collect()
    };
    let n = z.len();
    // Edge points between breakpoint j and j+1, in curve order.
    let split_into_pieces = |breaks: &[(usize, bool)]| -> Vec<Vec<Complex>> {
        let m = breaks.len();
        (0..m)
            .map(|j| {
                let (a, b) = (breaks[j].0, breaks[(j + 1) % m].0);
                let span = if m == 1 { n } else { (b + n - a) % n };
                let mut members: Vec<(usize, Complex)> = params
                    .iter()
                    .zip(targets)
                    .map(|(&t, q)| ((t + n - a) % n, *q))
                    .filter(|(offset, _)| *offset < span)
                    .collect();
                members.sort_by_key(|(offset, _)| *offset);
                members.into_iter().map(|(_, q)| q).collect()
            })
            .collect()
    };
    let classify_all = |breaks: &[(usize, bool)], pieces: &[Vec<Complex>]| -> Vec<Support> {
        let m = breaks.len();
        (0..m)
            .map(|j| classify(&trimmed(&pieces[j], breaks[j].1, breaks[(j + 1) % m].1)))
            .collect()
    };
    let mut pieces = split_into_pieces(&breaks);
    let mut supports = classify_all(&breaks, &pieces);
    // Split-and-fit: a shallow corner (below CORNER_DEGREES) leaves a piece
    // that is neither a line nor an arc; cut it at its sharpest interior
    // turn and classify again. Pieces that already fit are never cut.
    let mut length = vec![0.0];
    for i in 1..=n {
        length.push(length[i - 1] + (z[i % n] - z[i - 1]).abs());
    }
    let along = |from: usize, to: usize| {
        let d = length[to] - length[from];
        if d >= 0.0 {
            d
        } else {
            d + length[n]
        }
    };
    for _ in 0..32 {
        let m = breaks.len();
        let cut = (0..m)
            .filter(|&j| matches!(supports[j], Support::Free))
            .filter_map(|j| {
                let (a, b) = (breaks[j].0, breaks[(j + 1) % m].0);
                let span = if m == 1 { n } else { (b + n - a) % n };
                (1..span)
                    .map(|k| (a + k) % n)
                    .filter(|&i| along(a, i) >= 2.0 * WINDOW && along(i, b) >= 2.0 * WINDOW)
                    .max_by(|&x, &y| turn[x].total_cmp(&turn[y]))
                    .filter(|&i| turn[i] >= SPLIT_DEGREES)
            })
            .next();
        let Some(i) = cut else { break };
        breaks.push((i, true));
        breaks.sort_unstable();
        pieces = split_into_pieces(&breaks);
        supports = classify_all(&breaks, &pieces);
    }
    let m = breaks.len();
    // Corner j sits between piece j-1 and piece j.
    let corner_points: Vec<(Complex, bool, &'static str)> = (0..m)
        .map(|j| {
            let detected = z[breaks[j].0];
            if breaks[j].1 {
                let (q, how) = rebuild_corner(supports[(j + m - 1) % m], supports[j], detected);
                (q, true, how)
            } else {
                (detected, false, "smooth")
            }
        })
        .collect();
    let mut primitives = Vec::new();
    for j in 0..m {
        let from = corner_points[j].0;
        let to = corner_points[(j + 1) % m].0;
        match supports[j] {
            Support::Line(_) => primitives.push(Primitive::Line(from, to)),
            Support::Arc(circle) => {
                // A piece with no edge points (corners very close together)
                // has no middle to read; fall back to the curve.
                let mid = pieces[j].get(pieces[j].len() / 2).copied().unwrap_or(
                    z[(breaks[j].0 + ((breaks[(j + 1) % m].0 + n - breaks[j].0) % n) / 2) % n],
                );
                primitives.push(Primitive::Arc {
                    circle,
                    from,
                    to,
                    sweep: arc_sweep(circle, from, to, &pieces[j], mid),
                })
            }
            Support::Free => primitives.extend(cubics(&pieces[j], from, to, 0)),
        }
    }
    Recovered {
        primitives,
        corners: corner_points,
    }
}

/// Appends a primitive to a sub-path. Arcs become cubic Béziers of at most
/// 90° each, with handles 4/3 * tan(phi/4) * r along the tangent (for a
/// quarter circle this is the 0.5523 * r of Part 2-3).
fn append(sub: &mut Subpath, primitive: &Primitive) {
    match *primitive {
        Primitive::Line(_, b) => {
            sub.line_to(p(b));
        }
        Primitive::Cubic(q) => {
            sub.cubic_to(p(q[1]), p(q[2]), p(q[3]));
        }
        Primitive::Arc {
            circle,
            from,
            to,
            sweep,
        } => {
            let n = (sweep.abs() / FRAC_PI_2).ceil().max(1.0) as usize;
            let phi = sweep / n as f64;
            let handle = 4.0 / 3.0 * (phi / 4.0).tan() * circle.radius;
            let start = angle(from - circle.center);
            let on = |t: f64| circle.center + Complex::from_angle(t).scale(circle.radius);
            // Unit tangent in the direction of increasing angle.
            let tangent = |t: f64| Complex::from_angle(t + FRAC_PI_2);
            for i in 0..n {
                let (t0, t1) = (start + i as f64 * phi, start + (i + 1) as f64 * phi);
                let end = if i + 1 == n { to } else { on(t1) };
                sub.cubic_to(
                    p(on(t0) + tangent(t0).scale(handle)),
                    p(on(t1) - tangent(t1).scale(handle)),
                    p(end),
                );
            }
        }
    }
}

fn start_of(primitive: &Primitive) -> Complex {
    match *primitive {
        Primitive::Line(a, _) => a,
        Primitive::Cubic(q) => q[0],
        Primitive::Arc { from, .. } => from,
    }
}

pub(crate) fn to_subpath(primitives: &[Primitive]) -> Subpath {
    let mut sub = Subpath::new(p(start_of(&primitives[0])));
    for primitive in primitives {
        append(&mut sub, primitive);
    }
    sub.close();
    sub
}

/// Dense points along a primitive, about 4 per px.
pub(crate) fn sample(primitive: &Primitive) -> Vec<Complex> {
    match *primitive {
        Primitive::Line(a, b) => {
            let n = ((b - a).abs() * 4.0).ceil().max(1.0) as usize;
            (0..=n)
                .map(|i| a + (b - a).scale(i as f64 / n as f64))
                .collect()
        }
        Primitive::Cubic(q) => (0..=400)
            .map(|i| cubic_eval(&q, i as f64 / 400.0))
            .collect(),
        Primitive::Arc {
            circle,
            from,
            sweep,
            ..
        } => {
            let n = (sweep.abs() * circle.radius * 4.0).ceil().max(8.0) as usize;
            let start = angle(from - circle.center);
            (0..=n)
                .map(|i| {
                    circle.center
                        + Complex::from_angle(start + sweep * i as f64 / n as f64)
                            .scale(circle.radius)
                })
                .collect()
        }
    }
}

pub(crate) fn kind(primitive: &Primitive) -> &'static str {
    match primitive {
        Primitive::Line(..) => "line",
        Primitive::Arc { .. } => "arc",
        Primitive::Cubic(_) => "cubic",
    }
}

/// CSV columns x0,y0,x1,y1,cx,cy,radius,sweep,q1x,q1y,q2x,q2y for one primitive.
pub(crate) fn primitive_row(primitive: &Primitive) -> String {
    match *primitive {
        Primitive::Line(a, b) => {
            format!("{:.6},{:.6},{:.6},{:.6},,,,,,,,", a.re, a.im, b.re, b.im)
        }
        Primitive::Arc {
            circle,
            from,
            to,
            sweep,
        } => format!(
            "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.9},,,,",
            from.re,
            from.im,
            to.re,
            to.im,
            circle.center.re,
            circle.center.im,
            circle.radius,
            sweep
        ),
        Primitive::Cubic(q) => format!(
            "{:.6},{:.6},{:.6},{:.6},,,,,{:.6},{:.6},{:.6},{:.6}",
            q[0].re, q[0].im, q[3].re, q[3].im, q[1].re, q[1].im, q[2].re, q[2].im
        ),
    }
}

pub(crate) fn svg_path(sub: &Subpath) -> String {
    use multimedia::Segment;
    let mut d = format!("M{:.3} {:.3}", sub.start().x(), sub.start().y());
    for segment in sub.segments() {
        match segment {
            Segment::LineTo(e) => d += &format!(" L{:.3} {:.3}", e.x(), e.y()),
            Segment::CubicTo {
                control1,
                control2,
                end,
            } => {
                d += &format!(
                    " C{:.3} {:.3} {:.3} {:.3} {:.3} {:.3}",
                    control1.x(),
                    control1.y(),
                    control2.x(),
                    control2.y(),
                    end.x(),
                    end.y()
                )
            }
        }
    }
    d + " Z"
}

// ---------------------------------------------------------------- run ----

pub fn run(output: &Path) -> Result<(), Error> {
    let raster = Rasterizer::new(SIZE, SIZE)?;
    let (shapes, truths) = scene();
    let original = raster.render(&shapes.flatten(7)?, K)?;
    save_coverage(&original, &output.join("recover_scene.bmp"))?;
    write_coverage(&output.join("recover_scene_coverage.csv"), &original)?;

    // 1-2. Edge points, then DBSCAN decides how many outlines there are.
    let edges = edge_points(&original);
    let labels = dbscan(&edges, EPS, MIN_POINTS)?;
    let clusters = labels.iter().flatten().max().map_or(0, |m| m + 1);
    let mut points_file = BufWriter::new(File::create(output.join("recover_clusters.csv"))?);
    writeln!(points_file, "x,y,cluster")?;
    let palette = [RED, BLUE, GREEN, PURPLE, Color::rgb(0.9, 0.55, 0.0)];
    let mut cluster_image = backdrop(&original)?;
    for (q, label) in edges.iter().zip(&labels) {
        let text = label.map_or(String::from("noise"), |l| l.to_string());
        writeln!(points_file, "{:.12},{:.12},{text}", q.re, q.im)?;
        plot(
            &mut cluster_image,
            *q,
            label.map_or(BLACK, |l| palette[l % palette.len()]),
        );
    }
    points_file.flush()?;
    cluster_image.save_bmp(output.join("recover_clusters.bmp"))?;

    let mut fourier_image = backdrop(&original)?;
    let mut primitive_image = backdrop(&original)?;
    let mut losses = BufWriter::new(File::create(output.join("recover_loss.csv"))?);
    writeln!(losses, "cluster,harmonics,iteration,loss")?;
    let mut prim_file = BufWriter::new(File::create(output.join("recover_primitives.csv"))?);
    writeln!(
        prim_file,
        "cluster,index,kind,x0,y0,x1,y1,cx,cy,radius,sweep,q1x,q1y,q2x,q2y"
    )?;
    let mut corner_file = BufWriter::new(File::create(output.join("recover_corners.csv"))?);
    writeln!(corner_file, "cluster,x,y,sharp,method")?;
    let mut coefficient_file =
        BufWriter::new(File::create(output.join("recover_coefficients.csv"))?);
    writeln!(coefficient_file, "cluster,k,re,im")?;
    let mut subpaths = Vec::new();
    let mut svg =
        format!("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {SIZE} {SIZE}\">\n");
    let mut shape_json = Vec::new();
    for id in 0..clusters {
        let targets: Vec<Complex> = edges
            .iter()
            .zip(&labels)
            .filter(|(_, l)| **l == Some(id))
            .map(|(q, _)| *q)
            .collect();
        // 3. Fourier outline for this cluster, coarse to fine.
        let mut curve = initial_circle(&targets);
        let mut loss = 0.0;
        for harmonics in B_HARMONICS {
            curve = curve.extended(harmonics);
            loss = descend(&mut curve, &targets, |iteration, value| {
                writeln!(losses, "{id},{harmonics},{iteration},{value:.12}")
            })?;
        }
        for (i, coefficient) in curve.coefficients.iter().enumerate() {
            writeln!(
                coefficient_file,
                "{id},{},{:.12},{:.12}",
                i as i64 - curve.harmonics as i64,
                coefficient.re,
                coefficient.im
            )?;
        }
        let dense = curve.sample(DENSE);
        for pair in dense.windows(2) {
            plot_line(&mut fourier_image, pair[0], pair[1], RED);
        }
        // 4. Corners, pieces, primitives.
        let recovered = recover(&dense, &targets);
        let mut outline = Vec::new();
        for (index, primitive) in recovered.primitives.iter().enumerate() {
            let points = sample(primitive);
            let color = match primitive {
                Primitive::Line(..) => BLUE,
                Primitive::Arc { .. } => GREEN,
                Primitive::Cubic(_) => PURPLE,
            };
            for pair in points.windows(2) {
                plot_line(&mut primitive_image, pair[0], pair[1], color);
            }
            outline.extend(points);
            let row = primitive_row(primitive);
            writeln!(prim_file, "{id},{index},{},{row}", kind(primitive))?;
        }
        for &(q, sharp, how) in &recovered.corners {
            writeln!(corner_file, "{id},{:.9},{:.9},{sharp},{how}", q.re, q.im)?;
            if sharp {
                plot_line(
                    &mut primitive_image,
                    q - c(3.0, 3.0),
                    q + c(3.0, 3.0),
                    BLACK,
                );
                plot_line(
                    &mut primitive_image,
                    q - c(3.0, -3.0),
                    q + c(3.0, -3.0),
                    BLACK,
                );
            }
        }
        let sub = to_subpath(&recovered.primitives);
        svg += &format!("  <path d=\"{}\" fill=\"black\"/>\n", svg_path(&sub));
        subpaths.push(sub);

        // Measure: distance from this cluster's edge points to the Fourier
        // curve and to the recovered outline, and errors against the truth.
        let fourier_edge: Vec<f64> = targets.iter().map(|q| distance_to(&dense, *q)).collect();
        let vector_edge: Vec<f64> = targets.iter().map(|q| distance_to(&outline, *q)).collect();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let max = |v: &[f64]| v.iter().copied().fold(0.0, f64::max);
        let centroid = targets
            .iter()
            .fold(Complex::ZERO, |s, q| s + *q)
            .scale(1.0 / targets.len() as f64);
        let truth = truths
            .iter()
            .min_by(|a, b| {
                (a.center - centroid)
                    .abs()
                    .total_cmp(&(b.center - centroid).abs())
            })
            .unwrap();
        let sharp_points: Vec<Complex> = recovered
            .corners
            .iter()
            .filter(|x| x.1)
            .map(|x| x.0)
            .collect();
        let corner_error = truth
            .corners
            .iter()
            .map(|t| {
                sharp_points
                    .iter()
                    .map(|q| (*q - *t).abs())
                    .fold(f64::INFINITY, f64::min)
            })
            .fold(0.0, f64::max);
        let fourier_corner = truth
            .corners
            .iter()
            .map(|t| distance_to(&dense, *t))
            .fold(0.0, f64::max);
        let arc = recovered
            .primitives
            .iter()
            .filter_map(|x| match x {
                Primitive::Arc { circle, .. } => Some(*circle),
                _ => None,
            })
            .min_by(|a, b| {
                let r = truth.circle.map_or(0.0, |t| t.radius);
                (a.radius - r).abs().total_cmp(&(b.radius - r).abs())
            });
        let count = |k: &str| recovered.primitives.iter().filter(|x| kind(x) == k).count();
        let (center_error, radius_error) = match (truth.circle, arc) {
            (Some(t), Some(a)) => ((a.center - t.center).abs(), (a.radius - t.radius).abs()),
            _ => (f64::NAN, f64::NAN),
        };
        let json_number = |v: f64| {
            if v.is_finite() {
                format!("{v:.9}")
            } else {
                String::from("null")
            }
        };
        shape_json.push(format!(
            "{{\"cluster\":{id},\"truth\":\"{}\",\"edge_points\":{},\"fourier_loss\":{loss:.9},\"lines\":{},\"arcs\":{},\"cubics\":{},\"sharp_corners\":{},\"true_corners\":{},\"corner_error_px\":{},\"fourier_corner_error_px\":{},\"arc_center_error_px\":{},\"arc_radius_error_px\":{},\"fourier_mean_edge_px\":{:.9},\"fourier_max_edge_px\":{:.9},\"vector_mean_edge_px\":{:.9},\"vector_max_edge_px\":{:.9}}}",
            truth.name,
            targets.len(),
            count("line"),
            count("arc"),
            count("cubic"),
            sharp_points.len(),
            truth.corners.len(),
            json_number(if truth.corners.is_empty() { f64::NAN } else { corner_error }),
            json_number(if truth.corners.is_empty() { f64::NAN } else { fourier_corner }),
            json_number(center_error),
            json_number(radius_error),
            mean(&fourier_edge),
            max(&fourier_edge),
            mean(&vector_edge),
            max(&vector_edge)
        ));
    }
    for file in [
        &mut losses,
        &mut prim_file,
        &mut corner_file,
        &mut coefficient_file,
    ] {
        file.flush()?;
    }
    svg += "</svg>\n";
    std::fs::write(output.join("recover_vector.svg"), svg)?;
    fourier_image.save_bmp(output.join("recover_fourier.bmp"))?;
    primitive_image.save_bmp(output.join("recover_primitives.bmp"))?;
    // Fill the recovered vector shapes back with our own rasterizer.
    let rebuilt = raster.render(&ShapePath::new(subpaths).flatten(7)?, K)?;
    save_coverage(&rebuilt, &output.join("recover_rebuilt.bmp"))?;
    write_coverage(&output.join("recover_rebuilt_coverage.csv"), &rebuilt)?;
    let noise = labels.iter().filter(|l| l.is_none()).count();
    std::fs::write(
        output.join("recover_summary.json"),
        format!(
            "{{\"canvas\":{SIZE},\"k\":{K},\"eps\":{EPS},\"min_points\":{MIN_POINTS},\"edge_points\":{},\"clusters\":{clusters},\"noise\":{noise},\"window_px\":{WINDOW},\"corner_degrees\":{CORNER_DEGREES},\"trim_px\":{TRIM},\"tolerance_px\":{TOLERANCE},\"harmonics\":{:?},\"rebuilt_rms\":{:.12},\"rebuilt_misclassified\":{},\"shapes\":[{}]}}\n",
            edges.len(),
            B_HARMONICS,
            rms(&rebuilt, &original),
            misclassified(&rebuilt, &original),
            shape_json.join(",")
        ),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_4;

    #[test]
    fn arc_direction_follows_the_middle_unless_it_is_far_too_long() {
        let circle = Circle {
            center: Complex::ZERO,
            radius: 10.0,
        };
        let on = |t: f64| Complex::from_angle(t).scale(10.0);
        let quarter: Vec<Complex> = (0..=20).map(|i| on(FRAC_PI_2 * i as f64 / 20.0)).collect();
        let (from, to) = (on(0.0), on(FRAC_PI_2));
        // A sensible middle point gives the quarter turn.
        assert!((arc_sweep(circle, from, to, &quarter, on(FRAC_PI_4)) - FRAC_PI_2).abs() < 1e-9);
        // A misleading middle point would go the long way (3/4 of a turn,
        // 47 px for 16 px of points); the length veto turns it back.
        assert!((sweep(circle.center, from, to, on(PI)) + 3.0 * FRAC_PI_2).abs() < 1e-9);
        assert!((arc_sweep(circle, from, to, &quarter, on(PI)) - FRAC_PI_2).abs() < 1e-9);
        // Near a half circle both ways are equally long: the middle decides.
        let half: Vec<Complex> = (0..=20).map(|i| on(PI * i as f64 / 20.0)).collect();
        assert!((arc_sweep(circle, on(0.0), on(PI), &half, on(-FRAC_PI_2)) + PI).abs() < 1e-9);
    }
}
