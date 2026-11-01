//! Part 4, high fidelity: the paintings drawn straight from Fourier series,
//! with many more levels and harmonics, and with gradient descent.
//!
//! Each painting is shrunk less (The Dream at full size, Adam by 2) and cut
//! into Lab plates of 32 + 16 + 16 levels. Every outline gets about n/3
//! harmonics (n = its point count, up to MAX_HARMONICS), and every result
//! is rendered directly from the series (no lines, arcs or Béziers).
//!
//! Two ways to get the outlines:
//! - crack: as in the showcase, raw pixel levels (with small regions merged)
//!   traced along pixel cracks; the noise of single pixels and the pixel
//!   staircase both end up in the outline;
//! - smooth: the picture first goes through the bilateral filter, and each
//!   level boundary is the sub-pixel iso-line (marching squares) of the
//!   smoothed channel at the midpoint between two levels.
//!
//! Coefficients for the smooth outlines, three ways:
//! 1. dft: the ordered outline, resampled evenly by length and transformed;
//! 2. refined: the DFT result improved by chamfer gradient descent;
//! 3. descent: descent alone from a circle, coarse to fine, as for the test
//!    star; only the outline's points are used, in no particular order.
//!
//! The chamfer loss only asks that curve and edge points be close, not in
//! which order the curve visits them, so descent alone can run along the
//! edges and cut straight across between distant parts, flipping whole
//! areas inside out. A region term (`Region`), comparing the area the curve
//! encloses with the outline's (a region-based active contour, Chan-Vese),
//! was tried against this; `region_sweep` records how it fares on the
//! largest a* and b* outlines (hifi_region_sweep.csv).
//!
//! The crack outlines get the DFT and the refined fit, so the effect of the
//! outline source on refining is measured too.
//!
//! Curves are sampled evenly in t, so both the samples and the gradient are
//! a single FFT: O(S log S) per step instead of O(S * harmonics).
use crate::{
    layers::{delta_e, lab, signed_area},
    layers::hex,
    paintings::{
        bilateral, despeckle, kmeans_1d, lab_to_rgb, levels_of, loops, picture_name, shrink, shrink_factor, ssim,
        svg_lab_filter, MIN_AREA,
    },
};
use multimedia::{bmp, fft, Color, Complex, Error, FlattenedPath, Image, Point2, Polyline, Rasterizer};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

/// Without a given factor, a picture is shrunk by the smallest factor that
/// brings its longer side to at most this many pixels (The Dream by 1,
/// The Creation of Adam by 2).
const LONGEST: usize = 2000;
/// Levels for L*, a*, b* in the comparison.
const LEVELS: [usize; 3] = [32, 16, 16];
const MAX_HARMONICS: usize = 1024;
/// Steps of descent when refining the DFT result, and per coarse-to-fine
/// stage when starting from a circle.
const REFINE_STEPS: usize = 200;
const STAGE_STEPS: usize = 200;
const RATE: f64 = 0.2;
/// The region term's blur radius (px): it fades to zero at the edge, where
/// the chamfer loss is the better guide, and acts in full only on curve
/// pieces at least this far on the wrong side.
const REGION_BLUR: usize = 3;
/// Region weights the sweep tries; 0 is the chamfer loss alone.
const REGION_WEIGHTS: [f64; 3] = [0.0, 1.0, 4.0];
/// A sweep fit counts as diverged when its chamfer loss exceeds this (px²,
/// an RMS distance of 10 px) or is not finite.
const DIVERGED_LOSS: f64 = 100.0;
/// Supersampling of the refill (the canvases are already large).
const FILL_K: usize = 2;
/// A pixel differs visibly when ΔE exceeds this.
const VISIBLE_DELTA_E: f64 = 10.0;
/// Descent from a circle counts as stuck when its final chamfer loss is
/// worse than this many times the DFT's, plus a small allowance (px²).
const STUCK_FACTOR: f64 = 4.0;
const STUCK_ALLOWANCE: f64 = 0.25;
/// The bitmaps written, in this order.
const IMAGES: [&str; 7] = ["quant_raw", "crack_dft", "crack_refined", "quant", "dft", "refined", "descent"];
/// Coefficient variants of a smooth outline (a crack outline has the first
/// two).
const VARIANTS: [&str; 3] = ["dft", "refined", "descent"];

/// Static 2-d tree for exact nearest-point queries. The bucket grid of the
/// test star visits every cell out to the answer, so a query 300 px from
/// the curve (a starting circle against a long outline) costs tens of
/// thousands of cells; here it costs about log n nodes at any distance.
struct KdTree<'a> {
    points: &'a [Complex],
    /// Point indices; each slice's middle element splits it, by x at even
    /// depths and by y at odd ones.
    order: Vec<usize>,
}
impl<'a> KdTree<'a> {
    fn new(points: &'a [Complex]) -> Self {
        fn build(order: &mut [usize], points: &[Complex], depth: usize) {
            if order.len() <= 1 {
                return;
            }
            let mid = order.len() / 2;
            let key = |i: &usize| if depth.is_multiple_of(2) { points[*i].re } else { points[*i].im };
            order.select_nth_unstable_by(mid, |a, b| key(a).total_cmp(&key(b)));
            let (left, right) = order.split_at_mut(mid);
            build(left, points, depth + 1);
            build(&mut right[1..], points, depth + 1);
        }
        let mut order: Vec<usize> = (0..points.len()).collect();
        build(&mut order, points, 0);
        Self { points, order }
    }
    /// Squared distance and index of the nearest point; ties go to the
    /// lowest index, as a brute-force scan with `<` would choose.
    fn nearest(&self, q: Complex) -> (f64, usize) {
        let mut best = (f64::INFINITY, usize::MAX);
        self.search(&self.order, q, 0, &mut best);
        best
    }
    fn search(&self, order: &[usize], q: Complex, depth: usize, best: &mut (f64, usize)) {
        if order.is_empty() {
            return;
        }
        let mid = order.len() / 2;
        let i = order[mid];
        let p = self.points[i];
        let d = (p - q).norm_squared();
        if d < best.0 || (d == best.0 && i < best.1) {
            *best = (d, i);
        }
        let diff = if depth.is_multiple_of(2) { q.re - p.re } else { q.im - p.im };
        // Left of the middle are keys <= its key, right are keys >= it.
        let (near, far) = if diff < 0.0 {
            (&order[..mid], &order[mid + 1..])
        } else {
            (&order[mid + 1..], &order[..mid])
        };
        self.search(near, q, depth + 1, best);
        // Every point on the far side is at least |diff| away.
        if diff * diff <= best.0 {
            self.search(far, q, depth + 1, best);
        }
    }
}

/// Iso-lines of `field` (w x h values at pixel centres) at `level` by
/// marching squares: inside means value >= level, and each crossing is
/// placed by linear interpolation along the cell edge, so outlines are
/// sub-pixel instead of following the pixel staircase. Outside the image
/// counts as outside, its crossings on the image border. Loops keep the
/// inside on their right, like the crack loops: outer boundaries come out
/// screen-clockwise and holes counter-clockwise. Where a cell has two
/// inside corners on a diagonal, its centre (mean of the corners) decides
/// whether they connect.
fn contours(field: &[f64], w: usize, h: usize, level: f64) -> Vec<Vec<Complex>> {
    // Corner (i, j), i in 0..=w+1 and j in 0..=h+1, is pixel (i-1, j-1);
    // the outer ring is padding.
    let value = |i: usize, j: usize| (i >= 1 && j >= 1 && i <= w && j <= h).then(|| field[(j - 1) * w + i - 1]);
    let inside = |(i, j): (usize, usize)| value(i, j).is_some_and(|v| v >= level);
    // Edge ids: horizontal (i,j)-(i+1,j) is 2 * cell, vertical (i,j)-(i,j+1) is 2 * cell + 1.
    let stride = w + 2;
    let horizontal = |i: usize, j: usize| 2 * (j * stride + i);
    let vertical = |i: usize, j: usize| 2 * (j * stride + i) + 1;
    // Crossing on the edge from corner a to corner b (a left of or above b).
    let crossing = |a: (usize, usize), b: (usize, usize)| {
        let f = match (value(a.0, a.1), value(b.0, b.1)) {
            (Some(va), Some(vb)) => ((level - va) / (vb - va)).clamp(0.0, 1.0),
            _ => 0.5,
        };
        let (ax, ay) = (a.0 as f64 - 0.5, a.1 as f64 - 0.5);
        let (bx, by) = (b.0 as f64 - 0.5, b.1 as f64 - 0.5);
        Complex::new(ax + (bx - ax) * f, ay + (by - ay) * f)
    };
    let mut next: HashMap<usize, usize> = HashMap::new();
    let mut points: HashMap<usize, Complex> = HashMap::new();
    let mut starts = Vec::new();
    for j in 0..=h {
        for i in 0..=w {
            // Corners and edges clockwise on screen: top, right, bottom, left.
            let corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
            let state = corners.map(inside);
            if state.iter().all(|v| *v) || state.iter().all(|v| !*v) {
                continue;
            }
            let edges = [horizontal(i, j), vertical(i + 1, j), horizontal(i, j + 1), vertical(i, j)];
            let ends = [
                (corners[0], corners[1]),
                (corners[1], corners[2]),
                (corners[3], corners[2]),
                (corners[0], corners[3]),
            ];
            // Going clockwise, edge e leaves the inside when corner e is in
            // and e+1 is out. A segment from that crossing to the next one
            // that enters keeps the inside on its right.
            let exits: Vec<usize> = (0..4).filter(|&e| state[e] && !state[(e + 1) % 4]).collect();
            let centre_inside = exits.len() == 2 && {
                let sum: f64 = corners.iter().map(|&(a, b)| value(a, b).unwrap_or(f64::NEG_INFINITY)).sum();
                sum / 4.0 >= level
            };
            for &e in &exits {
                // One crossing pair: the only entering edge. A saddle (corners
                // alternate): connected insides join through the centre, so
                // pair with the next edge clockwise; separate insides cut off
                // their corners, so pair with the previous one.
                let enter = match (exits.len(), centre_inside) {
                    (1, _) => (0..4).find(|&k| !state[k] && state[(k + 1) % 4]).unwrap(),
                    (_, true) => (e + 1) % 4,
                    (_, false) => (e + 3) % 4,
                };
                for k in [e, enter] {
                    points.entry(edges[k]).or_insert_with(|| crossing(ends[k].0, ends[k].1));
                }
                next.insert(edges[e], edges[enter]);
                starts.push(edges[e]);
            }
        }
    }
    let mut visited = HashSet::new();
    let mut result = Vec::new();
    for &first in &starts {
        if visited.contains(&first) {
            continue;
        }
        let (mut outline, mut edge) = (Vec::new(), first);
        loop {
            visited.insert(edge);
            outline.push(points[&edge]);
            edge = next[&edge];
            if edge == first {
                break;
            }
        }
        result.push(outline);
    }
    result
}

/// The area a traced outline encloses, as a bitmap around it, for the
/// region term: `value` is its coverage at any point (bilinear between
/// pixel centres, 0 outside the box).
struct Region {
    x0: f64,
    y0: f64,
    w: usize,
    h: usize,
    cover: Vec<f64>,
    /// +1 for an outer outline, -1 for a hole (whose curve winds -1).
    sign: f64,
    /// REGION_WEIGHT divided by the outline's length.
    scale: f64,
    /// Highest harmonic the region term may move (see `chamfer`).
    harmonics: usize,
}
impl Region {
    fn new(points: &[Complex], weight: f64) -> Result<Self, Error> {
        let low = points.iter().fold((f64::INFINITY, f64::INFINITY), |m, p| (m.0.min(p.re), m.1.min(p.im)));
        let high = points.iter().fold((f64::NEG_INFINITY, f64::NEG_INFINITY), |m, p| (m.0.max(p.re), m.1.max(p.im)));
        let margin = (REGION_BLUR + 2) as f64;
        let (x0, y0) = (low.0.floor() - margin, low.1.floor() - margin);
        let (w, h) = ((high.0.ceil() - x0 + margin + 1.0) as usize, (high.1.ceil() - y0 + margin + 1.0) as usize);
        let polygon = Polyline {
            vertices: points.iter().map(|p| Point2::new(p.re - x0, p.im - y0)).collect(),
            closed: true,
        };
        let cover = Rasterizer::new(w, h)?.render(&FlattenedPath { subpaths: vec![polygon] }, 1)?;
        let length: f64 = (0..points.len()).map(|i| (points[(i + 1) % points.len()] - points[i]).abs()).sum();
        let scale = weight / length.max(1.0);
        Ok(Self {
            x0,
            y0,
            w,
            h,
            cover: blur(cover.pixels(), w, h, REGION_BLUR),
            sign: signed_area(points).signum(),
            scale,
            harmonics: ((0.5 / (RATE * std::f64::consts::TAU * scale)) as usize).max(1),
        })
    }
    fn value(&self, z: Complex) -> f64 {
        let (u, v) = (z.re - self.x0 - 0.5, z.im - self.y0 - 0.5);
        if !(u >= 0.0 && v >= 0.0 && u < (self.w - 1) as f64 && v < (self.h - 1) as f64) {
            return 0.0;
        }
        let (i, j) = (u.floor() as usize, v.floor() as usize);
        let (fu, fv) = (u - i as f64, v - j as f64);
        let at = |x: usize, y: usize| self.cover[y * self.w + x];
        (at(i, j) * (1.0 - fu) + at(i + 1, j) * fu) * (1.0 - fv) + (at(i, j + 1) * (1.0 - fu) + at(i + 1, j + 1) * fu) * fv
    }
}

/// Box blur of radius r, rows then columns (values beyond the edge are 0).
fn blur(values: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
    let pass = |input: &[f64], along: usize, across: usize, at: &dyn Fn(usize, usize) -> usize| {
        let mut out = vec![0.0; input.len()];
        for b in 0..across {
            let mut sum = 0.0;
            for a in 0..along.min(r) {
                sum += input[at(a, b)];
            }
            for a in 0..along {
                if a + r < along {
                    sum += input[at(a + r, b)];
                }
                if a > r {
                    sum -= input[at(a - r - 1, b)];
                }
                out[at(a, b)] = sum / (2 * r + 1) as f64;
            }
        }
        out
    };
    let rows = pass(values, w, h, &|x, y| y * w + x);
    pass(&rows, h, w, &|y, x| y * w + x)
}

/// Coefficients c_k, k = -m..=m, at index k + m, put into an FFT buffer of
/// length s (bin k mod s).
fn to_buffer(coefficients: &[Complex], m: usize, s: usize) -> Vec<Complex> {
    let mut buffer = vec![Complex::ZERO; s];
    for (i, c) in coefficients.iter().enumerate() {
        let k = i as i64 - m as i64;
        buffer[k.rem_euclid(s as i64) as usize] = *c;
    }
    buffer
}

/// z_j = sum_k c_k e^(2 pi i k j / s): an inverse FFT, times s because our
/// inverse transform divides by the length.
fn samples(coefficients: &[Complex], m: usize, s: usize) -> Result<Vec<Complex>, Error> {
    let mut buffer = to_buffer(coefficients, m, s);
    fft(&mut buffer, true)?;
    Ok(buffer.into_iter().map(|z| z.scale(s as f64)).collect())
}

/// Coefficients of an ordered loop: resample it to s points evenly spaced
/// along its length, FFT, and keep |k| <= m: c_k = X[k] / s.
fn dft(points: &[Complex], m: usize, s: usize) -> Result<Vec<Complex>, Error> {
    let n = points.len();
    let mut along = Vec::with_capacity(n + 1);
    along.push(0.0);
    for i in 0..n {
        along.push(along[i] + (points[(i + 1) % n] - points[i]).abs());
    }
    let total = along[n];
    let mut i = 0;
    let mut buffer: Vec<Complex> = (0..s)
        .map(|j| {
            let u = j as f64 * total / s as f64;
            while i + 1 < n && along[i + 1] <= u {
                i += 1;
            }
            let length = along[i + 1] - along[i];
            let f = if length > 0.0 { (u - along[i]) / length } else { 0.0 };
            points[i] + (points[(i + 1) % n] - points[i]).scale(f)
        })
        .collect();
    fft(&mut buffer, false)?;
    Ok((-(m as i64)..=m as i64)
        .map(|k| buffer[k.rem_euclid(s as i64) as usize].scale(1.0 / s as f64))
        .collect())
}

/// Symmetric chamfer loss of the curve against `targets`, and the gradient
/// with respect to every c_k, with the region term's pull added when one is
/// given. The gradient sum_j g_j e^(-ik t_j) is the forward FFT of the
/// per-sample pulls g_j.
///
/// Region term: L = sign * integral of winding(x) * (1 - 2 cover(x)), which
/// is smallest when the curve encloses exactly the covered area. Moving a
/// boundary point along its outward normal n by d changes it by
/// sign * (1 - 2 cover) * d * ds, so g_j = sign * (1 - 2 cover(z_j)) times
/// n_j times |tangent|. For screen coordinates (y down) and the inside on
/// the right, the outward normal of tangent t is (t.im, -t.re).
///
/// The enclosed area is pi * sum of k |c_k|^2, so where the curve lies
/// inside the covered area this term rewards every harmonic for growing,
/// in proportion to k: high harmonics could wind the curve round and round
/// to collect area, and the descent diverges. The term therefore moves
/// only |k| <= K, where one step changes c_k by at most half of it
/// (RATE * 2 pi * k * scale <= 0.5). Shortcuts across an outline are large
/// shapes, which low harmonics can correct.
fn chamfer(
    coefficients: &[Complex],
    m: usize,
    s: usize,
    targets: &[Complex],
    target_tree: &KdTree,
    region: Option<&Region>,
) -> Result<(f64, Vec<Complex>), Error> {
    let z = samples(coefficients, m, s)?;
    let sample_tree = KdTree::new(&z);
    let mut pull = vec![Complex::ZERO; s];
    let mut loss = 0.0;
    let (sf, pf) = (s as f64, targets.len() as f64);
    for (j, zj) in z.iter().enumerate() {
        let (d, i) = target_tree.nearest(*zj);
        loss += d / sf;
        pull[j] = pull[j] + (*zj - targets[i]).scale(2.0 / sf);
    }
    for p in targets {
        let (d, j) = sample_tree.nearest(*p);
        loss += d / pf;
        pull[j] = pull[j] + (z[j] - *p).scale(2.0 / pf);
    }
    fft(&mut pull, false)?;
    let mut gradient: Vec<Complex> = (-(m as i64)..=m as i64)
        .map(|k| pull[k.rem_euclid(s as i64) as usize])
        .collect();
    if let Some(r) = region {
        let mut push: Vec<Complex> = (0..s)
            .map(|j| {
                let t = (z[(j + 1) % s] - z[(j + s - 1) % s]).scale(0.5);
                let f = 1.0 - 2.0 * r.value(z[j]);
                Complex::new(t.im, -t.re).scale(r.sign * r.scale * f)
            })
            .collect();
        fft(&mut push, false)?;
        let top = r.harmonics.min(m) as i64;
        for k in -top..=top {
            let i = (k + m as i64) as usize;
            gradient[i] = gradient[i] + push[k.rem_euclid(s as i64) as usize];
        }
    }
    Ok((loss, gradient))
}

/// `steps` of c_k -= RATE * dL/dc_k; returns the chamfer loss after the
/// last step (the region term only steers).
fn descend(
    coefficients: &mut [Complex],
    m: usize,
    s: usize,
    targets: &[Complex],
    target_tree: &KdTree,
    region: Option<&Region>,
    steps: usize,
) -> Result<f64, Error> {
    for _ in 0..steps {
        let (_, gradient) = chamfer(coefficients, m, s, targets, target_tree, region)?;
        for (c, g) in coefficients.iter_mut().zip(gradient) {
            *c = *c - g.scale(RATE);
        }
    }
    Ok(chamfer(coefficients, m, s, targets, target_tree, None)?.0)
}

/// Same curve with more harmonics; the new ones start at zero.
fn widen(coefficients: &[Complex], from: usize, to: usize) -> Vec<Complex> {
    let mut wider = vec![Complex::ZERO; 2 * to + 1];
    wider[to - from..to + from + 1].copy_from_slice(coefficients);
    wider
}

/// Descent alone: a circle at the centroid, mean radius, turning the same
/// way as the loop (c_1 for screen-clockwise, c_-1 otherwise), then the
/// harmonics multiplied by 4 stage by stage up to m.
fn from_circle(
    points: &[Complex],
    m: usize,
    s: usize,
    tree: &KdTree,
    region: Option<&Region>,
) -> Result<(Vec<Complex>, f64), Error> {
    let n = points.len();
    let centroid = points.iter().fold(Complex::ZERO, |sum, p| sum + *p).scale(1.0 / n as f64);
    let radius = points.iter().map(|p| (*p - centroid).abs()).sum::<f64>() / n as f64;
    let mut stage = 1usize;
    let mut c = vec![Complex::ZERO; 3];
    c[1] = centroid;
    c[if signed_area(points) > 0.0 { 2 } else { 0 }] = Complex::new(radius, 0.0);
    loop {
        let next = (stage * 4).min(m);
        c = widen(&c, stage, next);
        stage = next;
        let loss = descend(&mut c, stage, s, points, tree, region, STAGE_STEPS)?;
        if stage == m {
            return Ok((c, loss));
        }
    }
}

/// Which fits to make beyond the DFT, which is always made.
#[derive(Clone, Copy)]
struct Want {
    refined: bool,
    descent: bool,
}

/// The fit an export draws with.
#[derive(Clone, Copy, PartialEq)]
pub enum FitKind {
    Dft,
    Refined,
    Descent,
}

/// The fits of one outline: the DFT first, then refined and descent when
/// wanted, in that order.
struct Fit {
    n: usize,
    m: usize,
    s: usize,
    coefficients: Vec<Vec<Complex>>,
    losses: Vec<f64>,
    seconds: Vec<f64>,
}

fn fit(points: &[Complex], want: Want) -> Result<Fit, Error> {
    let n = points.len();
    let m = (n / 3).clamp(2, MAX_HARMONICS);
    let s = (2 * n).max(4 * m + 2).next_power_of_two().max(64);
    let tree = KdTree::new(points);
    let mut f = Fit { n, m, s, coefficients: Vec::new(), losses: Vec::new(), seconds: Vec::new() };
    let started = Instant::now();
    let dft_coefficients = dft(points, m, s)?;
    f.losses.push(chamfer(&dft_coefficients, m, s, points, &tree, None)?.0);
    f.coefficients.push(dft_coefficients);
    f.seconds.push(started.elapsed().as_secs_f64());
    if want.refined {
        // Refined: descend from the DFT result.
        let started = Instant::now();
        let mut refined = f.coefficients[0].clone();
        f.losses.push(descend(&mut refined, m, s, points, &tree, None, REFINE_STEPS)?);
        f.coefficients.push(refined);
        f.seconds.push(started.elapsed().as_secs_f64());
    }
    if want.descent {
        // Descent alone.
        let started = Instant::now();
        let (c, loss) = from_circle(points, m, s, &tree, None)?;
        f.coefficients.push(c);
        f.losses.push(loss);
        f.seconds.push(started.elapsed().as_secs_f64());
    }
    Ok(f)
}

/// Fits every outline, spread over the available cores. The outlines wait
/// in one queue, largest first, and each thread takes the next one as soon
/// as it is free, so no core idles while another works through a long
/// list. Results keep the outlines' order, so the run does not depend on
/// timing.
fn fit_all(traced: &[(Vec<Complex>, Want)]) -> Result<Vec<Fit>, Error> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(32);
    let mut order: Vec<usize> = (0..traced.len()).collect();
    // Descent from a circle costs about five refinements.
    let cost = |w: Want| 1 + usize::from(w.refined) + 6 * usize::from(w.descent);
    order.sort_by_key(|&i| std::cmp::Reverse(traced[i].0.len() * cost(traced[i].1)));
    let next = AtomicUsize::new(0);
    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&i) = order.get(k) else { break };
                        done.push((i, fit(&traced[i].0, traced[i].1)?));
                    }
                    Ok::<_, Error>(done)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("fitting thread panicked"))
            .collect::<Result<Vec<_>, Error>>()
    })?;
    let mut all: Vec<(usize, Fit)> = results.into_iter().flatten().collect();
    all.sort_by_key(|(i, _)| *i);
    Ok(all.into_iter().map(|(_, f)| f).collect())
}

/// Rounds to the 8 bits a BMP stores, so every measure can be recomputed
/// from the saved files.
fn eight_bit(image: &Image<Color>) -> Result<Image<Color>, Error> {
    let r = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() / 255.0;
    image.map(|c| Color::rgb(r(c.r), r(c.g), r(c.b)))
}

fn measure(image: &Image<Color>, reference: &Image<Color>) -> Result<String, Error> {
    let a = eight_bit(image)?;
    let d: Vec<f64> = a.pixels().iter().zip(reference.pixels()).map(|(x, y)| delta_e(*x, *y)).collect();
    Ok(format!(
        "{{\"mean_delta_e\":{:.6},\"visible\":{},\"ssim\":{:.6}}}",
        d.iter().sum::<f64>() / d.len() as f64,
        d.iter().filter(|v| **v > VISIBLE_DELTA_E).count(),
        ssim(&a, reference)
    ))
}

fn json_list(values: &[f64], digits: usize) -> String {
    format!("[{}]", values.iter().map(|v| format!("{v:.digits$}")).collect::<Vec<_>>().join(","))
}

/// The smooth outlines of one channel: its levels (k-means on the smoothed
/// values), the thresholds between them (midpoints), and for each level
/// j >= 1 the sub-pixel iso-lines at threshold j - 1. Areas under MIN_AREA
/// pixels are dropped (islands vanish, small holes close), as the crack
/// outlines merge them; the count dropped is returned last.
#[allow(clippy::type_complexity)]
fn smooth_outlines(values: &[f64], w: usize, h: usize, count: usize) -> (Vec<f64>, Vec<f64>, Vec<Vec<Vec<Complex>>>, usize) {
    let mut labels = kmeans_1d(values, count);
    let levels = levels_of(values, &mut labels);
    let thresholds: Vec<f64> = levels.windows(2).map(|p| (p[0] + p[1]) / 2.0).collect();
    let mut dropped = 0;
    let outlines = thresholds
        .iter()
        .map(|&t| {
            contours(values, w, h, t)
                .into_iter()
                .filter(|l| {
                    let keep = signed_area(l).abs() >= MIN_AREA as f64;
                    dropped += usize::from(!keep);
                    keep
                })
                .collect()
        })
        .collect();
    (levels, thresholds, outlines, dropped)
}

/// One mask to fill: plate, outline family (0 crack, 1 smooth), level
/// index, level value, and the range of its outlines in the traced list.
struct Mask {
    channel: usize,
    family: usize,
    level: usize,
    value: f64,
    outlines: std::ops::Range<usize>,
}

fn painting(output: &Path, path: &Path, factor: Option<usize>, sweep: &mut Vec<String>) -> Result<String, Error> {
    let started = Instant::now();
    let name = picture_name(path)?;
    let source = bmp::read(&std::fs::read(path)?)?;
    let factor = shrink_factor(source.width(), source.height(), LONGEST, factor);
    let input = shrink(&source, factor)?;
    let (w, h) = (input.width(), input.height());
    let reference = eight_bit(&input)?;
    input.save_bmp(output.join(format!("hifi_{name}_input.bmp")))?;
    let smooth = bilateral(&input)?;
    smooth.save_bmp(output.join(format!("hifi_{name}_smooth.bmp")))?;
    let raw_labs: Vec<[f64; 3]> = input.pixels().iter().map(|c| lab(*c)).collect();
    let smooth_labs: Vec<[f64; 3]> = smooth.pixels().iter().map(|c| lab(*c)).collect();
    let raster = Rasterizer::new(w, h)?;
    let names = ["lightness", "a", "b"];
    // Plate values per bitmap of IMAGES.
    let mut plates: Vec<Vec<Image<f64>>> = Vec::new();
    let mut traced: Vec<(Vec<Complex>, Want)> = Vec::new();
    let (crack, smooth_all) = (Want { refined: true, descent: false }, Want { refined: true, descent: true });
    let mut masks: Vec<Mask> = Vec::new();
    let (mut raw_json, mut smooth_json) = (Vec::new(), Vec::new());
    let mut dropped = 0usize;
    for (channel, plate) in names.iter().enumerate() {
        // Crack outlines: raw levels, small regions merged, pixel cracks.
        let raw: Vec<f64> = raw_labs.iter().map(|v| v[channel]).collect();
        let mut labels = kmeans_1d(&raw, LEVELS[channel]);
        despeckle(&mut labels, w, h);
        let raw_levels = levels_of(&raw, &mut labels);
        raw_json.push(format!("\"{plate}\":{}", json_list(&raw_levels, 9)));
        // Smooth outlines: levels of the smoothed channel; a pixel takes the
        // level whose interval holds its value, the boundaries being the
        // midpoints between neighbouring levels.
        let values: Vec<f64> = smooth_labs.iter().map(|v| v[channel]).collect();
        let (levels, thresholds, smooth, lost) = smooth_outlines(&values, w, h, LEVELS[channel]);
        dropped += lost;
        smooth_json.push(format!(
            "\"{plate}\":{{\"levels\":{},\"thresholds\":{}}}",
            json_list(&levels, 9),
            json_list(&thresholds, 9)
        ));
        let mut images: Vec<Image<f64>> = Vec::new();
        for image in IMAGES {
            images.push(Image::new(w, h, if image == "quant_raw" || image.starts_with("crack") { raw_levels[0] } else { levels[0] })?);
        }
        for i in 0..w * h {
            *images[0].get_mut(i % w, i / w).unwrap() = raw_levels[labels[i]];
            let index = thresholds.iter().filter(|t| values[i] >= **t).count();
            *images[3].get_mut(i % w, i / w).unwrap() = levels[index];
        }
        plates.push(images);
        for (j, &value) in raw_levels.iter().enumerate().skip(1) {
            let inside: Vec<bool> = labels.iter().map(|&l| l >= j).collect();
            let found = loops(&inside, w, h);
            if !found.is_empty() {
                masks.push(Mask { channel, family: 0, level: j, value, outlines: traced.len()..traced.len() + found.len() });
                traced.extend(found.into_iter().map(|l| (l, crack)));
            }
        }
        for ((j, &value), found) in levels.iter().enumerate().skip(1).zip(smooth) {
            if !found.is_empty() {
                masks.push(Mask { channel, family: 1, level: j, value, outlines: traced.len()..traced.len() + found.len() });
                traced.extend(found.into_iter().map(|l| (l, smooth_all)));
            }
        }
    }
    // Every outline of every mask through one queue.
    let fits = fit_all(&traced)?;
    sweep.extend(region_sweep(&name, &traced, &masks, w, h)?);
    let mut loops_file = BufWriter::new(File::create(output.join(format!("hifi_{name}_loops.csv")))?);
    let columns = |prefix: &str| VARIANTS.iter().map(|v| format!("{prefix}_{v}")).collect::<Vec<_>>().join(",");
    writeln!(loops_file, "outline,plate,level,loop,n,harmonics,samples,{},{}", columns("loss"), columns("seconds"))?;
    // Per family: masks, loops, points, coefficients; per variant: weighted
    // edge error, seconds, stuck count.
    let mut count = [[0usize; 4]; 2];
    let mut weighted = [[0.0f64; 3]; 2];
    let mut seconds = [[0.0f64; 3]; 2];
    let mut stuck = 0usize;
    // Render each mask in level order (each level lies over the ones below).
    // Traced outlines run clockwise outside and counter-clockwise around
    // holes, every fit keeps that direction, so one non-zero fill of a
    // mask's outlines cuts its holes. Its variants fill in parallel.
    for mask in &masks {
        let fits = &fits[mask.outlines.clone()];
        let targets: Vec<usize> = if mask.family == 0 { vec![1, 2] } else { vec![4, 5, 6] };
        let images = &mut plates[mask.channel];
        let mut slots: Vec<&mut Image<f64>> = images
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| targets.contains(i))
            .map(|(_, image)| image)
            .collect();
        std::thread::scope(|scope| {
            let handles: Vec<_> = slots
                .iter_mut()
                .enumerate()
                .map(|(variant, image)| {
                    let raster = &raster;
                    let value = mask.value;
                    scope.spawn(move || {
                        let subpaths: Vec<Polyline> = fits
                            .iter()
                            .map(|f| {
                                samples(&f.coefficients[variant], f.m, f.s).map(|z| Polyline {
                                    vertices: z.iter().map(|q| Point2::new(q.re, q.im)).collect(),
                                    closed: true,
                                })
                            })
                            .collect::<Result<_, _>>()?;
                        let coverage = raster.render(&FlattenedPath { subpaths }, FILL_K)?;
                        for (i, m) in coverage.pixels().iter().enumerate() {
                            let v = image.get_mut(i % w, i / w).unwrap();
                            *v = *v * (1.0 - m) + value * m;
                        }
                        Ok::<_, Error>(())
                    })
                })
                .collect();
            handles
                .into_iter()
                .try_for_each(|h| h.join().expect("render thread panicked"))
        })?;
        count[mask.family][0] += 1;
        for (k, f) in fits.iter().enumerate() {
            let cell = |v: &[f64], i: usize, digits: usize| v.get(i).map_or(String::new(), |x| format!("{x:.digits$}"));
            writeln!(
                loops_file,
                "{},{},{},{k},{},{},{},{},{}",
                ["crack", "smooth"][mask.family],
                names[mask.channel],
                mask.level,
                f.n,
                f.m,
                f.s,
                (0..3).map(|i| cell(&f.losses, i, 9)).collect::<Vec<_>>().join(","),
                (0..3).map(|i| cell(&f.seconds, i, 6)).collect::<Vec<_>>().join(",")
            )?;
            count[mask.family][1] += 1;
            count[mask.family][2] += f.n;
            count[mask.family][3] += 2 * f.m + 1;
            for (v, loss) in f.losses.iter().enumerate() {
                weighted[mask.family][v] += f.n as f64 * loss.sqrt();
                seconds[mask.family][v] += f.seconds[v];
            }
            if mask.family == 1 {
                stuck += usize::from(f.losses[2] > STUCK_FACTOR * f.losses[0] + STUCK_ALLOWANCE);
            }
        }
    }
    loops_file.flush()?;
    let mut measures = Vec::new();
    for (v, variant) in IMAGES.iter().enumerate() {
        let mut image = Image::new(w, h, Color::rgb(0.0, 0.0, 0.0))?;
        for i in 0..w * h {
            *image.get_mut(i % w, i / w).unwrap() = lab_to_rgb(
                plates[0][v].pixels()[i],
                plates[1][v].pixels()[i],
                plates[2][v].pixels()[i],
            );
        }
        image.save_bmp(output.join(format!("hifi_{name}_{variant}.bmp")))?;
        measures.push(format!("\"{variant}\":{}", measure(&image, &reference)?));
    }
    let family = |f: usize, variants: &[&str]| {
        let points = count[f][2].max(1) as f64;
        format!(
            "{{\"masks\":{},\"loops\":{},\"edge_points\":{},\"coefficients\":{},\"edge_rms\":{{{}}},\"compute_seconds\":{{{}}}}}",
            count[f][0],
            count[f][1],
            count[f][2],
            count[f][3],
            variants.iter().enumerate().map(|(v, n)| format!("\"{n}\":{:.6}", weighted[f][v] / points)).collect::<Vec<_>>().join(","),
            variants.iter().enumerate().map(|(v, n)| format!("\"{n}\":{:.3}", seconds[f][v])).collect::<Vec<_>>().join(",")
        )
    };
    Ok(format!(
        "\"{name}\":{{\"source\":[{},{}],\"factor\":{factor},\"size\":[{w},{h}],\"bitmap_bytes\":{},\"raw_levels\":{{{}}},\"smooth_levels\":{{{}}},\"dropped_small\":{dropped},\"crack\":{},\"smooth\":{},\"descent_stuck\":{stuck},\"wall_seconds\":{:.3},\"measures\":{{{}}}}}",
        source.width(),
        source.height(),
        3 * w * h,
        raw_json.join(","),
        smooth_json.join(","),
        family(0, &VARIANTS[..2]),
        family(1, &VARIANTS),
        started.elapsed().as_secs_f64(),
        measures.join(",")
    ))
}

/// Runs the high-fidelity comparison on each picture, shrunk by `factor`
/// or by the LONGEST rule. `fingerprint` identifies the source that
/// produced the files.
pub fn run(output: &Path, images: &[PathBuf], factor: Option<usize>, fingerprint: &str) -> Result<(), Error> {
    let mut entries = Vec::new();
    let mut sweep = Vec::new();
    for path in images {
        entries.push(painting(output, path, factor, &mut sweep)?);
        println!("hi-fi: {} done", path.display());
    }
    std::fs::write(
        output.join("hifi_region_sweep.csv"),
        format!("painting,plate,level,points,harmonics,inside_pixels,fit,weight,loss,wrong_pixels,diverged,seconds\n{}", sweep.concat()),
    )?;
    std::fs::write(
        output.join("hifi_summary.json"),
        format!(
            "{{\"source_fingerprint\":\"{fingerprint}\",\"threads\":{},\"levels\":[{},{},{}],\"max_harmonics\":{MAX_HARMONICS},\"refine_steps\":{REFINE_STEPS},\"stage_steps\":{STAGE_STEPS},\"rate\":{RATE},\"fill_k\":{FILL_K},\"min_area\":{MIN_AREA},\"visible_delta_e\":{VISIBLE_DELTA_E},\"stuck_factor\":{STUCK_FACTOR},\"stuck_allowance\":{STUCK_ALLOWANCE},\"region_blur\":{REGION_BLUR},\"region_weights\":[{}],\"diverged_loss\":{DIVERGED_LOSS},\"paintings\":{{{}}}}}\n",
            std::thread::available_parallelism().map_or(4, |n| n.get()).min(32),
            LEVELS[0],
            LEVELS[1],
            LEVELS[2],
            REGION_WEIGHTS.map(|w| w.to_string()).join(","),
            entries.join(",")
        ),
    )?;
    Ok(())
}

/// The region-term sweep: the largest smooth outline of the a* and b* plates
/// is fitted from a circle with each of REGION_WEIGHTS (in parallel), and
/// by the DFT for reference. Each fit is filled (non-zero rule, one sample
/// per pixel) and compared with the traced outline's own fill; the pixels
/// that differ are counted. Returns CSV rows.
fn region_sweep(name: &str, traced: &[(Vec<Complex>, Want)], masks: &[Mask], w: usize, h: usize) -> Result<Vec<String>, Error> {
    let names = ["lightness", "a", "b"];
    let raster = Rasterizer::new(w, h)?;
    let fill = |points: &[Complex]| {
        raster.render(
            &FlattenedPath {
                subpaths: vec![Polyline { vertices: points.iter().map(|q| Point2::new(q.re, q.im)).collect(), closed: true }],
            },
            1,
        )
    };
    let mut rows = Vec::new();
    for channel in [1, 2] {
        let Some((mask, index)) = masks
            .iter()
            .filter(|m| m.family == 1 && m.channel == channel)
            .flat_map(|m| m.outlines.clone().map(move |i| (m, i)))
            .max_by_key(|&(_, i)| (traced[i].0.len(), std::cmp::Reverse(i)))
        else {
            continue;
        };
        let points = &traced[index].0;
        let n = points.len();
        let m = (n / 3).clamp(2, MAX_HARMONICS);
        let s = (2 * n).max(4 * m + 2).next_power_of_two().max(64);
        let tree = KdTree::new(points);
        let truth = fill(points)?;
        let inside = truth.pixels().iter().filter(|v| **v > 0.5).count();
        let started = Instant::now();
        let dft_coefficients = dft(points, m, s)?;
        let dft_loss = chamfer(&dft_coefficients, m, s, points, &tree, None)?.0;
        let mut results = vec![("dft", 0.0, dft_coefficients, dft_loss, started.elapsed().as_secs_f64())];
        let descents = std::thread::scope(|scope| {
            let handles: Vec<_> = REGION_WEIGHTS
                .iter()
                .map(|&weight| {
                    let tree = &tree;
                    scope.spawn(move || {
                        let started = Instant::now();
                        let region = Region::new(points, weight)?;
                        let (c, loss) = from_circle(points, m, s, tree, (weight > 0.0).then_some(&region))?;
                        Ok::<_, Error>(("descent", weight, c, loss, started.elapsed().as_secs_f64()))
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("sweep thread panicked"))
                .collect::<Result<Vec<_>, Error>>()
        })?;
        results.extend(descents);
        for (fit, weight, c, loss, seconds) in results {
            let diverged = !loss.is_finite() || loss > DIVERGED_LOSS || c.iter().any(|v| !v.re.is_finite() || !v.im.is_finite());
            // A diverged curve may be too wild to fill; its count is left empty.
            let wrong = if diverged {
                String::new()
            } else {
                let filled = fill(&samples(&c, m, s)?)?;
                filled.pixels().iter().zip(truth.pixels()).filter(|(a, b)| a != b).count().to_string()
            };
            rows.push(format!(
                "{name},{},{},{n},{m},{inside},{fit},{weight},{loss:.6},{wrong},{},{seconds:.3}\n",
                names[channel],
                mask.level,
                u8::from(diverged)
            ));
        }
    }
    Ok(rows)
}

/// One curve: outline points, harmonics m, coefficients c_-m..c_m.
type Curve = (usize, usize, Vec<Complex>);
/// One mask: plate, level, value, and its curves.
type Layer = (usize, usize, f64, Vec<Curve>);

/// A picture as Fourier series: canvas size, each plate's base level, and
/// its masks. Plates are L*, a*, b*; masks come in level order within each
/// plate.
struct Drawing {
    w: usize,
    h: usize,
    base: [f64; 3],
    masks: Vec<Layer>,
}

const PLATES: [&str; 3] = ["lightness", "a", "b"];

impl Drawing {
    /// Rendered `scale` times larger: each mask's curves are sampled as in
    /// the fits (s from the outline's points), times `scale`, and filled
    /// together with the non-zero rule, each level over the ones below.
    fn bitmap(&self, scale: usize) -> Result<Image<Color>, Error> {
        let (w, h) = (self.w * scale, self.h * scale);
        let raster = Rasterizer::new(w, h)?;
        let mut plates = Vec::new();
        for base in self.base {
            plates.push(Image::new(w, h, base)?);
        }
        for (channel, _, value, curves) in &self.masks {
            let subpaths: Vec<Polyline> = curves
                .iter()
                .map(|(n, m, c)| {
                    let s = (2 * n).max(4 * m + 2).next_power_of_two().max(64) * scale.next_power_of_two();
                    samples(c, *m, s).map(|z| Polyline {
                        vertices: z.iter().map(|q| Point2::new(q.re * scale as f64, q.im * scale as f64)).collect(),
                        closed: true,
                    })
                })
                .collect::<Result<_, _>>()?;
            let coverage = raster.render(&FlattenedPath { subpaths }, FILL_K)?;
            let image = &mut plates[*channel];
            for (i, m) in coverage.pixels().iter().enumerate() {
                let v = image.get_mut(i % w, i / w).unwrap();
                *v = *v * (1.0 - m) + value * m;
            }
        }
        let mut image = Image::new(w, h, Color::rgb(0.0, 0.0, 0.0))?;
        for i in 0..w * h {
            *image.get_mut(i % w, i / w).unwrap() =
                lab_to_rgb(plates[0].pixels()[i], plates[1].pixels()[i], plates[2].pixels()[i]);
        }
        Ok(image)
    }

    /// The browser SVG, `scale` times larger: each plate lights only its own
    /// channel (as in the showcase's browser SVG) and the Lab filter turns
    /// the three channels back into RGB. A curve with m harmonics is sampled
    /// at 4m points (at least twice its highest frequency), times `scale`;
    /// one non-zero path per level.
    fn write_svg(&self, path: &Path, scale: usize) -> Result<(), Error> {
        let (w, h) = (self.w * scale, self.h * scale);
        let encode = |channel: usize, v: f64| {
            let mut c = [0.0; 3];
            c[channel] = if channel == 0 { v / 100.0 } else { (v + 128.0) / 255.0 };
            hex(Color::rgb(c[0], c[1], c[2]))
        };
        let mut file = BufWriter::new(File::create(path)?);
        write!(
            file,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" width=\"{w}\" height=\"{h}\">\n{}  <g filter=\"url(#lab)\">\n    <rect width=\"{w}\" height=\"{h}\" fill=\"#000000\"/>\n",
            svg_lab_filter(w, h)
        )?;
        for (channel, base) in self.base.iter().enumerate() {
            writeln!(
                file,
                "    <g style=\"mix-blend-mode:screen;isolation:isolate\">\n    <rect width=\"{w}\" height=\"{h}\" fill=\"{}\"/>",
                encode(channel, *base)
            )?;
            for (_, _, value, curves) in self.masks.iter().filter(|m| m.0 == channel) {
                write!(file, "    <path fill=\"{}\" d=\"", encode(channel, *value))?;
                for (_, m, c) in curves {
                    let count = (4 * m).next_power_of_two().max(16) * scale.next_power_of_two();
                    let z = samples(c, *m, count)?;
                    let k = scale as f64;
                    write!(file, "M{:.1} {:.1}L", z[0].re * k, z[0].im * k)?;
                    for q in &z[1..] {
                        write!(file, "{:.1} {:.1} ", q.re * k, q.im * k)?;
                    }
                    write!(file, "Z")?;
                }
                writeln!(file, "\"/>")?;
            }
            writeln!(file, "    </g>")?;
        }
        writeln!(file, "  </g>\n</svg>")?;
        file.flush()?;
        Ok(())
    }

    /// Every c_k as one CSV row: plate, level, value, outline, points, k,
    /// re, im. With the summary's size and base levels this is the whole
    /// picture.
    fn write_coefficients(&self, path: &Path) -> Result<(), Error> {
        let mut file = BufWriter::new(File::create(path)?);
        writeln!(file, "plate,level,value,outline,points,k,re,im")?;
        for (channel, level, value, curves) in &self.masks {
            for (outline, (n, m, c)) in curves.iter().enumerate() {
                for (i, ck) in c.iter().enumerate() {
                    let k = i as i64 - *m as i64;
                    writeln!(file, "{},{level},{value:.9},{outline},{n},{k},{:.9},{:.9}", PLATES[*channel], ck.re, ck.im)?;
                }
            }
        }
        file.flush()?;
        Ok(())
    }

    /// Reads a coefficients CSV and the size and base levels from its JSON
    /// summary (the same stem, ending .json instead of _coefficients.csv).
    fn read(path: &Path) -> Result<Self, Error> {
        let bad = || Error::InvalidParameter("not a coefficients file written by `hifi IMAGE`");
        let name = path.file_name().and_then(|n| n.to_str()).ok_or_else(bad)?;
        let stem = name.strip_suffix("_coefficients.csv").ok_or_else(bad)?;
        let summary = std::fs::read_to_string(path.with_file_name(format!("{stem}.json")))?;
        // The hand-written summary has fixed keys, so a plain search suffices.
        let numbers_after = |key: &str| -> Result<Vec<f64>, Error> {
            let start = summary.find(key).ok_or_else(bad)? + key.len();
            let end = start + summary[start..].find(']').ok_or_else(bad)?;
            summary[start..end].split(',').map(|v| v.trim().parse::<f64>().map_err(|_| bad())).collect()
        };
        let size = numbers_after("\"size\":[")?;
        let mut base = [0.0; 3];
        for (channel, plate) in PLATES.iter().enumerate() {
            base[channel] = *numbers_after(&format!("\"{plate}\":{{\"levels\":["))?.first().ok_or_else(bad)?;
        }
        let mut drawing = Drawing { w: size[0] as usize, h: size[1] as usize, base, masks: Vec::new() };
        let text = std::fs::read_to_string(path)?;
        let mut key = None;
        for line in text.lines().skip(1) {
            let f: Vec<&str> = line.split(',').collect();
            if f.len() != 8 {
                return Err(bad());
            }
            let channel = PLATES.iter().position(|p| *p == f[0]).ok_or_else(bad)?;
            let number = |s: &str| s.parse::<f64>().map_err(|_| bad());
            let (level, value) = (f[1].parse::<usize>().map_err(|_| bad())?, number(f[2])?);
            let (outline, points, k) = (f[3], f[4].parse::<usize>().map_err(|_| bad())?, f[5].parse::<i64>().map_err(|_| bad())?);
            let c = Complex::new(number(f[6])?, number(f[7])?);
            if key != Some((channel, level)) {
                drawing.masks.push((channel, level, value, Vec::new()));
                key = Some((channel, level));
            }
            let curves = &mut drawing.masks.last_mut().unwrap().3;
            // Rows of one outline run from k = -m to m.
            if k < 0 && curves.last().is_none_or(|(_, m, c): &Curve| c.len() == 2 * m + 1) {
                curves.push((points, (-k) as usize, Vec::new()));
            }
            let current = curves.last_mut().ok_or_else(bad)?;
            if outline.is_empty() || current.0 != points {
                return Err(bad());
            }
            current.2.push(c);
        }
        if drawing.masks.iter().flat_map(|m| &m.3).any(|(_, m, c)| c.len() != 2 * m + 1) {
            return Err(bad());
        }
        Ok(drawing)
    }
}

/// Draws one picture from Fourier series with the smooth outlines, `levels`
/// L*, a*, b* levels and one kind of fit, and writes
/// export_<name>_<fit>_coefficients.csv (every c_k of every outline) and
/// export_<name>_<fit>.json (size, levels, counts, and mean ΔE and lightness
/// SSIM of the rendering against the input). `render` draws the CSV as SVG
/// or BMP at any scale. Returns the summary's path.
pub fn export(output: &Path, path: &Path, factor: Option<usize>, levels: [usize; 3], kind: FitKind) -> Result<PathBuf, Error> {
    let started = Instant::now();
    let name = picture_name(path)?;
    let source = bmp::read(&std::fs::read(path)?)?;
    let factor = shrink_factor(source.width(), source.height(), LONGEST, factor);
    let input = shrink(&source, factor)?;
    let (w, h) = (input.width(), input.height());
    let smooth = bilateral(&input)?;
    let labs: Vec<[f64; 3]> = smooth.pixels().iter().map(|c| lab(*c)).collect();
    let want = Want { refined: kind == FitKind::Refined, descent: kind == FitKind::Descent };
    let fit_name = match kind {
        FitKind::Dft => "dft",
        FitKind::Refined => "refined",
        FitKind::Descent => "descent",
    };
    let stem = format!("export_{name}_{fit_name}");
    let mut traced: Vec<(Vec<Complex>, Want)> = Vec::new();
    let mut masks: Vec<Mask> = Vec::new();
    let (mut dropped, mut base) = (0, [0.0; 3]);
    let mut levels_json = Vec::new();
    for (channel, plate) in PLATES.iter().enumerate() {
        let values: Vec<f64> = labs.iter().map(|v| v[channel]).collect();
        let (found_levels, thresholds, outlines, lost) = smooth_outlines(&values, w, h, levels[channel]);
        dropped += lost;
        levels_json.push(format!(
            "\"{plate}\":{{\"levels\":{},\"thresholds\":{}}}",
            json_list(&found_levels, 9),
            json_list(&thresholds, 9)
        ));
        for ((j, &value), found) in found_levels.iter().enumerate().skip(1).zip(outlines) {
            if !found.is_empty() {
                masks.push(Mask { channel, family: 1, level: j, value, outlines: traced.len()..traced.len() + found.len() });
                traced.extend(found.into_iter().map(|l| (l, want)));
            }
        }
        base[channel] = found_levels[0];
    }
    let fits = fit_all(&traced)?;
    // The chosen fit is the last one made.
    let drawing = Drawing {
        w,
        h,
        base,
        masks: masks
            .iter()
            .map(|m| {
                let curves = fits[m.outlines.clone()].iter().map(|f| (f.n, f.m, f.coefficients.last().unwrap().clone())).collect();
                (m.channel, m.level, m.value, curves)
            })
            .collect(),
    };
    let (mut points, mut coefficients, mut weighted) = (0usize, 0usize, 0.0);
    for f in &fits {
        points += f.n;
        coefficients += 2 * f.m + 1;
        weighted += f.n as f64 * f.losses.last().unwrap().sqrt();
    }
    let file_name = format!("{stem}_coefficients.csv");
    drawing.write_coefficients(&output.join(&file_name))?;
    // Measured on the rendering that `render` gives at scale 1.
    let measures = measure(&drawing.bitmap(1)?, &eight_bit(&input)?)?;
    let summary = output.join(format!("{stem}.json"));
    std::fs::write(
        &summary,
        format!(
            "{{\"image\":\"{name}\",\"source\":[{},{}],\"factor\":{factor},\"size\":[{w},{h}],\"fit\":\"{fit_name}\",\"levels\":{{{}}},\"outlines\":{},\"outline_points\":{points},\"coefficients\":{coefficients},\"dropped_small\":{dropped},\"edge_rms\":{:.6},\"measures\":{measures},\"coefficients_file\":\"{file_name}\",\"seconds\":{:.3}}}\n",
            source.width(),
            source.height(),
            levels_json.join(","),
            fits.len(),
            weighted / points.max(1) as f64,
            started.elapsed().as_secs_f64()
        ),
    )?;
    Ok(summary)
}

/// Draws a coefficients CSV written by `export`, `scale` times larger, as
/// the browser SVG and/or a BMP, named after the CSV's stem (with _x<scale>
/// when enlarged). Returns the files written.
pub fn render(output: &Path, path: &Path, scale: usize, svg: bool, bitmap: bool) -> Result<Vec<PathBuf>, Error> {
    let drawing = Drawing::read(path)?;
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    let stem = name.trim_end_matches("_coefficients.csv");
    let stem = if scale == 1 { stem.to_owned() } else { format!("{stem}_x{scale}") };
    let mut written = Vec::new();
    if svg {
        let file = output.join(format!("{stem}.svg"));
        drawing.write_svg(&file, scale)?;
        written.push(file);
    }
    if bitmap {
        let file = output.join(format!("{stem}.bmp"));
        drawing.bitmap(scale)?.save_bmp(&file)?;
        written.push(file);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_samples_and_dft_agree_with_the_direct_series() {
        let c = vec![
            Complex::new(0.5, -1.0),
            Complex::new(3.0, 1.0),
            Complex::new(10.0, 4.0),
            Complex::new(20.0, 0.0),
            Complex::new(-1.0, 2.0),
        ];
        let (m, s) = (2, 16);
        let z = samples(&c, m, s).unwrap();
        for (j, zj) in z.iter().enumerate() {
            let t = std::f64::consts::TAU * j as f64 / s as f64;
            let direct = c.iter().enumerate().fold(Complex::ZERO, |sum, (i, ck)| {
                sum + *ck * Complex::from_angle((i as f64 - m as f64) * t)
            });
            assert!((*zj - direct).abs() < 1e-9);
        }
        // Points evenly spaced in t are not evenly spaced in length here,
        // so check the round trip on a circle, where they are.
        let circle = vec![Complex::ZERO, Complex::ZERO, Complex::new(5.0, 5.0), Complex::new(7.0, 0.0), Complex::ZERO];
        let z = samples(&circle, m, s).unwrap();
        let back = dft(&z, m, s).unwrap();
        for (a, b) in back.iter().zip(&circle) {
            assert!((*a - *b).abs() < 1e-9);
        }
    }

    #[test]
    fn render_redraws_an_export_from_its_files() {
        // A small picture: two colour blocks and a disc on a light ground.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join(format!(
            "render-{}",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let (w, h) = (48, 40);
        let mut picture = Image::new(w, h, Color::rgb(0.9, 0.85, 0.7)).unwrap();
        for y in 0..h {
            for x in 0..w {
                let c = picture.get_mut(x, y).unwrap();
                if (6..20).contains(&x) && (5..30).contains(&y) {
                    *c = Color::rgb(0.8, 0.2, 0.1);
                } else if ((x as f64 - 33.0).powi(2) + (y as f64 - 20.0).powi(2)).sqrt() < 9.0 {
                    *c = Color::rgb(0.1, 0.2, 0.7);
                }
            }
        }
        let path = dir.join("blocks.bmp");
        picture.save_bmp(&path).unwrap();
        let summary = export(&dir, &path, Some(1), [4, 3, 3], FitKind::Dft).unwrap();
        let text = std::fs::read_to_string(&summary).unwrap();
        let recorded: f64 = text.split(r#""mean_delta_e":"#).nth(1).unwrap().split(',').next().unwrap().parse().unwrap();
        let csv = dir.join("export_blocks_dft_coefficients.csv");
        let written = render(&dir, &csv, 1, true, true).unwrap();
        assert_eq!(written.len(), 2);
        assert!(std::fs::read_to_string(&written[0]).unwrap().contains(r##"filter="url(#lab)""##));
        // Drawn from the file (coefficients rounded to 9 decimals), the
        // picture measures as it did when it was fitted.
        let redrawn = bmp::read(&std::fs::read(&written[1]).unwrap()).unwrap();
        let reference = eight_bit(&picture).unwrap();
        let again: f64 = measure(&redrawn, &reference).unwrap().split(r#""mean_delta_e":"#).nth(1).unwrap().split(',').next().unwrap().parse().unwrap();
        assert!((again - recorded).abs() < 1e-3, "{again} vs {recorded}");
        assert!(recorded < 10.0, "{recorded}");
        // Enlarged, the same series fills a canvas twice the size.
        let large = render(&dir, &csv, 2, false, true).unwrap();
        let enlarged = bmp::read(&std::fs::read(&large[0]).unwrap()).unwrap();
        assert_eq!((enlarged.width(), enlarged.height()), (2 * w, 2 * h));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn kd_tree_matches_a_brute_force_scan() {
        // A lattice has many equal distances, so the tie rule is exercised.
        let points: Vec<Complex> = (0..300)
            .map(|i| Complex::new((i % 17) as f64, (i / 17 * 3 % 23) as f64))
            .collect();
        let tree = KdTree::new(&points);
        for j in 0..500 {
            let q = Complex::new(j as f64 * 0.37 % 40.0 - 10.0, j as f64 * 0.91 % 50.0 - 12.0);
            let brute = points.iter().enumerate().fold((f64::INFINITY, usize::MAX), |best, (i, p)| {
                let d = (*p - q).norm_squared();
                if d < best.0 { (d, i) } else { best }
            });
            assert_eq!(tree.nearest(q), brute);
        }
    }

    #[test]
    fn contours_are_sub_pixel_closed_and_oriented() {
        // A disc of radius 6 with a hole of radius 2, as a smooth field.
        let (w, h) = (20, 20);
        let field: Vec<f64> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64 + 0.5 - 10.0, (i / w) as f64 + 0.5 - 10.0);
                let r = (x * x + y * y).sqrt();
                if r < 4.0 { r - 2.0 } else { 6.0 - r }
            })
            .collect();
        let found = contours(&field, w, h, 0.0);
        assert_eq!(found.len(), 2);
        let mut areas: Vec<f64> = found.iter().map(|l| signed_area(l)).collect();
        areas.sort_by(f64::total_cmp);
        // Hole counter-clockwise (negative), disc clockwise (positive), and
        // both close to the true areas (the iso-lines are sub-pixel).
        let pi = std::f64::consts::PI;
        assert!((areas[0] + pi * 4.0).abs() < 1.0, "{areas:?}");
        assert!((areas[1] - pi * 36.0).abs() < 1.5, "{areas:?}");
        for l in &found {
            for p in l {
                let r = ((p.re - 10.0).powi(2) + (p.im - 10.0).powi(2)).sqrt();
                assert!((r - 2.0).abs() < 0.05 || (r - 6.0).abs() < 0.05, "{r}");
            }
        }
        // A field touching the border closes along it, at the pixel edge.
        // Its four image corners are cut by 1/8 pixel each, as marching
        // squares cuts every convex pixel corner.
        let full = contours(&[1.0; 9], 3, 3, 0.5);
        assert_eq!(full.len(), 1);
        assert!((signed_area(&full[0]) - 8.5).abs() < 1e-9);
        // Diagonal neighbours stay apart or join by the centre value.
        let saddle = |d: f64| contours(&[d, 0.0, 0.0, d], 2, 2, 0.5).len();
        assert_eq!(saddle(0.6), 2);
        assert_eq!(saddle(1.5), 1);
    }

    #[test]
    fn box_blur_matches_a_direct_sum() {
        let (w, h, r) = (7, 5, 2);
        let values: Vec<f64> = (0..w * h).map(|i| ((i * 37) % 11) as f64).collect();
        let blurred = blur(&values, w, h, r);
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0;
                for dy in -(r as i64)..=r as i64 {
                    for dx in -(r as i64)..=r as i64 {
                        let (u, v) = (x as i64 + dx, y as i64 + dy);
                        if u >= 0 && v >= 0 && u < w as i64 && v < h as i64 {
                            sum += values[v as usize * w + u as usize];
                        }
                    }
                }
                assert!((blurred[y * w + x] - sum / ((2 * r + 1) * (2 * r + 1)) as f64).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn region_term_pulls_descent_onto_the_enclosed_area() {
        // An L shape: chamfer descent alone can hug its edges in a wrong
        // order; with the region term the enclosed area must match.
        let mut points = Vec::new();
        let corners: [(f64, f64); 6] = [(10.0, 10.0), (70.0, 10.0), (70.0, 30.0), (30.0, 30.0), (30.0, 90.0), (10.0, 90.0)];
        for k in 0..corners.len() {
            let (a, b) = (corners[k], corners[(k + 1) % corners.len()]);
            let steps = (b.0 - a.0).abs().max((b.1 - a.1).abs()) as usize;
            for i in 0..steps {
                let f = i as f64 / steps as f64;
                points.push(Complex::new(a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f));
            }
        }
        let (n, tree) = (points.len(), KdTree::new(&points));
        let m = (n / 3).clamp(2, MAX_HARMONICS);
        let s = (2 * n).max(4 * m + 2).next_power_of_two().max(64);
        let region = Region::new(&points, 1.0).unwrap();
        let d = dft(&points, m, s).unwrap();
        let mut refined = d.clone();
        let refined_loss = descend(&mut refined, m, s, &points, &tree, Some(&region), REFINE_STEPS).unwrap();
        let (c, loss) = from_circle(&points, m, s, &tree, Some(&region)).unwrap();
        let f = Fit { n, m, s, coefficients: vec![d, refined, Vec::new(), c], losses: vec![0.0, refined_loss, 0.0, loss], seconds: Vec::new() };
        let area = |c: &[Complex]| signed_area(&samples(c, f.m, f.s).unwrap());
        let truth = signed_area(&points);
        assert!(truth > 0.0);
        assert!((area(&f.coefficients[0]) - truth).abs() < 0.02 * truth);
        assert!((area(&f.coefficients[1]) - truth).abs() < 0.02 * truth);
        assert!((area(&f.coefficients[3]) - truth).abs() < 0.05 * truth, "{} vs {truth}", area(&f.coefficients[3]));
        assert!(f.losses[3] < 1.0, "{}", f.losses[3]);
    }
}

