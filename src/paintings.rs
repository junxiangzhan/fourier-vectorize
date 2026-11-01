//! Part 4 showcase: the vectorizer on two real paintings.
//!
//! Flat test shapes were separated by more than 10 px; in a painting,
//! regions of one colour touch and interleave, so unordered edge points can
//! no longer be split into outlines by DBSCAN (we measure how often it
//! merges them). Here every outline is therefore traced in order along the
//! pixel cracks, its Fourier coefficients come straight from a DFT, and
//! the rest is the same as before: corners, line / arc / Bézier fits,
//! nesting of holes, and refilling with our own rasterizer.
//!
//! Per painting: crop and box-average to a working size, quantize colours
//! by k-means in Lab, merge specks smaller than MIN_AREA pixels into their
//! neighbours, then vectorize one mask per colour.
use crate::{
    layers::{
        color_error, delta_e, hex, lab, paint, record, shapes, signed_area, svg_shape, write_rgb,
        Contour, Shape,
    },
    recover::{recover_with_params, sample, Primitive},
    vectorize::{edge_points, FourierCurve},
};
use multimedia::{
    bmp, cluster::dbscan, Color, Complex, Error, Image, Path as ShapePath, Rasterizer, Subpath,
};
use std::{
    collections::{HashMap, VecDeque},
    f64::consts::TAU,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

/// Without a given factor, a picture is shrunk by the smallest factor that
/// brings its longer side to at most this many pixels (The Dream by 2,
/// The Creation of Adam by 4).
const LONGEST: usize = 1000;
/// The names of RUNS, in order, as they end the file stems.
pub(crate) const METHOD_NAMES: [&str; 9] = [
    "k8",
    "k16",
    "k24",
    "k24_stack",
    "k24_stack_clean",
    "cmy888",
    "lab888",
    "lab1266",
    "lab1266_clean",
];
/// Regions smaller than this (pixels) are merged into a neighbour.
pub(crate) const MIN_AREA: usize = 16;
const KMEANS_ROUNDS: usize = 30;
/// Supersampling used when refilling the vector shapes.
const FILL_K: usize = 4;
/// A pixel differs visibly when its CIE76 colour difference exceeds this.
const VISIBLE_DELTA_E: f64 = 10.0;
/// Fallback when the fitted outline's area is off by more than this
/// fraction of the traced loop's area.
const FALLBACK_AREA: f64 = 0.15;
/// Douglas-Peucker tolerance (px) of the fallback polygon.
const FALLBACK_TOLERANCE: f64 = 0.8;
/// Cleaning: regions smaller than this (pixels) are merged into a similar
/// neighbour ...
const CLEAN_AREA: usize = 120;
/// ... where similar means palette colours closer than this (ΔE), or for
/// ink channels the adjacent level.
const CLEAN_DELTA_E: f64 = 12.0;
/// Bilateral smoothing before cleaned runs: radius (px), spatial sigma
/// (px), colour sigma (ΔE).
const BILATERAL: (usize, f64, f64) = (2, 2.0, 8.0);

// ---------------------------------------------------------- loading ----

/// Centre crop to a multiple of `factor`, then average each factor x factor
/// block (the same box filter as Part 3, per colour channel).
pub(crate) fn shrink(image: &Image<Color>, factor: usize) -> Result<Image<Color>, Error> {
    let (w, h) = (image.width() / factor, image.height() / factor);
    let (x0, y0) = (
        (image.width() - w * factor) / 2,
        (image.height() - h * factor) / 2,
    );
    let mut out = Image::new(w, h, Color::rgb(0.0, 0.0, 0.0))?;
    let n = (factor * factor) as f64;
    for y in 0..h {
        for x in 0..w {
            let mut sum = [0.0; 3];
            for sy in 0..factor {
                for sx in 0..factor {
                    let c = image
                        .get(x0 + x * factor + sx, y0 + y * factor + sy)
                        .unwrap();
                    sum[0] += c.r;
                    sum[1] += c.g;
                    sum[2] += c.b;
                }
            }
            *out.get_mut(x, y).unwrap() = Color::rgb(sum[0] / n, sum[1] / n, sum[2] / n);
        }
    }
    Ok(out)
}

// ------------------------------------------------------ quantizing ----

/// Small deterministic generator (64-bit LCG) so k-means++ is repeatable.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn dist2(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// k-means in Lab: k-means++ seeding (each new centre drawn with
/// probability proportional to squared distance), then Lloyd rounds until
/// no pixel changes cluster. Returns one label per pixel.
fn kmeans(points: &[[f64; 3]], k: usize, seed: u64) -> Vec<usize> {
    let mut rng = Lcg(seed);
    let mut centres = vec![points[(rng.next() * points.len() as f64) as usize % points.len()]];
    let mut nearest: Vec<f64> = points.iter().map(|p| dist2(p, &centres[0])).collect();
    while centres.len() < k {
        let total: f64 = nearest.iter().sum();
        let mut target = rng.next() * total;
        let mut chosen = points.len() - 1;
        for (i, d) in nearest.iter().enumerate() {
            if target < *d {
                chosen = i;
                break;
            }
            target -= d;
        }
        centres.push(points[chosen]);
        for (d, p) in nearest.iter_mut().zip(points) {
            *d = d.min(dist2(p, &centres[centres.len() - 1]));
        }
    }
    let assign = |centres: &[[f64; 3]]| -> Vec<usize> {
        points
            .iter()
            .map(|p| {
                (0..centres.len())
                    .min_by(|&a, &b| dist2(p, &centres[a]).total_cmp(&dist2(p, &centres[b])))
                    .unwrap()
            })
            .collect()
    };
    let mut labels = assign(&centres);
    for _ in 0..KMEANS_ROUNDS {
        let mut sums = vec![[0.0; 3]; k];
        let mut counts = vec![0usize; k];
        for (p, &l) in points.iter().zip(&labels) {
            for c in 0..3 {
                sums[l][c] += p[c];
            }
            counts[l] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                centres[c] = sums[c].map(|v| v / counts[c] as f64);
            }
        }
        let next = assign(&centres);
        if next == labels {
            break;
        }
        labels = next;
    }
    labels
}

/// 4-connected regions of equal label: (region id per pixel, sizes).
fn regions(labels: &[usize], w: usize, h: usize) -> (Vec<usize>, Vec<usize>) {
    let mut id = vec![usize::MAX; labels.len()];
    let mut sizes = Vec::new();
    for start in 0..labels.len() {
        if id[start] != usize::MAX {
            continue;
        }
        let region = sizes.len();
        let mut queue = VecDeque::from([start]);
        id[start] = region;
        let mut size = 0;
        while let Some(i) = queue.pop_front() {
            size += 1;
            let (x, y) = (i % w, i / w);
            let mut visit = |j: usize| {
                if id[j] == usize::MAX && labels[j] == labels[i] {
                    id[j] = region;
                    queue.push_back(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < w {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - w);
            }
            if y + 1 < h {
                visit(i + w);
            }
        }
        sizes.push(size);
    }
    (id, sizes)
}

/// Merges every region smaller than `min_area` into the allowed neighbouring
/// label it shares the longest border with (`allowed(own, other)`); regions
/// with no allowed neighbour stay. Regions are merged one at a time,
/// smallest first; a region whose neighbour already changed in this pass
/// waits for the next pass. Updating all regions at once could let two
/// small neighbours swap labels forever; here every pass either merges a
/// region (so the region count falls) or ends the loop.
fn merge_small(
    labels: &mut [usize],
    w: usize,
    h: usize,
    min_area: usize,
    allowed: &dyn Fn(usize, usize) -> bool,
) {
    loop {
        let (id, sizes) = regions(labels, w, h);
        let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, r) in id.iter().enumerate() {
            if sizes[*r] < min_area {
                members.entry(*r).or_default().push(i);
            }
        }
        let mut order: Vec<usize> = members.keys().copied().collect();
        order.sort_by_key(|r| (sizes[*r], *r));
        let mut touched = std::collections::HashSet::new();
        let mut merged = 0;
        for region in order {
            if touched.contains(&region) {
                continue;
            }
            let own = labels[members[&region][0]];
            let mut border: HashMap<usize, usize> = HashMap::new();
            let mut neighbours = std::collections::HashSet::new();
            for &i in &members[&region] {
                let (x, y) = (i % w, i / w);
                let around = [
                    (x > 0).then(|| i - 1),
                    (x + 1 < w).then(|| i + 1),
                    (y > 0).then(|| i - w),
                    (y + 1 < h).then(|| i + w),
                ];
                for j in around.into_iter().flatten() {
                    if id[j] != region && allowed(own, labels[j]) {
                        *border.entry(labels[j]).or_default() += 1;
                        neighbours.insert(id[j]);
                    }
                }
            }
            let Some((label, _)) = border
                .into_iter()
                .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
            else {
                continue;
            };
            for &i in &members[&region] {
                labels[i] = label;
            }
            merged += 1;
            touched.insert(region);
            touched.extend(neighbours);
        }
        if merged == 0 {
            return;
        }
    }
}

/// Merges every region smaller than MIN_AREA into any neighbour.
pub(crate) fn despeckle(labels: &mut [usize], w: usize, h: usize) {
    merge_small(labels, w, h, MIN_AREA, &|_, _| true);
}

// --------------------------------------------------------- tracing ----

/// Ordered boundary loops of a binary mask, following pixel cracks with
/// the inside on the right: outer boundaries come out screen-clockwise and
/// holes counter-clockwise. Where two inside pixels touch only at a corner,
/// the trace turns right, keeping them apart (4-connectivity). Each loop is
/// the list of crack midpoints, i.e. exactly the edge points used for the
/// test scenes, now in order and one pixel apart.
pub(crate) fn loops(inside: &[bool], w: usize, h: usize) -> Vec<Vec<Complex>> {
    let at = |x: i64, y: i64| {
        x >= 0 && y >= 0 && x < w as i64 && y < h as i64 && inside[(y * w as i64 + x) as usize]
    };
    // Directions: 0 right, 1 down, 2 left, 3 up (screen coordinates).
    let step = [(1i64, 0i64), (0, 1), (-1, 0), (0, -1)];
    let mut edges: Vec<(i64, i64, usize)> = Vec::new();
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            if !at(x, y) {
                continue;
            }
            // Clockwise around the pixel, only where the neighbour is outside.
            if !at(x, y - 1) {
                edges.push((x, y, 0));
            }
            if !at(x + 1, y) {
                edges.push((x + 1, y, 1));
            }
            if !at(x, y + 1) {
                edges.push((x + 1, y + 1, 2));
            }
            if !at(x - 1, y) {
                edges.push((x, y + 1, 3));
            }
        }
    }
    let mut leaving: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, e) in edges.iter().enumerate() {
        leaving.entry((e.0, e.1)).or_default().push(i);
    }
    let mut used = vec![false; edges.len()];
    let mut result = Vec::new();
    for first in 0..edges.len() {
        if used[first] {
            continue;
        }
        let mut lp = Vec::new();
        let mut current = first;
        loop {
            used[current] = true;
            let (x, y, d) = edges[current];
            lp.push(Complex::new(
                x as f64 + 0.5 * step[d].0 as f64,
                y as f64 + 0.5 * step[d].1 as f64,
            ));
            let end = (x + step[d].0, y + step[d].1);
            // Prefer a right turn, then straight on, then a left turn.
            let next = [(d + 1) % 4, d, (d + 3) % 4].into_iter().find_map(|want| {
                leaving
                    .get(&end)
                    .into_iter()
                    .flatten()
                    .copied()
                    .find(|&j| !used[j] && edges[j].2 == want)
            });
            match next {
                Some(j) => current = j,
                None => break,
            }
        }
        result.push(lp);
    }
    result
}

/// Even-odd ray test of `p` against a closed polygon.
fn inside_polygon(p: Complex, polygon: &[Complex]) -> bool {
    let mut inside = false;
    for (i, a) in polygon.iter().enumerate() {
        let b = polygon[(i + 1) % polygon.len()];
        if (a.im > p.im) != (b.im > p.im)
            && p.re < a.re + (p.im - a.im) * (b.re - a.re) / (b.im - a.im)
        {
            inside = !inside;
        }
    }
    inside
}

/// Depth and parent of every traced loop, tested on the exact crack loops
/// rather than the fitted outlines. Loops of one mask never share a crack,
/// so a loop's first crack midpoint never lies on another loop; the fitted
/// outlines, up to 0.8 px off, can cross where loops touch at a pixel
/// corner, and once made a large hole look like it sat inside a tiny one.
fn nest_loops(traced: &[Vec<Complex>]) -> Vec<(usize, Option<usize>)> {
    let bounds: Vec<(f64, f64, f64, f64)> = traced
        .iter()
        .map(|l| {
            l.iter().fold(
                (
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ),
                |b, z| (b.0.min(z.re), b.1.min(z.im), b.2.max(z.re), b.3.max(z.im)),
            )
        })
        .collect();
    let areas: Vec<f64> = traced.iter().map(|l| signed_area(l).abs()).collect();
    traced
        .iter()
        .enumerate()
        .map(|(i, lp)| {
            let q = lp[0];
            let containers: Vec<usize> = (0..traced.len())
                .filter(|&j| {
                    let b = bounds[j];
                    j != i
                        && q.re > b.0
                        && q.re < b.2
                        && q.im > b.1
                        && q.im < b.3
                        && inside_polygon(q, &traced[j])
                })
                .collect();
            let parent = containers
                .iter()
                .copied()
                .min_by(|&a, &b| areas[a].total_cmp(&areas[b]));
            (containers.len(), parent)
        })
        .collect()
}

/// Fourier coefficients of an ordered closed loop by direct DFT:
/// c_k = (1/n) sum_j z_j e^(-2 pi i k j / n), |k| <= harmonics.
fn dft(points: &[Complex], harmonics: usize) -> FourierCurve {
    let n = points.len() as f64;
    let m = harmonics as i64;
    let coefficients = (-m..=m)
        .map(|k| {
            points
                .iter()
                .enumerate()
                .fold(Complex::ZERO, |sum, (j, z)| {
                    sum + *z * Complex::from_angle(-TAU * k as f64 * j as f64 / n)
                })
                .scale(1.0 / n)
        })
        .collect();
    FourierCurve {
        harmonics,
        coefficients,
    }
}

/// Douglas-Peucker simplification of a closed loop: keep the point farthest
/// from the chord whenever it is more than `tolerance` away, recursively.
fn simplify(points: &[Complex], tolerance: f64) -> Vec<Complex> {
    fn split(points: &[Complex], a: usize, b: usize, tolerance: f64, keep: &mut Vec<usize>) {
        let (p, q) = (points[a], points[b]);
        let d = q - p;
        let far = (a + 1..b)
            .map(|i| {
                let e = points[i] - p;
                let dist = if d.abs() > 0.0 {
                    (d.re * e.im - d.im * e.re).abs() / d.abs()
                } else {
                    e.abs()
                };
                (dist, i)
            })
            .fold((0.0, a), |m, x| if x.0 > m.0 { x } else { m });
        if far.0 > tolerance {
            split(points, a, far.1, tolerance, keep);
            keep.push(far.1);
            split(points, far.1, b, tolerance, keep);
        }
    }
    // Close the loop by repeating the first point, and start the recursion
    // from the point farthest from it.
    let mut ring = points.to_vec();
    ring.push(points[0]);
    let far = (0..points.len())
        .max_by(|&i, &j| {
            (points[i] - points[0])
                .abs()
                .total_cmp(&(points[j] - points[0]).abs())
        })
        .unwrap();
    let mut keep = vec![0];
    split(&ring, 0, far, tolerance, &mut keep);
    keep.push(far);
    split(&ring, far, points.len(), tolerance, &mut keep);
    keep.into_iter().map(|i| ring[i]).collect()
}

/// One ordered loop to primitives. Harmonics ~ n/6 keep wavelengths of 6+
/// px, smoothing the one-pixel staircase while following the shape.
/// Returns whether the polygon fallback was needed.
fn contour(points: &[Complex], index: usize) -> (Contour, bool) {
    let n = points.len();
    let harmonics = (n / 6).clamp(4, 160);
    let curve = dft(points, harmonics);
    let count = (4 * n).max(256);
    let dense = curve.sample(count);
    // Point j sits at parameter j/n on the curve.
    let params: Vec<usize> = (0..n).map(|j| (j * count + n / 2) / n % count).collect();
    let mut primitives = recover_with_params(&dense, points, &params).primitives;
    let mut outline: Vec<Complex> = primitives.iter().flat_map(sample).collect();
    // Safety check: the recovered outline must enclose about the same area,
    // with the same orientation, as the traced pixel loop. Thin strokes
    // (narrower than the corner scale) and tangled outlines fail it; they
    // fall back to a Douglas-Peucker polygon of the traced loop.
    let traced = signed_area(points);
    let fitted = signed_area(&outline);
    let fallback = (fitted - traced).abs() > FALLBACK_AREA * traced.abs();
    if fallback {
        let corners = simplify(points, FALLBACK_TOLERANCE);
        primitives = (0..corners.len())
            .map(|i| Primitive::Line(corners[i], corners[(i + 1) % corners.len()]))
            .collect();
        outline = primitives.iter().flat_map(sample).collect();
    }
    (
        Contour {
            cluster: index,
            edges: n,
            area: signed_area(&outline),
            primitives,
            outline,
        },
        fallback,
    )
}

// ------------------------------------------------------------- run ----

struct Stats {
    loops: usize,
    shapes: usize,
    holes: usize,
    lines: usize,
    arcs: usize,
    cubics: usize,
    dbscan_clusters: usize,
    fallbacks: usize,
}

/// How a run turns the picture into layers to vectorize.
#[derive(Clone, Copy)]
enum Method {
    /// k-means palette of `colours` colours. Cut-out: one mask per colour,
    /// painted on the largest colour, so neighbouring colours meet along
    /// two separately fitted edges and the background can show between
    /// them. Stacked: colours ordered light to dark, layer i covers every
    /// pixel whose colour is i or darker, so each layer runs under the ones
    /// above and no gap can open.
    Palette { colours: usize, stacked: bool },
    /// One plate per channel, each cut into nested level masks.
    Channels { space: Space, levels: [usize; 3] },
}

#[derive(Clone, Copy, PartialEq)]
enum Space {
    /// Cyan, magenta, yellow ink; plates multiplied as in printing.
    Cmy,
    /// CIE L*, a*, b*: lightness and two colour-difference channels,
    /// recombined by converting back to RGB.
    Lab,
}

/// The runs per painting (method, cleanup). Palette 24, CMY 8+8+8 and Lab
/// 8+8+8 / 12+6+6 all use 24 layers.
const RUNS: [(Method, bool); 9] = [
    (
        Method::Palette {
            colours: 8,
            stacked: false,
        },
        false,
    ),
    (
        Method::Palette {
            colours: 16,
            stacked: false,
        },
        false,
    ),
    (
        Method::Palette {
            colours: 24,
            stacked: false,
        },
        false,
    ),
    (
        Method::Palette {
            colours: 24,
            stacked: true,
        },
        false,
    ),
    (
        Method::Palette {
            colours: 24,
            stacked: true,
        },
        true,
    ),
    (
        Method::Channels {
            space: Space::Cmy,
            levels: [8, 8, 8],
        },
        false,
    ),
    (
        Method::Channels {
            space: Space::Lab,
            levels: [8, 8, 8],
        },
        false,
    ),
    (
        Method::Channels {
            space: Space::Lab,
            levels: [12, 6, 6],
        },
        false,
    ),
    (
        Method::Channels {
            space: Space::Lab,
            levels: [12, 6, 6],
        },
        true,
    ),
];

/// The browser-viewable Lab SVG. The three plates are drawn into the red,
/// green and blue channels (r = L*/100, g = (a* + 128)/255,
/// b = (b* + 128)/255) and combined with `screen`, which is exact when
/// each plate lights only its own channel. An SVG filter then rebuilds RGB
/// in four primitives: a matrix to f(X), f(Y), f(Z); a per-channel cube; a
/// matrix from XYZ to linear RGB; the sRGB gamma curve. SVG filters clamp
/// every result to [0, 1] and have no piecewise functions, so the short
/// linear pieces of the Lab and sRGB curves near black are left out: that
/// approximation is measured, not assumed.
const LAB_TO_F: [[f64; 4]; 3] = [
    // f(X) = f(Y) + a*/500, f(Y) = (L* + 16)/116, f(Z) = f(Y) - b*/200,
    // written in the encoded channels: [r, g, b, constant].
    [
        100.0 / 116.0,
        255.0 / 500.0,
        0.0,
        16.0 / 116.0 - 128.0 / 500.0,
    ],
    [100.0 / 116.0, 0.0, 0.0, 16.0 / 116.0],
    [
        100.0 / 116.0,
        0.0,
        -255.0 / 200.0,
        16.0 / 116.0 + 128.0 / 200.0,
    ],
];
/// Normalized XYZ (X/Xn, Y/Yn, Z/Zn) to linear sRGB, D65.
const XYZ_TO_RGB: [[f64; 3]; 3] = [
    [3.2406 * 0.95047, -1.5372, -0.4986 * 1.08883],
    [-0.9689 * 0.95047, 1.8758, 0.0415 * 1.08883],
    [0.0557 * 0.95047, -0.2040, 1.0570 * 1.08883],
];

/// The colour a browser computes from the plate values through the filter,
/// primitive by primitive, with the same clamping. With `eight_bit`, every
/// intermediate result is also rounded to 1/255, as when a renderer keeps
/// 8-bit buffers between filter primitives.
fn svg_filter_colour(l: f64, a: f64, b: f64, eight_bit: bool) -> Color {
    let clamp = |v: f64| {
        let v = v.clamp(0.0, 1.0);
        if eight_bit {
            (v * 255.0).round() / 255.0
        } else {
            v
        }
    };
    let encoded = [
        clamp(l / 100.0),
        clamp((a + 128.0) / 255.0),
        clamp((b + 128.0) / 255.0),
    ];
    let f: Vec<f64> = LAB_TO_F
        .iter()
        .map(|row| clamp(row[0] * encoded[0] + row[1] * encoded[1] + row[2] * encoded[2] + row[3]))
        .collect();
    let xyz: Vec<f64> = f.iter().map(|v| clamp(v * v * v)).collect();
    let rgb: Vec<f64> = XYZ_TO_RGB
        .iter()
        .map(|row| {
            let linear = clamp(row[0] * xyz[0] + row[1] * xyz[1] + row[2] * xyz[2]);
            clamp(1.055 * linear.powf(1.0 / 2.4) - 0.055)
        })
        .collect();
    Color::rgb(rgb[0], rgb[1], rgb[2])
}

/// The <filter> element applying `svg_filter_colour`'s steps.
pub(crate) fn svg_lab_filter(w: usize, h: usize) -> String {
    let m1: Vec<String> = LAB_TO_F
        .iter()
        .map(|r| format!("{:.6} {:.6} {:.6} 0 {:.6}", r[0], r[1], r[2], r[3]))
        .collect();
    let m2: Vec<String> = XYZ_TO_RGB
        .iter()
        .map(|r| format!("{:.6} {:.6} {:.6} 0 0", r[0], r[1], r[2]))
        .collect();
    let funcs = |attributes: &str| {
        ["R", "G", "B"]
            .iter()
            .map(|c| format!("<feFunc{c} type=\"gamma\" {attributes}/>"))
            .collect::<String>()
    };
    format!(
        "  <defs><filter id=\"lab\" filterUnits=\"userSpaceOnUse\" x=\"0\" y=\"0\" width=\"{w}\" height=\"{h}\" color-interpolation-filters=\"sRGB\">\n    <feColorMatrix type=\"matrix\" values=\"{} 0 0 0 1 0\"/>\n    <feComponentTransfer>{}</feComponentTransfer>\n    <feColorMatrix type=\"matrix\" values=\"{} 0 0 0 1 0\"/>\n    <feComponentTransfer>{}</feComponentTransfer>\n  </filter></defs>\n",
        m1.join(" "),
        funcs("amplitude=\"1\" exponent=\"3\" offset=\"0\""),
        m2.join(" "),
        funcs("amplitude=\"1.055\" exponent=\"0.416667\" offset=\"-0.055\"")
    )
}

/// CIE L*a*b* (D65) back to sRGB in [0, 1]: the inverse of `lab`.
pub(crate) fn lab_to_rgb(l: f64, a: f64, b: f64) -> Color {
    let fy = (l + 16.0) / 116.0;
    let (fx, fz) = (fy + a / 500.0, fy - b / 200.0);
    let finv = |t: f64| {
        if t * t * t > 0.008856 {
            t * t * t
        } else {
            (t - 16.0 / 116.0) / 7.787
        }
    };
    let (x, y, z) = (0.95047 * finv(fx), finv(fy), 1.08883 * finv(fz));
    let gamma = |v: f64| {
        let v = if v <= 0.0031308 {
            12.92 * v
        } else {
            1.055 * v.max(0.0).powf(1.0 / 2.4) - 0.055
        };
        v.clamp(0.0, 1.0)
    };
    Color::rgb(
        gamma(3.2406 * x - 1.5372 * y - 0.4986 * z),
        gamma(-0.9689 * x + 1.8758 * y + 0.0415 * z),
        gamma(0.0557 * x - 0.2040 * y + 1.0570 * z),
    )
}

/// Edge-preserving smoothing: each pixel becomes a mean of its neighbours
/// within BILATERAL.0 px, weighted by distance (sigma BILATERAL.1 px) and by
/// colour difference (sigma BILATERAL.2 in ΔE). Cracks and plaster speckle
/// fade; edges between clearly different colours stay sharp.
pub(crate) fn bilateral(input: &Image<Color>) -> Result<Image<Color>, Error> {
    let (w, h) = (input.width(), input.height());
    let (radius, sigma_s, sigma_r) = BILATERAL;
    let labs: Vec<[f64; 3]> = input.pixels().iter().map(|c| lab(*c)).collect();
    let pixels = input.pixels();
    let r = radius as i64;
    let mut out = Image::new(w, h, Color::rgb(0.0, 0.0, 0.0))?;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let (mut sum, mut total) = ([0.0; 3], 0.0);
            for dy in -r..=r {
                for dx in -r..=r {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    let weight = (-((dx * dx + dy * dy) as f64) / (2.0 * sigma_s * sigma_s)).exp()
                        * (-dist2(&labs[i], &labs[j]) / (2.0 * sigma_r * sigma_r)).exp();
                    sum[0] += weight * pixels[j].r;
                    sum[1] += weight * pixels[j].g;
                    sum[2] += weight * pixels[j].b;
                    total += weight;
                }
            }
            *out.get_mut(x, y).unwrap() =
                Color::rgb(sum[0] / total, sum[1] / total, sum[2] / total);
        }
    }
    Ok(out)
}

/// Mean colour of each label, relabelled by decreasing area (label 0 is
/// the largest region set, used as background); empty labels disappear.
fn palette_of(source: &Image<Color>, labels: &mut [usize]) -> Vec<(Color, usize)> {
    let mut sums: HashMap<usize, ([f64; 3], usize)> = HashMap::new();
    for (c, &l) in source.pixels().iter().zip(labels.iter()) {
        let entry = sums.entry(l).or_insert(([0.0; 3], 0));
        entry.0[0] += c.r;
        entry.0[1] += c.g;
        entry.0[2] += c.b;
        entry.1 += 1;
    }
    let mut used: Vec<usize> = sums.keys().copied().collect();
    used.sort_by(|a, b| sums[b].1.cmp(&sums[a].1).then(a.cmp(b)));
    let remap: HashMap<usize, usize> = used.iter().enumerate().map(|(i, l)| (*l, i)).collect();
    for l in labels.iter_mut() {
        *l = remap[l];
    }
    used.iter()
        .map(|l| {
            let (s, n) = sums[l];
            (
                Color::rgb(s[0] / n as f64, s[1] / n as f64, s[2] / n as f64),
                n,
            )
        })
        .collect()
}

/// 1-D k-means of one ink channel into `levels` values, seeded at evenly
/// spaced quantiles. Labels are ranked by ink, 0 = least ink.
pub(crate) fn kmeans_1d(values: &[f64], levels: usize) -> Vec<usize> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut centres: Vec<f64> = (0..levels)
        .map(|i| sorted[(2 * i + 1) * sorted.len() / (2 * levels)])
        .collect();
    let assign = |centres: &[f64]| -> Vec<usize> {
        values
            .iter()
            .map(|v| {
                (0..centres.len())
                    .min_by(|&a, &b| (v - centres[a]).abs().total_cmp(&(v - centres[b]).abs()))
                    .unwrap()
            })
            .collect()
    };
    let mut labels = assign(&centres);
    for _ in 0..KMEANS_ROUNDS {
        let mut sums = vec![(0.0, 0usize); levels];
        for (v, &l) in values.iter().zip(&labels) {
            sums[l].0 += v;
            sums[l].1 += 1;
        }
        for (c, (s, n)) in centres.iter_mut().zip(&sums) {
            if *n > 0 {
                *c = s / *n as f64;
            }
        }
        let next = assign(&centres);
        if next == labels {
            break;
        }
        labels = next;
    }
    let mut order: Vec<usize> = (0..levels).collect();
    order.sort_by(|&a, &b| centres[a].total_cmp(&centres[b]).then(a.cmp(&b)));
    let mut rank = vec![0; levels];
    for (r, &l) in order.iter().enumerate() {
        rank[l] = r;
    }
    labels.iter().map(|l| rank[*l]).collect()
}

/// Mean ink of each level label, dropping levels left empty and keeping
/// the order (0 = least ink).
pub(crate) fn levels_of(values: &[f64], labels: &mut [usize]) -> Vec<f64> {
    let top = labels.iter().copied().max().unwrap_or(0);
    let mut sums = vec![(0.0, 0usize); top + 1];
    for (v, &l) in values.iter().zip(labels.iter()) {
        sums[l].0 += v;
        sums[l].1 += 1;
    }
    let kept: Vec<usize> = (0..=top).filter(|&l| sums[l].1 > 0).collect();
    let mut rank = vec![0; top + 1];
    for (r, &l) in kept.iter().enumerate() {
        rank[l] = r;
    }
    for l in labels.iter_mut() {
        *l = rank[*l];
    }
    kept.iter().map(|&l| sums[l].0 / sums[l].1 as f64).collect()
}

/// Traces masks into shapes, keeps the CSV records and the statistics.
struct Tracer {
    w: usize,
    h: usize,
    raster: Rasterizer,
    contours_file: BufWriter<File>,
    primitives_file: BufWriter<File>,
    stats: Stats,
}
impl Tracer {
    fn new(output: &Path, stem: &str, w: usize, h: usize) -> Result<Self, Error> {
        let mut contours_file =
            BufWriter::new(File::create(output.join(format!("{stem}_contours.csv")))?);
        writeln!(
            contours_file,
            "layer,contour,cluster,edges,depth,parent,area,reversed,lines,arcs,cubics"
        )?;
        let mut primitives_file =
            BufWriter::new(File::create(output.join(format!("{stem}_primitives.csv")))?);
        writeln!(
            primitives_file,
            "layer,contour,index,kind,x0,y0,x1,y1,cx,cy,radius,sweep,q1x,q1y,q2x,q2y"
        )?;
        Ok(Self {
            w,
            h,
            raster: Rasterizer::new(w, h)?,
            contours_file,
            primitives_file,
            stats: Stats {
                loops: 0,
                shapes: 0,
                holes: 0,
                lines: 0,
                arcs: 0,
                cubics: 0,
                dbscan_clusters: 0,
                fallbacks: 0,
            },
        })
    }

    /// One mask: trace, fit, nest, record; returns its shapes and their
    /// coverage filled with our rasterizer (None if the mask is empty).
    fn layer(
        &mut self,
        inside: &[bool],
        name: &str,
    ) -> Result<(Vec<Shape>, Option<Image<f64>>), Error> {
        let (w, h) = (self.w, self.h);
        let traced = loops(inside, w, h);
        // How DBSCAN on the same, unordered points would have grouped them.
        let mut mask = Image::new(w, h, 0.0)?;
        for (i, v) in inside.iter().enumerate() {
            *mask.get_mut(i % w, i / w).unwrap() = if *v { 1.0 } else { 0.0 };
        }
        self.stats.dbscan_clusters += dbscan(&edge_points(&mask), 3.0, 3)?
            .iter()
            .flatten()
            .max()
            .map_or(0, |m| m + 1);
        let mut contours = Vec::new();
        for (i, lp) in traced.iter().enumerate() {
            let (c, fallback) = contour(lp, i);
            self.stats.fallbacks += usize::from(fallback);
            contours.push(c);
        }
        let nesting = nest_loops(&traced);
        let (found, reversed) = shapes(&contours, &nesting);
        record(
            &mut self.contours_file,
            &mut self.primitives_file,
            name,
            &contours,
            &nesting,
            &reversed,
        )?;
        self.stats.loops += contours.len();
        self.stats.shapes += found.len();
        self.stats.holes += found.iter().map(|s| s.holes.len()).sum::<usize>();
        for c in &contours {
            for p in &c.primitives {
                match crate::recover::kind(p) {
                    "line" => self.stats.lines += 1,
                    "arc" => self.stats.arcs += 1,
                    _ => self.stats.cubics += 1,
                }
            }
        }
        let subpaths: Vec<Subpath> = found
            .iter()
            .flat_map(|s| s.subpaths.iter().cloned())
            .collect();
        let coverage = if subpaths.is_empty() {
            None
        } else {
            Some(
                self.raster
                    .render(&ShapePath::new(subpaths).flatten(5)?, FILL_K)?,
            )
        };
        Ok((found, coverage))
    }
}

/// Structural similarity of lightness L*: mean SSIM over non-overlapping
/// 8 x 8 blocks (C1 = (0.01 * 100)^2, C2 = (0.03 * 100)^2 on the 0..100 L*
/// scale; NaN below 8 x 8 px). Mean ΔE scores each pixel's colour alone; SSIM asks whether
/// local light and shade, i.e. the structure, survived.
pub(crate) fn ssim(a: &Image<Color>, b: &Image<Color>) -> f64 {
    let (w, h) = (a.width(), a.height());
    let la: Vec<f64> = a.pixels().iter().map(|c| lab(*c)[0]).collect();
    let lb: Vec<f64> = b.pixels().iter().map(|c| lab(*c)[0]).collect();
    let (c1, c2) = (1.0, 9.0);
    let (mut total, mut blocks) = (0.0, 0usize);
    for by in 0..h / 8 {
        for bx in 0..w / 8 {
            let index = |k: usize| (by * 8 + k / 8) * w + bx * 8 + k % 8;
            let n = 64.0;
            let ma = (0..64).map(|k| la[index(k)]).sum::<f64>() / n;
            let mb = (0..64).map(|k| lb[index(k)]).sum::<f64>() / n;
            let (mut va, mut vb, mut cov) = (0.0, 0.0, 0.0);
            for k in 0..64 {
                let (x, y) = (la[index(k)] - ma, lb[index(k)] - mb);
                va += x * x;
                vb += y * y;
                cov += x * y;
            }
            let (va, vb, cov) = (va / n, vb / n, cov / n);
            total += ((2.0 * ma * mb + c1) * (2.0 * cov + c2))
                / ((ma * ma + mb * mb + c1) * (va + vb + c2));
            blocks += 1;
        }
    }
    // Smaller than one block: no structure to compare.
    if blocks == 0 {
        return f64::NAN;
    }
    total / blocks as f64
}

fn vectorize_run(
    output: &Path,
    name: &str,
    original: &Image<Color>,
    smoothed: &Image<Color>,
    method: Method,
    clean: bool,
) -> Result<String, Error> {
    let started = Instant::now();
    let (w, h) = (original.width(), original.height());
    // Cleaning starts from the smoothed picture; errors are always measured
    // against the original.
    let source = if clean { smoothed } else { original };
    let stem = match method {
        Method::Palette { colours, stacked } => {
            format!(
                "paint_{name}_k{colours}{}",
                if stacked { "_stack" } else { "" }
            )
        }
        Method::Channels { space, levels } => format!(
            "paint_{name}_{}{}{}{}",
            if space == Space::Cmy { "cmy" } else { "lab" },
            levels[0],
            levels[1],
            levels[2]
        ),
    } + if clean { "_clean" } else { "" };
    let mut tracer = Tracer::new(output, &stem, w, h)?;
    let mut quantized = Image::new(w, h, Color::rgb(0.0, 0.0, 0.0))?;
    let rebuilt;
    let mut svg = format!("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\">\n");
    let method_json;
    // Gap measure for palette runs: pixels whose refilled colour is nearest
    // to a different palette colour than their own, and those where it is
    // the canvas colour showing through.
    let mut gaps = String::from("\"misassigned\":null,\"background_exposed\":null");
    match method {
        Method::Palette { colours, stacked } => {
            // k-means in Lab, then merge specks (and, cleaning, small regions
            // into a neighbour of similar colour).
            let points: Vec<[f64; 3]> = source.pixels().iter().map(|c| lab(*c)).collect();
            let mut labels = kmeans(&points, colours, 1151);
            despeckle(&mut labels, w, h);
            if clean {
                let first = palette_of(source, &mut labels);
                merge_small(&mut labels, w, h, CLEAN_AREA, &|a, b| {
                    delta_e(first[a].0, first[b].0) < CLEAN_DELTA_E
                });
            }
            let mut palette = palette_of(source, &mut labels);
            if stacked {
                // Reorder light to dark: label 0 (the canvas) is the lightest.
                let mut order: Vec<usize> = (0..palette.len()).collect();
                order.sort_by(|&a, &b| {
                    lab(palette[b].0)[0]
                        .total_cmp(&lab(palette[a].0)[0])
                        .then(a.cmp(&b))
                });
                let mut rank = vec![0; palette.len()];
                for (r, &l) in order.iter().enumerate() {
                    rank[l] = r;
                }
                for l in labels.iter_mut() {
                    *l = rank[*l];
                }
                palette = order.iter().map(|&l| palette[l]).collect();
            }
            for (i, &l) in labels.iter().enumerate() {
                *quantized.get_mut(i % w, i / w).unwrap() = palette[l].0;
            }
            let mut file = BufWriter::new(File::create(output.join(format!("{stem}_labels.csv")))?);
            writeln!(file, "x,y,label")?;
            for (i, l) in labels.iter().enumerate() {
                writeln!(file, "{},{},{l}", i % w, i / w)?;
            }
            file.flush()?;
            // Colour 0 is the canvas; the rest are painted on top in order.
            let mut image = Image::new(w, h, palette[0].0)?;
            svg += &format!(
                "  <rect width=\"{w}\" height=\"{h}\" fill=\"{}\"/>\n",
                hex(palette[0].0)
            );
            for (colour, &(fill, _)) in palette.iter().enumerate().skip(1) {
                let inside: Vec<bool> = labels
                    .iter()
                    .map(|&l| if stacked { l >= colour } else { l == colour })
                    .collect();
                let (found, coverage) = tracer.layer(&inside, &format!("colour{colour}"))?;
                if let Some(coverage) = coverage {
                    paint(&mut image, &coverage, fill);
                }
                for shape in &found {
                    svg += &svg_shape(shape, fill);
                }
            }
            let nearest = |c: Color| {
                (0..palette.len())
                    .min_by(|&a, &b| delta_e(c, palette[a].0).total_cmp(&delta_e(c, palette[b].0)))
                    .unwrap()
            };
            let (mut wrong, mut exposed) = (0, 0);
            for (c, &l) in image.pixels().iter().zip(&labels) {
                let n = nearest(*c);
                wrong += usize::from(n != l);
                exposed += usize::from(n == 0 && l != 0);
            }
            gaps = format!("\"misassigned\":{wrong},\"background_exposed\":{exposed}");
            rebuilt = image;
            let entries: Vec<String> = palette
                .iter()
                .map(|(c, n)| {
                    format!(
                        "{{\"rgb\":[{:.12},{:.12},{:.12}],\"pixels\":{n}}}",
                        c.r, c.g, c.b
                    )
                })
                .collect();
            method_json = format!(
                "\"method\":\"palette\",\"colours\":{colours},\"stacked\":{stacked},\"order\":\"{}\",\"layers\":{},\"palette\":[{}]",
                if stacked { "lightness" } else { "area" },
                palette.len(),
                entries.join(",")
            );
        }
        Method::Channels { space, levels } => {
            let names = if space == Space::Cmy {
                ["cyan", "magenta", "yellow"]
            } else {
                ["lightness", "a", "b"]
            };
            let labs: Vec<[f64; 3]> = source.pixels().iter().map(|c| lab(*c)).collect();
            let mut plates = Vec::new();
            let mut quantized_values = Vec::new();
            let mut level_json = Vec::new();
            let mut label_sets = Vec::new();
            let mut layers = 0;
            // Browser-viewable version (Lab only): plates as colour channels.
            let mut browser = String::new();
            if space == Space::Lab {
                svg += "  <!-- Lab plates: grey = L*/100 for lightness, (v + 128)/255 for a* and b*. Standard SVG blend modes work in RGB and cannot recombine Lab; this program converts the three plates back to RGB itself. -->\n";
            }
            for (channel, plate) in names.iter().enumerate() {
                let values: Vec<f64> = match space {
                    Space::Cmy => source
                        .pixels()
                        .iter()
                        .map(|c| 1.0 - [c.r, c.g, c.b][channel])
                        .collect(),
                    Space::Lab => labs.iter().map(|v| v[channel]).collect(),
                };
                let mut labels = kmeans_1d(&values, levels[channel]);
                despeckle(&mut labels, w, h);
                if clean {
                    // Small regions may only step to an adjacent level.
                    merge_small(&mut labels, w, h, CLEAN_AREA, &|a, b| a.abs_diff(b) == 1);
                }
                let level = levels_of(&values, &mut labels);
                layers += level.len();
                level_json.push(format!(
                    "\"{plate}\":[{}]",
                    level
                        .iter()
                        .map(|v| format!("{v:.12}"))
                        .collect::<Vec<_>>()
                        .join(",")
                ));
                // How each level is drawn in the SVG.
                let fill = |v: f64| match space {
                    Space::Cmy => {
                        let mut c = [1.0; 3];
                        c[channel] = 1.0 - v;
                        Color::rgb(c[0], c[1], c[2])
                    }
                    Space::Lab => {
                        let g = if channel == 0 {
                            v / 100.0
                        } else {
                            (v + 128.0) / 255.0
                        };
                        Color::rgb(g, g, g)
                    }
                };
                svg += &match space {
                    Space::Cmy => {
                        String::from("  <g style=\"mix-blend-mode:multiply;isolation:isolate\">\n")
                    }
                    Space::Lab => format!("  <g id=\"{plate}\">\n"),
                };
                svg += &format!(
                    "    <rect width=\"{w}\" height=\"{h}\" fill=\"{}\"/>\n",
                    hex(fill(level[0]))
                );
                // In the browser SVG each plate lights only its own channel.
                let encode = |v: f64| {
                    let mut c = [0.0; 3];
                    c[channel] = if channel == 0 {
                        v / 100.0
                    } else {
                        (v + 128.0) / 255.0
                    };
                    Color::rgb(c[0], c[1], c[2])
                };
                browser += &format!(
                    "    <g style=\"mix-blend-mode:screen;isolation:isolate\">\n    <rect width=\"{w}\" height=\"{h}\" fill=\"{}\"/>\n",
                    hex(encode(level[0]))
                );
                // Nested masks of increasing level, each painted over the last.
                let mut value_image = Image::new(w, h, level[0])?;
                for (j, &value) in level.iter().enumerate().skip(1) {
                    let inside: Vec<bool> = labels.iter().map(|&l| l >= j).collect();
                    let (found, coverage) = tracer.layer(&inside, &format!("{plate}{j}"))?;
                    if let Some(coverage) = coverage {
                        for (i, m) in coverage.pixels().iter().enumerate() {
                            let v = value_image.get_mut(i % w, i / w).unwrap();
                            *v = *v * (1.0 - m) + value * m;
                        }
                    }
                    for shape in &found {
                        svg += &svg_shape(shape, fill(value));
                        browser += &svg_shape(shape, encode(value));
                    }
                }
                svg += "  </g>\n";
                browser += "    </g>\n";
                quantized_values.push(labels.iter().map(|&l| level[l]).collect::<Vec<f64>>());
                plates.push(value_image);
                label_sets.push(labels);
            }
            let mut file = BufWriter::new(File::create(output.join(format!("{stem}_labels.csv")))?);
            writeln!(file, "x,y,{},{},{}", names[0], names[1], names[2])?;
            let triples = label_sets[0].iter().zip(&label_sets[1]).zip(&label_sets[2]);
            for (i, ((c, m), y)) in triples.enumerate() {
                writeln!(file, "{},{},{c},{m},{y}", i % w, i / w)?;
            }
            file.flush()?;
            // Recombine: CMY multiplies ink on white (channel k = 1 - ink_k);
            // Lab converts the three values back to RGB.
            let combine = |v: [f64; 3]| match space {
                Space::Cmy => Color::rgb(1.0 - v[0], 1.0 - v[1], 1.0 - v[2]),
                Space::Lab => lab_to_rgb(v[0], v[1], v[2]),
            };
            let mut image = Image::new(w, h, Color::rgb(1.0, 1.0, 1.0))?;
            let fitted = plates[0]
                .pixels()
                .iter()
                .zip(plates[1].pixels())
                .zip(plates[2].pixels());
            for (i, ((a, b), c)) in fitted.enumerate() {
                *image.get_mut(i % w, i / w).unwrap() = combine([*a, *b, *c]);
            }
            let levelled = quantized_values[0]
                .iter()
                .zip(&quantized_values[1])
                .zip(&quantized_values[2]);
            for (i, ((a, b), c)) in levelled.enumerate() {
                *quantized.get_mut(i % w, i / w).unwrap() = combine([*a, *b, *c]);
            }
            // Browser SVG: write it, and measure what its filter chain gives
            // against the exact conversion, pixel by pixel.
            let mut browser_json = String::from("\"browser_svg_bytes\":null,\"filter_mean_delta_e\":null,\"filter_max_delta_e\":null,\"filter_8bit_mean_delta_e\":null");
            if space == Space::Lab {
                let text = format!(
                    "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" width=\"{w}\" height=\"{h}\">\n{}  <g filter=\"url(#lab)\">\n    <rect width=\"{w}\" height=\"{h}\" fill=\"#000000\"/>\n{browser}  </g>\n</svg>\n",
                    svg_lab_filter(w, h)
                );
                std::fs::write(output.join(format!("{stem}_browser.svg")), &text)?;
                let mut file =
                    BufWriter::new(File::create(output.join(format!("{stem}_plates.csv")))?);
                writeln!(file, "x,y,l,a,b")?;
                let (mut sum, mut worst, mut sum8) = (0.0, 0.0_f64, 0.0);
                let fitted = plates[0]
                    .pixels()
                    .iter()
                    .zip(plates[1].pixels())
                    .zip(plates[2].pixels());
                for (i, ((l, a), b)) in fitted.enumerate() {
                    writeln!(file, "{},{},{l:.9},{a:.9},{b:.9}", i % w, i / w)?;
                    let d = delta_e(svg_filter_colour(*l, *a, *b, false), image.pixels()[i]);
                    sum += d;
                    worst = worst.max(d);
                    sum8 += delta_e(svg_filter_colour(*l, *a, *b, true), image.pixels()[i]);
                }
                file.flush()?;
                browser_json = format!(
                    "\"browser_svg_bytes\":{},\"filter_mean_delta_e\":{:.6},\"filter_max_delta_e\":{worst:.6},\"filter_8bit_mean_delta_e\":{:.6}",
                    text.len(),
                    sum / (w * h) as f64,
                    sum8 / (w * h) as f64
                );
            }
            rebuilt = image;
            method_json = format!(
                "{browser_json},\"method\":\"channels\",\"space\":\"{}\",\"levels_per_channel\":[{},{},{}],\"layers\":{layers},\"levels\":{{{}}}",
                if space == Space::Cmy { "cmy" } else { "lab" },
                levels[0],
                levels[1],
                levels[2],
                level_json.join(",")
            );
        }
    }
    tracer.contours_file.flush()?;
    tracer.primitives_file.flush()?;
    svg += "</svg>\n";
    std::fs::write(output.join(format!("{stem}.svg")), &svg)?;
    quantized.save_bmp(output.join(format!("{stem}_quant.bmp")))?;
    write_rgb(&output.join(format!("{stem}_quant.csv")), &quantized)?;
    rebuilt.save_bmp(output.join(format!("{stem}_vector.bmp")))?;
    write_rgb(&output.join(format!("{stem}_vector.csv")), &rebuilt)?;
    // Colour difference map (darker = larger ΔE against the original).
    let mut map = Image::new(w, h, Color::rgb(1.0, 1.0, 1.0))?;
    for (i, (x, y)) in rebuilt.pixels().iter().zip(original.pixels()).enumerate() {
        let v = 1.0 - (delta_e(*x, *y) / 30.0).min(1.0);
        *map.get_mut(i % w, i / w).unwrap() = Color::rgb(v, v, v);
    }
    map.save_bmp(output.join(format!("{stem}_error.bmp")))?;
    let mean_delta = |a: &Image<Color>| -> (f64, usize) {
        let d: Vec<f64> = a
            .pixels()
            .iter()
            .zip(original.pixels())
            .map(|(x, y)| delta_e(*x, *y))
            .collect();
        (
            d.iter().sum::<f64>() / d.len() as f64,
            d.iter().filter(|v| **v > VISIBLE_DELTA_E).count(),
        )
    };
    let (quant_rms, _) = color_error(&quantized, original);
    let (vector_rms, _) = color_error(&rebuilt, original);
    let (quant_de, quant_visible) = mean_delta(&quantized);
    let (vector_de, vector_visible) = mean_delta(&rebuilt);
    let mismatch = rebuilt
        .pixels()
        .iter()
        .zip(quantized.pixels())
        .filter(|(a, b)| delta_e(**a, **b) > VISIBLE_DELTA_E)
        .count();
    let (quant_ssim, vector_ssim) = (ssim(&quantized, original), ssim(&rebuilt, original));
    let s = &tracer.stats;
    Ok(format!(
        "{{\"painting\":\"{name}\",\"stem\":\"{stem}\",\"clean\":{clean},{method_json},{gaps},\"loops\":{},\"dbscan_clusters\":{},\"shapes\":{},\"holes\":{},\"lines\":{},\"arcs\":{},\"cubics\":{},\"fallbacks\":{},\"svg_bytes\":{},\"bitmap_bytes\":{},\"quant_rms\":{quant_rms:.9},\"quant_mean_delta_e\":{quant_de:.6},\"quant_visible\":{quant_visible},\"vector_rms\":{vector_rms:.9},\"vector_mean_delta_e\":{vector_de:.6},\"vector_visible\":{vector_visible},\"vector_vs_quant_visible\":{mismatch},\"quant_ssim\":{quant_ssim:.9},\"vector_ssim\":{vector_ssim:.9},\"seconds\":{:.3}}}",
        s.loops,
        s.dbscan_clusters,
        s.shapes,
        s.holes,
        s.lines,
        s.arcs,
        s.cubics,
        s.fallbacks,
        svg.len(),
        w * h * 3,
        started.elapsed().as_secs_f64()
    ))
}

/// Runs the showcase on `<folder>/<name>.bmp` (decoded by
/// scripts/decode_paintings.py).
/// `factor`, or the smallest one that brings the longer side of a
/// `width` x `height` picture to at most `longest` pixels.
pub(crate) fn shrink_factor(width: usize, height: usize, longest: usize, factor: Option<usize>) -> usize {
    factor.unwrap_or_else(|| width.max(height).div_ceil(longest).max(1))
}

/// The file stem of a picture, as it appears in every output name.
pub(crate) fn picture_name(path: &Path) -> Result<String, Error> {
    path.file_stem()
        .and_then(|s| s.to_str())
        .map(str::to_owned)
        .ok_or(Error::InvalidParameter("image path has no usable file name"))
}

/// Runs the showcase methods (all, or those named in `methods`) on each
/// picture, shrunk by `factor` or by the LONGEST rule.
pub fn run(output: &Path, images: &[PathBuf], factor: Option<usize>, methods: Option<&[String]>) -> Result<(), Error> {
    let mut entries = Vec::new();
    let mut sizes = Vec::new();
    for path in images {
        let name = picture_name(path)?;
        let source = bmp::read(&std::fs::read(path)?)?;
        let factor = shrink_factor(source.width(), source.height(), LONGEST, factor);
        let input = shrink(&source, factor)?;
        input.save_bmp(output.join(format!("paint_{name}_input.bmp")))?;
        write_rgb(&output.join(format!("paint_{name}_input.csv")), &input)?;
        let smoothed = bilateral(&input)?;
        smoothed.save_bmp(output.join(format!("paint_{name}_smooth.bmp")))?;
        write_rgb(&output.join(format!("paint_{name}_smooth.csv")), &smoothed)?;
        sizes.push(format!(
            "\"{name}\":{{\"source\":[{},{}],\"factor\":{factor},\"size\":[{},{}]}}",
            source.width(),
            source.height(),
            input.width(),
            input.height()
        ));
        for ((method, clean), method_name) in RUNS.into_iter().zip(METHOD_NAMES) {
            if methods.is_some_and(|m| !m.iter().any(|n| n == method_name)) {
                continue;
            }
            entries.push(vectorize_run(
                output, &name, &input, &smoothed, method, clean,
            )?);
        }
    }
    std::fs::write(
        output.join("paintings_summary.json"),
        format!(
            "{{\"min_area\":{MIN_AREA},\"clean_area\":{CLEAN_AREA},\"clean_delta_e\":{CLEAN_DELTA_E},\"bilateral\":{{\"radius\":{},\"sigma_space\":{},\"sigma_delta_e\":{}}},\"fill_k\":{FILL_K},\"fallback_area\":{FALLBACK_AREA},\"fallback_tolerance\":{FALLBACK_TOLERANCE},\"visible_delta_e\":{VISIBLE_DELTA_E},\"kmeans_seed\":1151,\"inputs\":{{{}}},\"runs\":[{}]}}\n",
            BILATERAL.0,
            BILATERAL.1,
            BILATERAL.2,
            sizes.join(","),
            entries.join(",")
        ),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traced_loops_run_clockwise_outside_and_counter_clockwise_around_holes() {
        // A 4 x 4 square with a 2 x 2 hole: one outer loop, one hole loop.
        let (w, h) = (6, 6);
        let inside: Vec<bool> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                (1..5).contains(&x)
                    && (1..5).contains(&y)
                    && !((2..4).contains(&x) && (2..4).contains(&y))
            })
            .collect();
        let mut found = loops(&inside, w, h);
        found.sort_by_key(|l| std::cmp::Reverse(l.len()));
        assert_eq!(found.len(), 2);
        assert_eq!((found[0].len(), found[1].len()), (16, 8));
        // Screen-clockwise has positive shoelace area with y pointing down.
        // Joining crack midpoints cuts 1/8 px² off each square corner.
        assert!((signed_area(&found[0]) - 15.5).abs() < 1e-9);
        assert!((signed_area(&found[1]) + 3.5).abs() < 1e-9);
        // Consecutive crack midpoints are at most one pixel apart.
        for lp in &found {
            for i in 0..lp.len() {
                assert!((lp[(i + 1) % lp.len()] - lp[i]).abs() <= 1.0 + 1e-12);
            }
        }
        // Pixels touching only at a corner stay separate (4-connectivity).
        let diagonal = [true, false, false, true];
        assert_eq!(loops(&diagonal, 2, 2).len(), 2);
    }

    #[test]
    fn despeckling_terminates_on_a_checkerboard_and_leaves_no_small_region() {
        // Every region is one pixel: merging all at once could swap labels
        // forever; the sequential merge must finish.
        let (w, h) = (12, 12);
        let mut labels: Vec<usize> = (0..w * h).map(|i| (i % w + i / w) % 2).collect();
        despeckle(&mut labels, w, h);
        let (_, sizes) = regions(&labels, w, h);
        assert!(sizes.iter().all(|s| *s >= MIN_AREA));
    }

    #[test]
    fn small_regions_merge_only_into_allowed_neighbours() {
        // A 2 x 2 island of label 1 inside label 0, and one of label 2.
        let (w, h) = (8, 4);
        let mut labels = vec![0usize; w * h];
        for (x, y) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
            labels[y * w + x] = 1;
        }
        for (x, y) in [(5, 1), (6, 1), (5, 2), (6, 2)] {
            labels[y * w + x] = 2;
        }
        // Label 2 may not merge into 0; label 1 may.
        merge_small(&mut labels, w, h, 10, &|own, _| own != 2);
        assert_eq!(labels.iter().filter(|&&l| l == 1).count(), 0);
        assert_eq!(labels.iter().filter(|&&l| l == 2).count(), 4);
    }

    #[test]
    fn ink_levels_are_ranked_and_bilateral_keeps_edges() {
        let values: Vec<f64> = [0.9, 0.1, 0.5, 0.11, 0.52, 0.88].repeat(5);
        let labels = kmeans_1d(&values, 3);
        assert_eq!(&labels[..6], &[2, 0, 1, 0, 1, 2]);
        // A sharp black/white edge with one noisy pixel on the white side.
        let (w, h) = (16, 16);
        let mut image = Image::new(w, h, Color::rgb(1.0, 1.0, 1.0)).unwrap();
        for y in 0..h {
            for x in 0..5 {
                *image.get_mut(x, y).unwrap() = Color::rgb(0.0, 0.0, 0.0);
            }
        }
        *image.get_mut(7, 3).unwrap() = Color::rgb(0.94, 0.94, 0.94);
        let smooth = bilateral(&image).unwrap();
        assert!(smooth.get(4, 3).unwrap().r < 0.02 && smooth.get(5, 3).unwrap().r > 0.98);
        assert!(smooth.get(7, 3).unwrap().r > 0.95);
        assert!((ssim(&image, &image) - 1.0).abs() < 1e-12);
        assert!(ssim(&smooth, &image) > 0.9);
    }

    #[test]
    fn svg_filter_chain_matches_lab_away_from_black() {
        // Mid-tone colours: the filter's missing linear pieces near black
        // do not apply, so it should agree with the exact conversion.
        for c in [
            Color::rgb(0.8, 0.6, 0.5),
            Color::rgb(0.3, 0.5, 0.7),
            Color::rgb(0.9, 0.9, 0.85),
        ] {
            let [l, a, b] = lab(c);
            assert!(delta_e(svg_filter_colour(l, a, b, false), lab_to_rgb(l, a, b)) < 0.05);
            // 8-bit buffers between primitives cost a few ΔE at most.
            assert!(delta_e(svg_filter_colour(l, a, b, true), lab_to_rgb(l, a, b)) < 5.0);
        }
        let filter = svg_lab_filter(10, 20);
        assert_eq!(filter.matches("<feColorMatrix").count(), 2);
        assert_eq!(filter.matches("<feFunc").count(), 6);
    }

    #[test]
    fn lab_round_trips_back_to_rgb() {
        for c in [
            Color::rgb(0.9, 0.1, 0.1),
            Color::rgb(0.2, 0.5, 0.8),
            Color::rgb(0.95, 0.93, 0.88),
            Color::rgb(0.01, 0.02, 0.03),
        ] {
            let [l, a, b] = lab(c);
            let back = lab_to_rgb(l, a, b);
            assert!(
                (back.r - c.r).abs() < 2e-3
                    && (back.g - c.g).abs() < 2e-3
                    && (back.b - c.b).abs() < 2e-3
            );
        }
    }

    #[test]
    fn dft_of_a_sampled_circle_is_a_single_harmonic() {
        let points: Vec<Complex> = (0..64)
            .map(|j| {
                Complex::new(10.0, 5.0) + Complex::from_angle(TAU * j as f64 / 64.0).scale(7.0)
            })
            .collect();
        let curve = dft(&points, 4);
        for (i, c) in curve.coefficients.iter().enumerate() {
            let k = i as i64 - 4;
            let expected = match k {
                0 => Complex::new(10.0, 5.0),
                1 => Complex::new(7.0, 0.0),
                _ => Complex::ZERO,
            };
            assert!((*c - expected).abs() < 1e-9, "k = {k}");
        }
    }
}
