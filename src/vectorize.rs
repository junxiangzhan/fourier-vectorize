//! Part 4 appendix (Vectorizing, Chapter 3): describe the star bitmap with N
//! Fourier coefficients in two different ways and draw it back.
//!
//! A. Scan the bitmap along a Hilbert curve into a 1D brightness signal,
//!    keep the N strongest Fourier coefficients of that signal, and put the
//!    reconstruction back along the same curve. The result is still a bitmap.
//! B. Find the bitmap's sub-pixel edge points (unordered) and fit a closed
//!    Fourier curve z(t) = sum_{|k|<=M} c_k e^(ikt), N = 2M+1 coefficients,
//!    by gradient descent on the chamfer distance. The result is a vector
//!    outline. A reference ("oracle") truncates the Fourier series of the
//!    true outline, which a vectorizer would not have.
//!
//! Every picture, including curves and circles, is written pixel by pixel
//! into our own arrays; no drawing library is involved.
use crate::{
    experiments::{save_coverage, star_path, write_coverage},
    presentation::write_points,
};
use multimedia::{
    fft, hilbert, Color, Complex, Error, FlattenedPath, Image, Point2, Polyline, Rasterizer,
};
use std::{
    collections::HashMap,
    f64::consts::{PI, TAU},
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

/// Hilbert curves need a power-of-two square canvas.
pub(crate) const SIZE: usize = 256;
pub(crate) const K: usize = 8;
const A_BUDGETS: [usize; 8] = [9, 17, 33, 65, 257, 1025, 4097, 16385];
pub(crate) const B_HARMONICS: [usize; 5] = [4, 8, 16, 32, 64];
/// Points on the fitted curve while descending (must exceed 2 * max M).
const SAMPLES: usize = 512;
const ITERATIONS: usize = 300;
const RATE: f64 = 0.2;
/// Points used to draw and measure a finished curve.
pub(crate) const DENSE: usize = 4096;

pub(crate) fn rms(a: &Image<f64>, b: &Image<f64>) -> f64 {
    let sum: f64 = a
        .pixels()
        .iter()
        .zip(b.pixels())
        .map(|(x, y)| (x - y) * (x - y))
        .sum();
    (sum / a.pixels().len() as f64).sqrt()
}

/// Pixels on the wrong side of 50% coverage: inside/outside mistakes.
pub(crate) fn misclassified(a: &Image<f64>, b: &Image<f64>) -> usize {
    a.pixels()
        .iter()
        .zip(b.pixels())
        .filter(|(x, y)| (**x >= 0.5) != (**y >= 0.5))
        .count()
}

pub(crate) fn point(z: Complex) -> Point2 {
    Point2::new(z.re, z.im)
}

// ---------------------------------------------------------------- A ----

/// Pixel positions in Hilbert order, and the values read there.
type Scan = (Vec<(usize, usize)>, Vec<f64>);

/// Reads the image along the Hilbert curve: positions and values in order.
fn hilbert_scan(image: &Image<f64>) -> Result<Scan, Error> {
    let mut order = Vec::with_capacity(SIZE * SIZE);
    let mut signal = Vec::with_capacity(SIZE * SIZE);
    for d in 0..SIZE * SIZE {
        let (x, y) = hilbert::d2xy(SIZE, d)?;
        order.push((x, y));
        signal.push(*image.get(x, y).unwrap());
    }
    Ok((order, signal))
}

/// Indices f of the half spectrum (0..=L/2) sorted by the energy they carry,
/// strongest first. A real signal has X[L-f] = conj(X[f]), so every f
/// except 0 and L/2 stands for a conjugate pair and counts twice.
fn ranked_frequencies(spectrum: &[Complex]) -> Vec<usize> {
    let length = spectrum.len();
    let energy = |f: usize| {
        let pair = if f == 0 || f == length / 2 { 1.0 } else { 2.0 };
        spectrum[f].norm_squared() * pair
    };
    let mut ranked: Vec<usize> = (0..=length / 2).collect();
    ranked.sort_by(|&a, &b| energy(b).total_cmp(&energy(a)).then(a.cmp(&b)));
    ranked
}

/// Keeps the `budget` strongest frequencies (with their conjugates), inverts
/// the transform, and writes the signal back along the Hilbert curve.
fn reconstruct_a(
    spectrum: &[Complex],
    ranked: &[usize],
    order: &[(usize, usize)],
    budget: usize,
) -> Result<Image<f64>, Error> {
    let length = spectrum.len();
    let mut kept = vec![Complex::ZERO; length];
    for &f in &ranked[..budget] {
        kept[f] = spectrum[f];
        if f != 0 && f != length / 2 {
            kept[length - f] = spectrum[length - f];
        }
    }
    fft(&mut kept, true)?;
    let mut image = Image::new(SIZE, SIZE, 0.0)?;
    for (&(x, y), value) in order.iter().zip(&kept) {
        // Brightness is physical only in [0, 1]; truncation overshoots.
        *image.get_mut(x, y).unwrap() = value.re.clamp(0.0, 1.0);
    }
    Ok(image)
}

// ---------------------------------------------------------------- B ----

/// Where coverage crosses 50% between horizontally or vertically adjacent
/// pixel centers, by linear interpolation: sub-pixel edge points, in no
/// particular order. Ordering them is exactly what B avoids having to do.
pub(crate) fn edge_points(image: &Image<f64>) -> Vec<Complex> {
    let mut points = Vec::new();
    let value = |x: usize, y: usize| *image.get(x, y).unwrap();
    for y in 0..image.height() {
        for x in 0..image.width() {
            let here = value(x, y);
            for (nx, ny, horizontal) in [(x + 1, y, true), (x, y + 1, false)] {
                if nx >= image.width() || ny >= image.height() {
                    continue;
                }
                let there = value(nx, ny);
                // Inside means coverage >= 0.5; a pixel at exactly 0.5 must not
                // leave a gap in the outline.
                if (here >= 0.5) != (there >= 0.5) {
                    let t = (0.5 - here) / (there - here);
                    let (px, py) = if horizontal {
                        (x as f64 + 0.5 + t, y as f64 + 0.5)
                    } else {
                        (x as f64 + 0.5, y as f64 + 0.5 + t)
                    };
                    points.push(Complex::new(px, py));
                }
            }
        }
    }
    points
}

/// Coefficients c_k for k = -M..=M, stored at index k + M.
pub(crate) struct FourierCurve {
    pub(crate) harmonics: usize,
    pub(crate) coefficients: Vec<Complex>,
}
impl FourierCurve {
    /// z(t) = sum_k c_k e^(ikt), t in [0, 2*pi).
    pub(crate) fn eval(&self, t: f64) -> Complex {
        let m = self.harmonics as i64;
        self.coefficients
            .iter()
            .enumerate()
            .fold(Complex::ZERO, |sum, (i, c)| {
                sum + *c * Complex::from_angle((i as i64 - m) as f64 * t)
            })
    }
    pub(crate) fn sample(&self, count: usize) -> Vec<Complex> {
        (0..count)
            .map(|j| self.eval(TAU * j as f64 / count as f64))
            .collect()
    }
    /// Same curve with more harmonics; the new ones start at zero.
    pub(crate) fn extended(&self, harmonics: usize) -> Self {
        let pad = harmonics - self.harmonics;
        let mut coefficients = vec![Complex::ZERO; 2 * harmonics + 1];
        coefficients[pad..pad + self.coefficients.len()].copy_from_slice(&self.coefficients);
        Self {
            harmonics,
            coefficients,
        }
    }
}

/// Symmetric chamfer loss: mean squared distance from each curve sample to
/// its nearest edge point, plus from each edge point to its nearest curve
/// sample. The first term keeps the curve on the edges; the second pulls it
/// out into every tip. Returns the loss and dL/dc_k for every k.
/// Points bucketed into square cells for exact nearest-neighbour queries.
/// A query visits rings of cells around its own cell, nearest ring first,
/// and stops once every unvisited cell is farther than the best match.
struct Grid<'a> {
    points: &'a [Complex],
    buckets: HashMap<(i64, i64), Vec<usize>>,
    /// Largest ring that can still contain a point, from any query cell.
    reach: i64,
    low: (i64, i64),
    high: (i64, i64),
}
impl<'a> Grid<'a> {
    const CELL: f64 = 4.0;
    fn cell(q: Complex) -> (i64, i64) {
        (
            (q.re / Self::CELL).floor() as i64,
            (q.im / Self::CELL).floor() as i64,
        )
    }
    fn new(points: &'a [Complex]) -> Self {
        let mut buckets: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        let (mut low, mut high) = ((i64::MAX, i64::MAX), (i64::MIN, i64::MIN));
        for (i, q) in points.iter().enumerate() {
            let c = Self::cell(*q);
            low = (low.0.min(c.0), low.1.min(c.1));
            high = (high.0.max(c.0), high.1.max(c.1));
            buckets.entry(c).or_default().push(i);
        }
        Self {
            points,
            buckets,
            reach: 0,
            low,
            high,
        }
    }
    /// Squared distance and index of the nearest point; ties go to the
    /// lowest index, exactly as a brute-force scan with `<` would choose.
    fn nearest(&self, q: Complex) -> (f64, usize) {
        let (cx, cy) = Self::cell(q);
        // Rings beyond this distance (in cells) hold no points at all.
        let reach = self.reach.max(
            (cx - self.low.0)
                .abs()
                .max((cx - self.high.0).abs())
                .max((cy - self.low.1).abs())
                .max((cy - self.high.1).abs()),
        );
        let mut best = (f64::INFINITY, usize::MAX);
        for r in 0..=reach {
            for dx in -r..=r {
                for dy in -r..=r {
                    if dx.abs() != r && dy.abs() != r {
                        continue; // interior cells were visited earlier
                    }
                    for &i in self.buckets.get(&(cx + dx, cy + dy)).into_iter().flatten() {
                        let d = (self.points[i] - q).norm_squared();
                        if d < best.0 || (d == best.0 && i < best.1) {
                            best = (d, i);
                        }
                    }
                }
            }
            // Any point in ring r+1 or beyond is at least r cells away.
            let bound = r as f64 * Self::CELL;
            if best.0 < bound * bound {
                break;
            }
        }
        best
    }
}

/// The table e^(ik t_j), k = -M..=M, t_j = 2*pi*j/SAMPLES: built once per
/// descent stage, so each iteration needs no sine or cosine.
pub(crate) struct Basis {
    table: Vec<Vec<Complex>>,
}
impl Basis {
    pub(crate) fn new(harmonics: usize) -> Self {
        let m = harmonics as i64;
        let table = (-m..=m)
            .map(|k| {
                (0..SAMPLES)
                    .map(|j| Complex::from_angle(k as f64 * (TAU * j as f64 / SAMPLES as f64)))
                    .collect()
            })
            .collect();
        Self { table }
    }
}

pub(crate) fn chamfer(
    curve: &FourierCurve,
    targets: &[Complex],
    basis: &Basis,
) -> (f64, Vec<Complex>) {
    // z_j = sum_k c_k e^(ik t_j), read from the table.
    let z: Vec<Complex> = (0..SAMPLES)
        .map(|j| {
            curve
                .coefficients
                .iter()
                .zip(&basis.table)
                .fold(Complex::ZERO, |sum, (c, row)| sum + *c * row[j])
        })
        .collect();
    // Exact nearest neighbours in both directions, via uniform grids.
    let target_grid = Grid::new(targets);
    let sample_grid = Grid::new(&z);
    let nearest_target: Vec<(f64, usize)> = z.iter().map(|q| target_grid.nearest(*q)).collect();
    let nearest_sample: Vec<(f64, usize)> =
        targets.iter().map(|q| sample_grid.nearest(*q)).collect();
    // g_j = dL/dx_j + i dL/dy_j: the gradient on each curve sample.
    let mut g = vec![Complex::ZERO; z.len()];
    let mut loss = 0.0;
    let (s, p) = (z.len() as f64, targets.len() as f64);
    for (j, &(d, i)) in nearest_target.iter().enumerate() {
        loss += d / s;
        g[j] = g[j] + (z[j] - targets[i]).scale(2.0 / s);
    }
    for (i, &(d, j)) in nearest_sample.iter().enumerate() {
        loss += d / p;
        g[j] = g[j] + (z[j] - targets[i]).scale(2.0 / p);
    }
    // Chain rule through z_j = sum_k c_k e^(ik t_j): the steepest-descent
    // direction for c_k is -sum_j g_j e^(-ik t_j).
    let gradient = basis
        .table
        .iter()
        .map(|row| {
            g.iter()
                .zip(row)
                .fold(Complex::ZERO, |sum, (gj, e)| sum + *gj * e.conj())
        })
        .collect();
    (loss, gradient)
}

/// Starting curve for B: a circle at the edge points' centroid with their
/// mean distance from it as radius.
pub(crate) fn initial_circle(targets: &[Complex]) -> FourierCurve {
    let centroid = targets
        .iter()
        .fold(Complex::ZERO, |sum, p| sum + *p)
        .scale(1.0 / targets.len() as f64);
    let radius = targets.iter().map(|p| (*p - centroid).abs()).sum::<f64>() / targets.len() as f64;
    FourierCurve {
        harmonics: 1,
        coefficients: vec![Complex::ZERO, centroid, Complex::new(radius, 0.0)],
    }
}

/// One gradient-descent stage at the curve's current number of harmonics:
/// ITERATIONS steps of c_k -= RATE * dL/dc_k. `record` sees the loss
/// before every step and after the last; the final loss is returned.
pub(crate) fn descend(
    curve: &mut FourierCurve,
    targets: &[Complex],
    mut record: impl FnMut(usize, f64) -> std::io::Result<()>,
) -> Result<f64, Error> {
    let basis = Basis::new(curve.harmonics);
    let mut loss = 0.0;
    for iteration in 0..=ITERATIONS {
        let (value, gradient) = chamfer(curve, targets, &basis);
        loss = value;
        record(iteration, value)?;
        if iteration == ITERATIONS {
            break;
        }
        for (c, g) in curve.coefficients.iter_mut().zip(gradient) {
            *c = *c - g.scale(RATE);
        }
    }
    Ok(loss)
}

/// The true filled outline of the five-pointed star: a concave decagon
/// alternating tips and inner corners, rotated 5° like the Part 3 star.
fn true_outline() -> (Vec<Complex>, Vec<Complex>) {
    let center = SIZE as f64 / 2.0;
    let outer = 0.42 * SIZE as f64;
    // Inner corners of a regular pentagram: R * cos(72°) / cos(36°).
    let inner = outer * (2.0 * PI / 5.0).cos() / (PI / 5.0).cos();
    let spin = 5.0_f64.to_radians();
    let at = |radius: f64, angle: f64| {
        Complex::new(center, center) + Complex::from_angle(angle).scale(radius)
    };
    let tips = (0..5)
        .map(|i| at(outer, -PI / 2.0 + i as f64 * TAU / 5.0 + spin))
        .collect();
    let corners = (0..5)
        .map(|i| at(inner, -PI / 2.0 + i as f64 * TAU / 5.0 + PI / 5.0 + spin))
        .collect();
    (tips, corners)
}

/// Oracle: Fourier series of the true outline, sampled uniformly by arc
/// length, truncated to M harmonics. Needs the ordered vector outline.
fn oracle(tips: &[Complex], corners: &[Complex], harmonics: usize) -> Result<FourierCurve, Error> {
    let polygon: Vec<Complex> = (0..5).flat_map(|i| [tips[i], corners[i]]).collect();
    let length: f64 = (0..10)
        .map(|i| (polygon[(i + 1) % 10] - polygon[i]).abs())
        .sum();
    let count = DENSE;
    let mut samples = Vec::with_capacity(count);
    let (mut edge, mut walked) = (0, 0.0);
    for j in 0..count {
        let target = length * j as f64 / count as f64;
        loop {
            let a = polygon[edge];
            let b = polygon[(edge + 1) % 10];
            let edge_length = (b - a).abs();
            if target <= walked + edge_length || edge == 9 {
                samples.push(a + (b - a).scale((target - walked) / edge_length));
                break;
            }
            walked += edge_length;
            edge += 1;
        }
    }
    fft(&mut samples, false)?;
    // With z_n = sum_k c_k e^(2*pi*i*k*n/N): c_k = X[k]/N, negative k wraps.
    let coefficients = (-(harmonics as i64)..=harmonics as i64)
        .map(|k| samples[k.rem_euclid(count as i64) as usize].scale(1.0 / count as f64))
        .collect();
    Ok(FourierCurve {
        harmonics,
        coefficients,
    })
}

/// Distance from `p` to the nearest point of a densely sampled curve.
pub(crate) fn distance_to(curve: &[Complex], p: Complex) -> f64 {
    curve
        .iter()
        .map(|z| (*z - p).norm_squared())
        .fold(f64::INFINITY, f64::min)
        .sqrt()
}

struct Outline {
    rms: f64,
    misclassified: usize,
    mean_edge: f64,
    max_edge: f64,
    tip: f64,
    corner: f64,
    image: Image<f64>,
    dense: Vec<Complex>,
}

/// Fills a Fourier curve with our own rasterizer and measures it.
fn measure(
    curve: &FourierCurve,
    original: &Image<f64>,
    targets: &[Complex],
    tips: &[Complex],
    corners: &[Complex],
) -> Result<Outline, Error> {
    let dense = curve.sample(DENSE);
    let path = FlattenedPath {
        subpaths: vec![Polyline {
            vertices: dense.iter().map(|z| point(*z)).collect(),
            closed: true,
        }],
    };
    let image = Rasterizer::new(SIZE, SIZE)?.render(&path, K)?;
    let edges: Vec<f64> = targets.iter().map(|p| distance_to(&dense, *p)).collect();
    Ok(Outline {
        rms: rms(&image, original),
        misclassified: misclassified(&image, original),
        mean_edge: edges.iter().sum::<f64>() / edges.len() as f64,
        max_edge: edges.iter().copied().fold(0.0, f64::max),
        tip: tips
            .iter()
            .map(|p| distance_to(&dense, *p))
            .fold(0.0, f64::max),
        corner: corners
            .iter()
            .map(|p| distance_to(&dense, *p))
            .fold(0.0, f64::max),
        image,
        dense,
    })
}

// ------------------------------------------------------------ drawing ----

/// Marks the pixel containing `z`: point sampling into our own array.
pub(crate) fn plot(image: &mut Image<Color>, z: Complex, color: Color) {
    if z.re >= 0.0 && z.im >= 0.0 {
        if let Some(pixel) = image.get_mut(z.re as usize, z.im as usize) {
            *pixel = color;
        }
    }
}

/// A line as closely spaced points (4 per pixel of length), so no pixel
/// along it is skipped.
pub(crate) fn plot_line(image: &mut Image<Color>, a: Complex, b: Complex, color: Color) {
    let steps = ((b - a).abs() * 4.0).ceil().max(1.0) as usize;
    for i in 0..=steps {
        plot(image, a + (b - a).scale(i as f64 / steps as f64), color);
    }
}

pub(crate) fn plot_circle(image: &mut Image<Color>, center: Complex, radius: f64, color: Color) {
    let steps = (radius * TAU * 4.0).ceil().max(8.0) as usize;
    for i in 0..steps {
        plot(
            image,
            center + Complex::from_angle(TAU * i as f64 / steps as f64).scale(radius),
            color,
        );
    }
}

/// The original bitmap in light gray as a background for overlays.
pub(crate) fn backdrop(original: &Image<f64>) -> Result<Image<Color>, Error> {
    original.map(|c| Color::rgb(1.0, 1.0, 1.0).mix(Color::rgb(0.78, 0.80, 0.84), *c))
}

pub(crate) const RED: Color = Color::rgb(0.85, 0.15, 0.1);
pub(crate) const BLUE: Color = Color::rgb(0.08, 0.45, 0.75);

/// Epicycles: the fitted terms as a chain of rotating circles at parameter
/// t, largest first, with the traced outline. The chain's tip is z(t).
fn epicycles(
    original: &Image<f64>,
    curve: &FourierCurve,
    dense: &[Complex],
    t: f64,
) -> Result<Image<Color>, Error> {
    let mut image = backdrop(original)?;
    for pair in dense.windows(2) {
        plot_line(&mut image, pair[0], pair[1], RED);
    }
    let m = curve.harmonics as i64;
    let mut terms: Vec<(i64, Complex)> = curve
        .coefficients
        .iter()
        .enumerate()
        .map(|(i, c)| (i as i64 - m, *c))
        .filter(|(k, _)| *k != 0)
        .collect();
    terms.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
    // The constant term c_0 is where the chain is anchored.
    let mut tip = curve.coefficients[curve.harmonics];
    for (k, c) in terms {
        let arm = c * Complex::from_angle(k as f64 * t);
        plot_circle(&mut image, tip, c.abs(), BLUE);
        plot_line(&mut image, tip, tip + arm, Color::rgb(0.0, 0.0, 0.0));
        tip = tip + arm;
    }
    Ok(image)
}

// ---------------------------------------------------------------- run ----

pub fn run(output: &Path) -> Result<(), Error> {
    let raster = Rasterizer::new(SIZE, SIZE)?;
    // The bitmap to vectorize: the 5°-rotated star, k = 8 coverage.
    let original = raster.render(&star_path(SIZE, 5.0_f64.to_radians()).flatten(0)?, K)?;
    save_coverage(&original, &output.join("vectorize_original.bmp"))?;
    write_coverage(&output.join("vectorize_original_coverage.csv"), &original)?;
    let mut metrics = BufWriter::new(File::create(output.join("vectorize_metrics.csv"))?);
    writeln!(metrics, "method,budget,harmonics,rms,misclassified,mean_edge_px,max_edge_px,tip_px,corner_px,final_loss")?;

    // A: Hilbert scan, 1D spectrum, strongest N coefficients.
    let (order, signal) = hilbert_scan(&original)?;
    let mut scan = BufWriter::new(File::create(output.join("vectorize_signal.csv"))?);
    writeln!(scan, "index,x,y,value")?;
    for (d, (&(x, y), v)) in order.iter().zip(&signal).enumerate() {
        writeln!(scan, "{d},{x},{y},{v:.9}")?;
    }
    scan.flush()?;
    // Visualize the scan order: gray level = position along the curve.
    let mut index_map = Image::new(SIZE, SIZE, 0.0)?;
    for (d, &(x, y)) in order.iter().enumerate() {
        *index_map.get_mut(x, y).unwrap() = d as f64 / (order.len() - 1) as f64;
    }
    save_coverage(&index_map, &output.join("vectorize_hilbert_order.bmp"))?;
    let mut spectrum: Vec<Complex> = signal.iter().map(|v| Complex::new(*v, 0.0)).collect();
    fft(&mut spectrum, false)?;
    let ranked = ranked_frequencies(&spectrum);
    for budget in A_BUDGETS {
        let image = reconstruct_a(&spectrum, &ranked, &order, budget)?;
        save_coverage(&image, &output.join(format!("vectorize_a_n{budget}.bmp")))?;
        write_coverage(
            &output.join(format!("vectorize_a_n{budget}_coverage.csv")),
            &image,
        )?;
        writeln!(
            metrics,
            "hilbert,{budget},,{:.12},{},,,,,",
            rms(&image, &original),
            misclassified(&image, &original)
        )?;
    }

    // B: fit a Fourier outline to the unordered edge points.
    let targets = edge_points(&original);
    write_points(
        &output.join("vectorize_edge_points.csv"),
        &targets.iter().map(|z| point(*z)).collect::<Vec<_>>(),
    )?;
    let (tips, corners) = true_outline();
    let mut curve = initial_circle(&targets);
    let mut losses = BufWriter::new(File::create(output.join("vectorize_b_loss.csv"))?);
    writeln!(losses, "harmonics,iteration,loss")?;
    let mut coefficients =
        BufWriter::new(File::create(output.join("vectorize_b_coefficients.csv"))?);
    writeln!(coefficients, "harmonics,k,re,im")?;
    // Coarse to fine: each stage unlocks more harmonics, starting from the
    // previous stage's answer, so early stages settle the overall shape.
    for harmonics in B_HARMONICS {
        curve = curve.extended(harmonics);
        let loss = descend(&mut curve, &targets, |iteration, value| {
            writeln!(losses, "{harmonics},{iteration},{value:.12}")
        })?;
        for (i, c) in curve.coefficients.iter().enumerate() {
            writeln!(
                coefficients,
                "{harmonics},{},{:.12},{:.12}",
                i as i64 - harmonics as i64,
                c.re,
                c.im
            )?;
        }
        let result = measure(&curve, &original, &targets, &tips, &corners)?;
        save_coverage(
            &result.image,
            &output.join(format!("vectorize_b_m{harmonics}.bmp")),
        )?;
        write_coverage(
            &output.join(format!("vectorize_b_m{harmonics}_coverage.csv")),
            &result.image,
        )?;
        let mut overlay = backdrop(&original)?;
        for pair in result.dense.windows(2) {
            plot_line(&mut overlay, pair[0], pair[1], RED);
        }
        overlay.save_bmp(output.join(format!("vectorize_b_overlay_m{harmonics}.bmp")))?;
        writeln!(
            metrics,
            "fourier_fit,{},{harmonics},{:.12},{},{:.12},{:.12},{:.12},{:.12},{loss:.12}",
            2 * harmonics + 1,
            result.rms,
            result.misclassified,
            result.mean_edge,
            result.max_edge,
            result.tip,
            result.corner
        )?;
        if harmonics == 8 {
            epicycles(&original, &curve, &result.dense, 0.15 * TAU)?
                .save_bmp(output.join("vectorize_epicycles_m8.bmp"))?;
        }
        // Oracle with the same number of harmonics, for reference.
        let reference = measure(
            &oracle(&tips, &corners, harmonics)?,
            &original,
            &targets,
            &tips,
            &corners,
        )?;
        writeln!(
            metrics,
            "oracle,{},{harmonics},{:.12},{},{:.12},{:.12},{:.12},{:.12},",
            2 * harmonics + 1,
            reference.rms,
            reference.misclassified,
            reference.mean_edge,
            reference.max_edge,
            reference.tip,
            reference.corner
        )?;
    }
    losses.flush()?;
    coefficients.flush()?;
    metrics.flush()?;
    std::fs::write(
        output.join("vectorize_summary.json"),
        format!(
            "{{\"canvas\":{SIZE},\"k\":{K},\"signal_length\":{},\"a_budgets\":{:?},\"a_selection\":\"strongest half-spectrum frequencies by energy, conjugate pairs counted twice\",\"edge_points\":{},\"b_harmonics\":{:?},\"b_samples\":{SAMPLES},\"b_iterations_per_stage\":{ITERATIONS},\"b_rate\":{RATE},\"b_init\":\"circle at edge centroid, mean radius\",\"epicycle_t\":{:.12},\"inner_radius_ratio\":{:.12}}}\n",
            signal.len(),
            A_BUDGETS,
            targets.len(),
            B_HARMONICS,
            0.15 * TAU,
            (2.0 * PI / 5.0).cos() / (PI / 5.0).cos()
        ),
    )?;
    Ok(())
}
