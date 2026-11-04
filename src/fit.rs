//! Fourier fits of one outline: the DFT, and chamfer gradient descent.
//!
//! Curves are sampled evenly in t, so both the samples and the gradient are
//! a single FFT: O(S log S) per step instead of O(S * harmonics).
use crate::{fft, outline::signed_area, Complex, Error};
use std::sync::atomic::{AtomicUsize, Ordering};

const MAX_HARMONICS: usize = 1024;
/// Steps of descent when refining the DFT result, and per coarse-to-fine
/// stage when starting from a circle.
const REFINE_STEPS: usize = 200;
const STAGE_STEPS: usize = 200;
pub(crate) const RATE: f64 = 0.2;

/// Static 2-d tree for exact nearest-point queries: a query costs about log n
/// nodes at any distance, even 300 px from the curve (a starting circle
/// against a long outline).
pub(crate) struct KdTree<'a> {
    points: &'a [Complex],
    /// Point indices; each slice's middle element splits it, by x at even
    /// depths and by y at odd ones.
    order: Vec<usize>,
}
impl<'a> KdTree<'a> {
    pub(crate) fn new(points: &'a [Complex]) -> Self {
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
pub(crate) fn samples(coefficients: &[Complex], m: usize, s: usize) -> Result<Vec<Complex>, Error> {
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
/// with respect to every c_k. The gradient sum_j g_j e^(-ik t_j) is the
/// forward FFT of the per-sample pulls g_j.
pub(crate) fn chamfer(
    coefficients: &[Complex],
    m: usize,
    s: usize,
    targets: &[Complex],
    target_tree: &KdTree,
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
    let gradient: Vec<Complex> = (-(m as i64)..=m as i64)
        .map(|k| pull[k.rem_euclid(s as i64) as usize])
        .collect();
    Ok((loss, gradient))
}

/// `steps` of c_k -= RATE * dL/dc_k; returns the chamfer loss after the last
/// step.
fn descend(
    coefficients: &mut [Complex],
    m: usize,
    s: usize,
    targets: &[Complex],
    target_tree: &KdTree,
    steps: usize,
) -> Result<f64, Error> {
    for _ in 0..steps {
        let (_, gradient) = chamfer(coefficients, m, s, targets, target_tree)?;
        for (c, g) in coefficients.iter_mut().zip(gradient) {
            *c = *c - g.scale(RATE);
        }
    }
    Ok(chamfer(coefficients, m, s, targets, target_tree)?.0)
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
        let loss = descend(&mut c, stage, s, points, tree, STAGE_STEPS)?;
        if stage == m {
            return Ok((c, loss));
        }
    }
}

/// How to get an outline's coefficients.
#[derive(Clone, Copy, PartialEq)]
pub enum FitKind {
    /// The ordered outline, resampled evenly by length and transformed.
    Dft,
    /// The DFT result improved by chamfer gradient descent.
    Refined,
    /// Descent alone from a circle, coarse to fine; only the outline's points
    /// are used, in no particular order, so it can cut across between distant
    /// parts and flip areas inside out.
    Descent,
}
impl FitKind {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Dft => "dft",
            Self::Refined => "refined",
            Self::Descent => "descent",
        }
    }
}

/// One outline's fit: its points, harmonics, samples per descent step,
/// coefficients c_-m..c_m and chamfer loss.
pub(crate) struct Fit {
    pub(crate) n: usize,
    pub(crate) m: usize,
    pub(crate) s: usize,
    pub(crate) coefficients: Vec<Complex>,
    pub(crate) loss: f64,
}

/// Harmonics of an outline with n points.
pub(crate) fn harmonics(n: usize) -> usize {
    (n / 3).clamp(2, MAX_HARMONICS)
}

fn fit(points: &[Complex], kind: FitKind) -> Result<Fit, Error> {
    let n = points.len();
    let m = harmonics(n);
    let s = (2 * n).max(4 * m + 2).next_power_of_two().max(64);
    let tree = KdTree::new(points);
    let (coefficients, loss) = match kind {
        FitKind::Dft => {
            let c = dft(points, m, s)?;
            let loss = chamfer(&c, m, s, points, &tree)?.0;
            (c, loss)
        }
        FitKind::Refined => {
            let mut c = dft(points, m, s)?;
            let loss = descend(&mut c, m, s, points, &tree, REFINE_STEPS)?;
            (c, loss)
        }
        FitKind::Descent => from_circle(points, m, s, &tree)?,
    };
    Ok(Fit { n, m, s, coefficients, loss })
}

/// Runs `work` on every index of `order`, spread over the available cores.
/// The indices wait in one queue and each thread takes the next one as soon
/// as it is free, so no core idles while another works through a long list.
/// Results come back sorted by index, so the run does not depend on timing.
pub(crate) fn in_queue<T: Send>(order: &[usize], work: impl Fn(usize) -> Result<T, Error> + Sync) -> Result<Vec<T>, Error> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(32);
    let next = AtomicUsize::new(0);
    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&i) = order.get(k) else { break };
                        done.push((i, work(i)?));
                    }
                    Ok::<_, Error>(done)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker thread panicked"))
            .collect::<Result<Vec<_>, Error>>()
    })?;
    let mut all: Vec<(usize, T)> = results.into_iter().flatten().collect();
    all.sort_by_key(|(i, _)| *i);
    Ok(all.into_iter().map(|(_, t)| t).collect())
}

/// Fits every outline in one queue, largest first.
pub(crate) fn fit_all(outlines: &[Vec<Complex>], kind: FitKind) -> Result<Vec<Fit>, Error> {
    let mut order: Vec<usize> = (0..outlines.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(outlines[i].len()));
    in_queue(&order, |i| fit(&outlines[i], kind))
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
}
