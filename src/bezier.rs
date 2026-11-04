//! Replacing a closed curve's many sample points with a few straight lines and
//! cubic Bezier curves, each staying within a tolerance of the points it spans.
use crate::Complex;

/// One step along a closed curve, from wherever the last one ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Segment {
    Line { end: Complex },
    Cubic { first: Complex, second: Complex, end: Complex },
}

/// A line costs one point and a cubic three, so a cubic has to span more than
/// three times as many samples to be worth it.
const CUBIC_COST: usize = 3;
/// Least-squares fits, each refining where the samples fall along the curve.
const REFINEMENTS: usize = 3;

/// Segments that go once around the closed curve `z`, starting and ending at
/// `z[0]`, whose direction at each sample is `tangents` (which set how each
/// cubic leaves and arrives, so that cubics join smoothly). Each sample ends up
/// within `tolerance` of the segment that spans it.
pub(crate) fn fit_loop(z: &[Complex], tangents: &[Complex], tolerance: f64) -> Vec<Segment> {
    let n = z.len();
    let at = |i: usize| z[i % n];
    let mut closed = Vec::with_capacity(n + 1);
    closed.extend_from_slice(z);
    closed.push(z[0]);
    let mut fit = Fit::default();
    let mut segments = Vec::new();
    let mut i = 0;
    while i < n {
        let straight = longest(n - i, |span| line_error(&closed[i..=i + span]) <= tolerance);
        let curved = longest(n - i, |span| {
            span < 3 || fit.cubic(&closed[i..=i + span], tangents[i], tangents[(i + span) % n]) <= tolerance
        });
        if curved < 3 || straight * CUBIC_COST >= curved {
            i += straight;
            segments.push(Segment::Line { end: at(i) });
        } else {
            fit.cubic(&closed[i..=i + curved], tangents[i], tangents[(i + curved) % n]);
            i += curved;
            segments.push(Segment::Cubic { first: fit.first, second: fit.second, end: at(i) });
        }
    }
    segments
}

/// The longest span up to `most` that `fits`, doubling and then bisecting. A
/// span of 1 always counts: it has no samples inside it to miss.
fn longest(most: usize, mut fits: impl FnMut(usize) -> bool) -> usize {
    let (mut low, mut step) = (1, 2);
    while step <= most && fits(step) {
        (low, step) = (step, step * 2);
    }
    let mut high = step.min(most + 1);
    while high - low > 1 {
        let middle = (low + high) / 2;
        if fits(middle) {
            low = middle;
        } else {
            high = middle;
        }
    }
    low
}

fn dot(a: Complex, b: Complex) -> f64 {
    a.re * b.re + a.im * b.im
}

/// The farthest any inner point lies from the straight line through the ends.
fn line_error(points: &[Complex]) -> f64 {
    let (a, b) = (points[0], points[points.len() - 1]);
    let along = b - a;
    let squared = along.norm_squared();
    let mut worst = 0.0f64;
    for &point in &points[1..points.len() - 1] {
        let q = point - a;
        let t = if squared > 0.0 { (dot(q, along) / squared).clamp(0.0, 1.0) } else { 0.0 };
        worst = worst.max((q - along.scale(t)).abs());
    }
    worst
}

/// The direction of `v`, or `otherwise` when it has none.
fn unit(v: Complex, otherwise: Complex) -> Complex {
    let length = v.abs();
    if length > 0.0 {
        v.scale(1.0 / length)
    } else {
        otherwise
    }
}

/// A cubic's value at t.
fn evaluate(a: Complex, first: Complex, second: Complex, b: Complex, t: f64) -> Complex {
    let v = 1.0 - t;
    a.scale(v * v * v) + first.scale(3.0 * t * v * v) + second.scale(3.0 * t * t * v) + b.scale(t * t * t)
}

/// The last cubic fitted, and the buffer of where its samples fall on it.
struct Fit {
    first: Complex,
    second: Complex,
    t: Vec<f64>,
}
impl Default for Fit {
    fn default() -> Self {
        Self { first: Complex::ZERO, second: Complex::ZERO, t: Vec::new() }
    }
}
impl Fit {
    /// Fits a cubic through `points`' ends that leaves along `start` and
    /// arrives along `end` (Schneider's method: least squares for how far
    /// the two controls reach, then Newton steps for where each point falls).
    /// Returns the farthest any point lies from where it falls.
    fn cubic(&mut self, points: &[Complex], start: Complex, end: Complex) -> f64 {
        let (a, b) = (points[0], points[points.len() - 1]);
        self.t.clear();
        let mut arc = 0.0;
        self.t.push(0.0);
        for pair in points.windows(2) {
            arc += (pair[1] - pair[0]).abs();
            self.t.push(arc);
        }
        if arc > 0.0 {
            self.t.iter_mut().for_each(|t| *t /= arc);
        }
        let chord = b - a;
        let leave = unit(start, unit(chord, Complex::ZERO));
        let arrive = unit(Complex::ZERO - end, unit(Complex::ZERO - chord, Complex::ZERO));
        let turned = dot(leave, arrive);
        for _ in 0..REFINEMENTS {
            let (mut xx, mut xy, mut yy, mut rx, mut ry) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for (&t, &point) in self.t.iter().zip(points) {
                let v = 1.0 - t;
                let (b1, b2) = (3.0 * t * v * v, 3.0 * t * t * v);
                let rest = point - a.scale(v * v * v + b1) - b.scale(b2 + t * t * t);
                xx += b1 * b1;
                xy += b1 * b2 * turned;
                yy += b2 * b2;
                rx += b1 * dot(rest, leave);
                ry += b2 * dot(rest, arrive);
            }
            // A reach beyond the whole arc would make a loop the samples miss.
            let determinant = xx * yy - xy * xy;
            let (mut reach_a, mut reach_b) = (arc / 3.0, arc / 3.0);
            if determinant.abs() > 1e-12 {
                let (left, right) = ((yy * rx - xy * ry) / determinant, (xx * ry - xy * rx) / determinant);
                let floor = 1e-6 * chord.abs();
                if left > floor && right > floor && left <= arc && right <= arc {
                    (reach_a, reach_b) = (left, right);
                }
            }
            self.first = a + leave.scale(reach_a);
            self.second = b + arrive.scale(reach_b);
            let (c, d) = (self.first, self.second);
            for (t, &point) in self.t.iter_mut().zip(points) {
                let v = 1.0 - *t;
                let off = evaluate(a, c, d, b, *t) - point;
                let slope = (c - a).scale(3.0 * v * v) + (d - c).scale(6.0 * *t * v) + (b - d).scale(3.0 * *t * *t);
                let bend = (d - c.scale(2.0) + a).scale(6.0 * v) + (b - d.scale(2.0) + c).scale(6.0 * *t);
                let denominator = slope.norm_squared() + dot(off, bend);
                if denominator.abs() > 1e-12 {
                    *t = (*t - dot(off, slope) / denominator).clamp(0.0, 1.0);
                }
            }
        }
        let (c, d) = (self.first, self.second);
        self.t.iter().zip(points).map(|(&t, &point)| (evaluate(a, c, d, b, t) - point).abs()).fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    /// A series' points and directions at n samples, from (k, re, im) terms.
    fn series(terms: &[(f64, f64, f64)], n: usize) -> (Vec<Complex>, Vec<Complex>) {
        let around = |i: usize, power: bool| {
            terms.iter().fold(Complex::ZERO, |sum, &(k, re, im)| {
                let turn = Complex::from_angle(TAU * k * i as f64 / n as f64);
                let term = Complex::new(re, im) * turn;
                sum + if power { term * Complex::new(0.0, k) } else { term }
            })
        };
        ((0..n).map(|i| around(i, false)).collect(), (0..n).map(|i| around(i, true)).collect())
    }

    /// Many points along the segments, in order.
    fn trace(segments: &[Segment], start: Complex) -> Vec<Complex> {
        let mut points = vec![start];
        let mut from = start;
        for segment in segments {
            match *segment {
                Segment::Line { end } => {
                    points.extend((1..=50).map(|i| from + (end - from).scale(i as f64 / 50.0)));
                    from = end;
                }
                Segment::Cubic { first, second, end } => {
                    points.extend((1..=200).map(|i| evaluate(from, first, second, end, i as f64 / 200.0)));
                    from = end;
                }
            }
        }
        points
    }

    fn distance_to_polyline(q: Complex, line: &[Complex]) -> f64 {
        line.windows(2)
            .map(|pair| {
                let along = pair[1] - pair[0];
                let squared = along.norm_squared();
                let t = if squared > 0.0 { (dot(q - pair[0], along) / squared).clamp(0.0, 1.0) } else { 0.0 };
                (q - pair[0] - along.scale(t)).abs()
            })
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn a_polygon_stays_straight_lines() {
        let mut square = Vec::new();
        for (x, y, dx, dy) in [(0.0, 0.0, 1.0, 0.0), (100.0, 0.0, 0.0, 1.0), (100.0, 100.0, -1.0, 0.0), (0.0, 100.0, 0.0, -1.0)] {
            square.extend((0..100).map(|i| Complex::new(x + dx * i as f64, y + dy * i as f64)));
        }
        let tangents = vec![Complex::new(1.0, 0.0); square.len()];
        let segments = fit_loop(&square, &tangents, 0.1);
        let ends = [(100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (0.0, 0.0)];
        assert_eq!(segments, ends.map(|(x, y)| Segment::Line { end: Complex::new(x, y) }));
    }

    #[test]
    fn a_circle_takes_a_few_cubics_that_join_smoothly() {
        let (z, tangents) = series(&[(1.0, 50.0, 0.0)], 1024);
        let segments = fit_loop(&z, &tangents, 0.1);
        assert!(segments.len() <= 8, "{segments:?}");
        let mut last = z[0];
        for segment in &segments {
            let Segment::Cubic { first, second, end } = *segment else { panic!("{segments:?}") };
            // Controls lie on the tangents, which on a circle cross the radius at a right angle.
            assert!(dot(first - last, last).abs() < 1e-6 * 2500.0, "{segments:?}");
            assert!(dot(second - end, end).abs() < 1e-6 * 2500.0, "{segments:?}");
            last = end;
        }
        assert!((last - z[0]).abs() < 1e-9);
    }

    #[test]
    fn segments_and_samples_stay_within_the_tolerance_of_each_other_both_ways() {
        // Wiggly curves, as the fits make, with a fixed pseudo-random spectrum.
        let mut state = 12345u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5
        };
        for (reach, tolerance) in [(6, 0.1), (20, 0.1), (20, 0.5), (60, 0.25)] {
            let mut terms = vec![(1.0, 200.0, 0.0)];
            for k in 2..=reach {
                let size = 30.0 / k as f64;
                terms.push((k as f64, size * next(), size * next()));
                terms.push((-(k as f64), size * next(), size * next()));
            }
            let (z, tangents) = series(&terms, 4096);
            let segments = fit_loop(&z, &tangents, tolerance);
            assert!(segments.len() < z.len() / 2, "{}", segments.len());
            let drawn = trace(&segments, z[0]);
            let mut loop_points = z.clone();
            loop_points.push(z[0]);
            // The samples are under a pixel apart, and the drawn points denser.
            let slack = 0.05;
            for q in z.iter().step_by(7) {
                let d = distance_to_polyline(*q, &drawn);
                assert!(d <= tolerance + slack, "sample off by {d} at reach {reach}");
            }
            for q in drawn.iter().step_by(7) {
                let d = distance_to_polyline(*q, &loop_points);
                assert!(d <= tolerance + slack, "drawn point off by {d} at reach {reach}");
            }
        }
    }

    #[test]
    fn a_curve_that_returns_to_where_it_started_still_fits() {
        // Two points only, and a loop whose samples all coincide.
        let two = [Complex::new(0.0, 0.0), Complex::new(1.0, 1.0)];
        assert_eq!(fit_loop(&two, &[Complex::ZERO; 2], 0.1).len(), 2);
        let same = [Complex::new(3.0, 4.0); 16];
        let segments = fit_loop(&same, &[Complex::ZERO; 16], 0.1);
        assert_eq!(segments.last(), Some(&Segment::Line { end: same[0] }));
    }
}
