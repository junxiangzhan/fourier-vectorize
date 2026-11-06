//! Pruning: dropping coefficients and descending on the rest, which shrinks
//! the coefficients file at some cost in accuracy.
use crate::{
    fit::{chamfer, in_queue, samples, Fit, KdTree, RATE},
    Complex, Error,
};

/// How far past its RMS budget a reduced curve may stray at any one point.
const WORST_REDUCED: f64 = 4.0;
/// Steps of descent after reducing a curve's harmonics. Measured on two
/// paintings, these get all that more steps do, and most of it comes early.
const REDUCED_STEPS: usize = 20;

/// Which coefficients pruning keeps.
#[derive(Clone, Copy, PartialEq)]
pub enum PruneBy {
    /// The largest |c_k| over all curves, whatever their k.
    Largest,
    /// Each curve's lowest harmonics.
    LowPass,
}
impl PruneBy {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Largest => "largest",
            Self::LowPass => "lowpass",
        }
    }
}

/// How to prune a fit's coefficients.
#[derive(Clone, Copy)]
pub struct Prune {
    /// Share of all coefficients to keep, in (0, 1). k = -1, 0, 1 always
    /// stay, so that no curve collapses to a point, and they can hold the
    /// result above this share.
    pub keep: f64,
    pub by: PruneBy,
    /// Cuts on the way to `keep`, each dropping the same fraction; 1 drops
    /// everything at once.
    pub rounds: usize,
    /// Steps of descent in all: half shared out over the rounds, half on the
    /// final set (with one round, all of them on the final set).
    pub steps: usize,
}

/// Whether the coefficient at index i of a curve with m harmonics may go.
fn prunable(i: usize, m: usize) -> bool {
    (i as i64 - m as i64).abs() > 1
}

/// Drops the smallest |c_k| over all curves until about `target` remain.
fn keep_largest(fits: &mut [Fit], live: &mut [Vec<bool>], target: usize) {
    let now: usize = live.iter().map(|l| l.iter().filter(|k| **k).count()).sum();
    let mut sizes: Vec<f64> = Vec::new();
    for (f, l) in fits.iter().zip(live.iter()) {
        sizes.extend((0..f.coefficients.len()).filter(|&i| l[i] && prunable(i, f.m)).map(|i| f.coefficients[i].abs()));
    }
    let drop = now.saturating_sub(target).min(sizes.len());
    if drop == 0 {
        return;
    }
    let (_, cut, _) = sizes.select_nth_unstable_by(drop - 1, f64::total_cmp);
    let cut = *cut;
    for (f, l) in fits.iter_mut().zip(live.iter_mut()) {
        let m = f.m;
        for (i, c) in f.coefficients.iter_mut().enumerate() {
            if l[i] && prunable(i, m) && c.abs() <= cut {
                l[i] = false;
                *c = Complex::ZERO;
            }
        }
    }
}

/// Keeps |k| <= top on each curve, with 2 top + 1 about `share` of its
/// coefficients.
fn low_pass(fits: &mut [Fit], live: &mut [Vec<bool>], share: f64) {
    for (f, l) in fits.iter_mut().zip(live.iter_mut()) {
        let m = f.m;
        let top = ((((2 * m + 1) as f64 * share - 1.0) / 2.0).round() as i64).max(1);
        for (i, c) in f.coefficients.iter_mut().enumerate() {
            if (i as i64 - m as i64).abs() > top {
                l[i] = false;
                *c = Complex::ZERO;
            }
        }
    }
}

/// `steps` of descent that move only the live coefficients; the loss recorded
/// is the chamfer loss after them.
fn descend_kept(outlines: &[Vec<Complex>], fits: &mut [Fit], live: &[Vec<bool>], steps: usize) -> Result<(), Error> {
    let mut order: Vec<usize> = (0..fits.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(fits[i].n));
    let current: &[Fit] = fits;
    let after = in_queue(&order, |i| {
        let (f, points) = (&current[i], &outlines[i]);
        let tree = KdTree::new(points);
        let mut c = f.coefficients.clone();
        for _ in 0..steps {
            let (_, gradient) = chamfer(&c, f.m, f.s, points, &tree)?;
            for ((v, g), l) in c.iter_mut().zip(gradient).zip(&live[i]) {
                if *l {
                    *v = *v - g.scale(RATE);
                }
            }
        }
        let loss = chamfer(&c, f.m, f.s, points, &tree)?.0;
        Ok((c, loss))
    })?;
    for (f, (c, loss)) in fits.iter_mut().zip(after) {
        f.coefficients = c;
        f.loss = loss;
    }
    Ok(())
}

/// The fewest harmonics of a curve that stay within `budget` of it, as an RMS
/// in pixels, and within `WORST_REDUCED` times that at every sample. Dropping
/// harmonics moves a curve evenly sampled in t by the RMS of what they sum to,
/// so the tail's energy gives the answer without drawing it.
fn fewest_harmonics(fit: &Fit, budget: f64) -> Result<usize, Error> {
    let (m, c) = (fit.m, &fit.coefficients);
    let (mut top, mut tail) = (m, 0.0);
    while top > 1 {
        let wider = tail + c[m - top].norm_squared() + c[m + top].norm_squared();
        if wider > budget * budget {
            break;
        }
        tail = wider;
        top -= 1;
    }
    let whole = samples(c, m, fit.s)?;
    while top < m {
        let mut kept = c.clone();
        kept[..m - top].fill(Complex::ZERO);
        kept[m + top + 1..].fill(Complex::ZERO);
        let worst = samples(&kept, m, fit.s)?.iter().zip(&whole).map(|(a, b)| (*a - *b).norm_squared()).fold(0.0, f64::max);
        if worst.sqrt() <= WORST_REDUCED * budget {
            break;
        }
        top = (top + top / 4 + 1).min(m);
    }
    Ok(top)
}

/// Gives each curve only the harmonics it needs, as `fewest_harmonics` finds,
/// then descends on those. Dropped coefficients end up exactly zero.
pub(crate) fn reduce_fits(outlines: &[Vec<Complex>], fits: &mut [Fit], budget: f64) -> Result<(), Error> {
    let order: Vec<usize> = (0..fits.len()).collect();
    let current: &[Fit] = fits;
    let tops = in_queue(&order, |i| fewest_harmonics(&current[i], budget))?;
    let mut live = Vec::with_capacity(fits.len());
    for (f, top) in fits.iter_mut().zip(tops) {
        let m = f.m;
        f.coefficients[..m - top].fill(Complex::ZERO);
        f.coefficients[m + top + 1..].fill(Complex::ZERO);
        live.push((0..2 * m + 1).map(|i| (i as i64 - m as i64).unsigned_abs() as usize <= top).collect());
    }
    descend_kept(outlines, fits, &live, REDUCED_STEPS)
}

/// Prunes each fit: cut, descend, and so on, as `prune` says. Dropped
/// coefficients end up exactly zero.
pub(crate) fn prune_fits(outlines: &[Vec<Complex>], fits: &mut [Fit], prune: Prune) -> Result<(), Error> {
    let mut live: Vec<Vec<bool>> = fits.iter().map(|f| vec![true; 2 * f.m + 1]).collect();
    let total: usize = live.iter().map(Vec::len).sum();
    let shared = prune.steps / 2;
    for round in 0..prune.rounds {
        let share = prune.keep.powf((round + 1) as f64 / prune.rounds as f64);
        match prune.by {
            PruneBy::Largest => keep_largest(fits, &mut live, (total as f64 * share).round() as usize),
            PruneBy::LowPass => low_pass(fits, &mut live, share),
        }
        let last = round + 1 == prune.rounds;
        let steps = shared * (round + 1) / prune.rounds - shared * round / prune.rounds + if last { prune.steps - shared } else { 0 };
        descend_kept(outlines, fits, &live, steps)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{fit::fit_all, FitKind};

    /// A ring with a large slow wave and a small fast one, at harmonics 5 and 40.
    fn wavy_ring(fast: f64) -> Vec<Complex> {
        (0..600)
            .map(|i| {
                let a = i as f64 / 600.0 * std::f64::consts::TAU;
                let r = 50.0 + 6.0 * (5.0 * a).sin() + fast * (40.0 * a).sin();
                Complex::new(60.0 + r * a.cos(), 60.0 + r * a.sin())
            })
            .collect()
    }

    fn top_of(f: &Fit) -> usize {
        (0..=f.m).rev().find(|k| f.coefficients[f.m + k].abs() > 0.0 || f.coefficients[f.m - k].abs() > 0.0).unwrap()
    }

    #[test]
    fn a_curve_keeps_only_the_harmonics_its_budget_needs() {
        let outlines = vec![wavy_ring(0.0), wavy_ring(0.5)];
        let mut fits = fit_all(&outlines, FitKind::Dft).unwrap();
        let (m, whole) = (fits[0].m, fits[0].coefficients.clone());
        let before: Vec<f64> = fits.iter().map(|f| f.loss).collect();
        reduce_fits(&outlines, &mut fits, 0.1).unwrap();
        // The smooth ring needs its slow wave and little else; a fast wave worth more than the budget stays.
        assert!(top_of(&fits[0]) < 20 && top_of(&fits[0]) >= 4, "{}", top_of(&fits[0]));
        assert!(top_of(&fits[1]) >= 39 && top_of(&fits[1]) < m / 2, "{}", top_of(&fits[1]));
        // What is dropped is exactly zero.
        let top = top_of(&fits[0]);
        assert!(fits[0].coefficients[..m - top].iter().all(|c| c.re == 0.0 && c.im == 0.0));
        // Descent brings what is kept closer to the outline than cutting alone leaves it.
        let mut cut = whole.clone();
        cut[..m - top].fill(Complex::ZERO);
        cut[m + top + 1..].fill(Complex::ZERO);
        let tree = KdTree::new(&outlines[0]);
        let (cut_loss, _) = chamfer(&cut, m, fits[0].s, &outlines[0], &tree).unwrap();
        assert!(fits[0].loss <= cut_loss, "{} vs {cut_loss}", fits[0].loss);
        // It lies no further from the outline than the budget adds. Descent slides
        // samples along the curve, so this is the distance to compare, not sample by sample.
        for (f, was) in fits.iter().zip(&before) {
            assert!(f.loss.sqrt() <= was.sqrt() + 0.1, "{} from {}", f.loss.sqrt(), was.sqrt());
        }
    }

    #[test]
    fn no_budget_is_too_small_or_too_large_for_a_curve() {
        let outlines = vec![wavy_ring(0.5)];
        // Too small to drop anything worth keeping, whatever is rounding-sized aside.
        let mut fits = fit_all(&outlines, FitKind::Dft).unwrap();
        let m = fits[0].m;
        reduce_fits(&outlines, &mut fits, 1e-12).unwrap();
        assert!(top_of(&fits[0]) >= 40 && top_of(&fits[0]) <= m);
        // So large that all could go: the curve still has k = -1, 0 and 1.
        let mut fits = fit_all(&outlines, FitKind::Dft).unwrap();
        fits[0].coefficients[m + 1] = Complex::new(50.0, 0.0);
        reduce_fits(&outlines, &mut fits, 1e6).unwrap();
        assert_eq!(top_of(&fits[0]), 1);
    }

    #[test]
    fn a_sharp_spike_keeps_more_than_the_rms_alone_would_allow() {
        // One corner far out on a long ring: small in RMS, large at that one place.
        let mut points = wavy_ring(0.0);
        points[300] = points[300] + Complex::new(-3.0, 0.0);
        let outlines = vec![points];
        let mut fits = fit_all(&outlines, FitKind::Dft).unwrap();
        let (m, c, s) = (fits[0].m, fits[0].coefficients.clone(), fits[0].s);
        let budget = 0.2;
        let top = fewest_harmonics(&fits[0], budget).unwrap();
        let mut kept = c.clone();
        kept[..m - top].fill(Complex::ZERO);
        kept[m + top + 1..].fill(Complex::ZERO);
        let (a, b) = (samples(&kept, m, s).unwrap(), samples(&c, m, s).unwrap());
        let worst = a.iter().zip(&b).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max);
        assert!(worst <= WORST_REDUCED * budget || top == m, "{worst} at {top}");
        reduce_fits(&outlines, &mut fits, budget).unwrap();
    }

    #[test]
    fn the_rounds_reach_the_same_share_and_the_steps_all_run() {
        // A wavy ring, so that coefficients have many different sizes.
        let points: Vec<Complex> = (0..600)
            .map(|i| {
                let a = i as f64 / 600.0 * std::f64::consts::TAU;
                let r = 50.0 + 6.0 * (5.0 * a).sin() + 2.0 * (17.0 * a).cos() + 0.5 * (40.0 * a).sin();
                Complex::new(60.0 + r * a.cos(), 60.0 + r * a.sin())
            })
            .collect();
        let outlines = vec![points];
        let start = fit_all(&outlines, FitKind::Dft).unwrap();
        let (total, before) = (start[0].coefficients.len(), start[0].loss);
        let run = |rounds: usize, steps: usize| {
            let mut fits = fit_all(&outlines, FitKind::Dft).unwrap();
            prune_fits(&outlines, &mut fits, Prune { keep: 0.2, by: PruneBy::Largest, rounds, steps }).unwrap();
            fits.remove(0)
        };
        let count = |f: &Fit| f.coefficients.iter().filter(|c| c.re != 0.0 || c.im != 0.0).count();
        let (once, many, none) = (run(1, 60), run(4, 60), run(1, 0));
        let target = (total as f64 * 0.2).round() as usize;
        for f in [&once, &many, &none] {
            assert!(count(f).abs_diff(target) <= 2, "{} vs {target}", count(f));
        }
        // Descent after the cut lowers the loss the cut raised, and the
        // recorded loss is the current one.
        assert!(none.loss > before);
        assert!(once.loss < none.loss);
        assert!(many.loss < none.loss);
    }
}
