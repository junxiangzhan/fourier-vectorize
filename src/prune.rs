//! Pruning: dropping coefficients and descending on the rest, which shrinks
//! the coefficients file at some cost in accuracy.
use crate::{
    fit::{chamfer, in_queue, Fit, KdTree, RATE},
    Complex, Error,
};

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
