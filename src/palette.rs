//! A joint colour table: colours picked by k-means in Lab, each painted as one
//! level over the ones before it, largest area first, the largest of all being
//! the canvas.
//!
//! A level only has to be exact where it lies over a colour painted before it.
//! Under the colours painted after it, it may reach as far as is cheapest, as
//! they will cover it: it fills any part of the canvas it encloses that holds
//! only later colours, and reaches a pixel further under them, so that no seam
//! shows between it and them. Where it does meet an earlier colour, the outline
//! goes where a pixel's nearest colour changes from one of the earlier ones to
//! one of the rest, found to a fraction of a pixel from the distances to them.
use crate::{
    outline::{contours, signed_area, MIN_AREA},
    Complex,
};

/// k-means looks at about this many pixels, evenly spread.
const SAMPLE: usize = 200_000;
const ROUNDS: usize = 12;
/// The smallest value either side of an outline, so that a pixel's side never
/// flips however its distances fall.
const SIDE: f64 = 0.01;

fn distance2(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

fn closest(p: &[f64; 3], centres: &[[f64; 3]]) -> usize {
    let mut best = (f64::INFINITY, 0);
    for (i, c) in centres.iter().enumerate() {
        let d = distance2(p, c);
        if d < best.0 {
            best = (d, i);
        }
    }
    best.1
}

/// Up to `k` colours for `points`, the same every time: k-means++ seeds from a
/// fixed sequence, then Lloyd's rounds. A colour nothing is nearest to is dropped.
pub(crate) fn kmeans(points: &[[f64; 3]], k: usize) -> Vec<[f64; 3]> {
    let sample: Vec<[f64; 3]> = points.iter().step_by((points.len() / SAMPLE).max(1)).copied().collect();
    if sample.is_empty() {
        return Vec::new();
    }
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut random = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut centres = vec![sample[sample.len() / 2]];
    let mut nearest: Vec<f64> = sample.iter().map(|p| distance2(p, &centres[0])).collect();
    while centres.len() < k {
        let total: f64 = nearest.iter().sum();
        if total <= 0.0 {
            break;
        }
        let mut target = random() * total;
        let pick = nearest
            .iter()
            .position(|d| {
                target -= d;
                target <= 0.0
            })
            .unwrap_or(sample.len() - 1);
        let centre = sample[pick];
        centres.push(centre);
        for (d, p) in nearest.iter_mut().zip(&sample) {
            *d = d.min(distance2(p, &centre));
        }
    }
    for _ in 0..ROUNDS {
        let mut sums = vec![([0.0; 3], 0usize); centres.len()];
        for p in &sample {
            let (sum, count) = &mut sums[closest(p, &centres)];
            for (s, v) in sum.iter_mut().zip(p) {
                *s += v;
            }
            *count += 1;
        }
        centres = sums.into_iter().filter(|(_, count)| *count > 0).map(|(sum, count)| sum.map(|s| s / count as f64)).collect();
    }
    centres
}

/// A table's colours in the order they are painted, and the outlines of each
/// level after the first, the canvas.
pub(crate) struct Stack {
    pub(crate) colours: Vec<[f64; 3]>,
    /// For each colour after the first: its outlines.
    pub(crate) outlines: Vec<Vec<Vec<Complex>>>,
    /// Areas under `MIN_AREA` pixels, given to a neighbour or dropped.
    pub(crate) dropped: usize,
}

/// The smallest area an outline may enclose, in pixels. Areas of fewer pixels
/// than `MIN_AREA` are gone by then, so what is left this small is a thin line,
/// and dropping it would uncover whatever colour was painted before it.
const SMALLEST_OUTLINE: f64 = 4.0;
/// How many times small areas are given away, each pass after the last one's
/// changes.
const PASSES: usize = 8;

/// The 4-connected areas of one colour each, as lists of pixels.
fn areas_of(colour: &[usize], w: usize, h: usize) -> Vec<Vec<usize>> {
    let mut seen = vec![false; w * h];
    let mut areas = Vec::new();
    for start in 0..w * h {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let (mut stack, mut area) = (vec![start], Vec::new());
        while let Some(i) = stack.pop() {
            area.push(i);
            let (x, y) = (i % w, i / w);
            let neighbours = [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)];
            for j in neighbours.into_iter().flatten() {
                if !seen[j] && colour[j] == colour[start] {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        areas.push(area);
    }
    areas
}

/// Gives each area of one colour under `MIN_AREA` pixels to the neighbouring
/// colour nearest its mean, so that dropping it later cannot uncover some
/// unrelated colour painted far earlier. Returns how many it gave away.
fn merge_small(colour: &mut [usize], labs: &[[f64; 3]], w: usize, h: usize, centres: &[[f64; 3]]) -> usize {
    let mut merged = 0;
    for _ in 0..PASSES {
        let before = merged;
        for area in areas_of(colour, w, h).into_iter().filter(|a| a.len() < MIN_AREA) {
            let own = colour[area[0]];
            let mut mean = [0.0; 3];
            for &i in &area {
                for (m, v) in mean.iter_mut().zip(&labs[i]) {
                    *m += v / area.len() as f64;
                }
            }
            let mut best = (f64::INFINITY, own);
            for &i in &area {
                let (x, y) = (i % w, i / w);
                let neighbours = [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)];
                for j in neighbours.into_iter().flatten() {
                    let other = colour[j];
                    if other != own && distance2(&mean, &centres[other]) < best.0 {
                        best = (distance2(&mean, &centres[other]), other);
                    }
                }
            }
            if best.1 != own {
                for &i in &area {
                    colour[i] = best.1;
                }
                merged += 1;
            }
        }
        if merged == before {
            break;
        }
    }
    merged
}

/// `labs` (w x h) painted in `centres`: each pixel takes its nearest colour,
/// areas too small to keep go to a neighbouring colour, and colours are
/// painted largest area first.
pub(crate) fn stack(labs: &[[f64; 3]], w: usize, h: usize, centres: &[[f64; 3]]) -> Stack {
    let mut nearest: Vec<usize> = labs.iter().map(|p| closest(p, centres)).collect();
    let merged = merge_small(&mut nearest, labs, w, h, centres);
    let mut areas = vec![0usize; centres.len()];
    for &c in &nearest {
        areas[c] += 1;
    }
    let mut order: Vec<usize> = (0..centres.len()).filter(|&c| areas[c] > 0).collect();
    order.sort_by_key(|&c| (std::cmp::Reverse(areas[c]), c));
    let mut rank = vec![0usize; centres.len()];
    for (position, &c) in order.iter().enumerate() {
        rank[c] = position;
    }
    let ranks: Vec<usize> = nearest.iter().map(|&c| rank[c]).collect();
    let painted: Vec<[f64; 3]> = order.iter().map(|&c| centres[c]).collect();
    let mut outlines = Vec::new();
    let mut dropped = merged;
    for position in 1..painted.len() {
        let inside = level_mask(&ranks, w, h, position);
        let field = side_field(&inside, &ranks, labs, w, h, &painted, position);
        let kept: Vec<Vec<Complex>> = contours(&field, w, h, 0.0)
            .into_iter()
            .filter(|l| {
                let keep = signed_area(l).abs() >= SMALLEST_OUTLINE;
                dropped += usize::from(!keep);
                keep
            })
            .collect();
        outlines.push(kept);
    }
    Stack { colours: painted, outlines, dropped }
}

/// Which pixels the level of the colour painted at `position` covers: its own,
/// every part of the rest of the canvas without an earlier colour (4-connected),
/// and a pixel further under later colours.
fn level_mask(ranks: &[usize], w: usize, h: usize, position: usize) -> Vec<bool> {
    let own: Vec<bool> = ranks.iter().map(|&r| r == position).collect();
    let mut inside = own.clone();
    let mut seen = own.clone();
    let mut part = Vec::new();
    for start in 0..w * h {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let (mut stack, mut earlier) = (vec![start], false);
        part.clear();
        while let Some(i) = stack.pop() {
            part.push(i);
            earlier |= ranks[i] < position;
            let (x, y) = (i % w, i / w);
            let neighbours = [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)];
            for j in neighbours.into_iter().flatten() {
                if !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        if !earlier {
            for &i in &part {
                inside[i] = true;
            }
        }
    }
    let grown: Vec<bool> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            inside[i] || (ranks[i] > position && ((x > 0 && inside[i - 1]) || (x + 1 < w && inside[i + 1]) || (y > 0 && inside[i - w]) || (y + 1 < h && inside[i + w])))
        })
        .collect();
    grown
}

/// A field that is positive on `inside` and negative elsewhere. Next to an
/// outline it is how much nearer the pixel is to a colour of its neighbourhood
/// painted at or after `position` than to an earlier one of its neighbourhood,
/// so its outline at 0 falls where one changes into the other. Only colours the
/// pixels around it were given count: a colour taken from a sliver too small to
/// keep would otherwise move the outline to that sliver's pixel edges.
fn side_field(inside: &[bool], ranks: &[usize], labs: &[[f64; 3]], w: usize, h: usize, painted: &[[f64; 3]], position: usize) -> Vec<f64> {
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let side = if inside[i] { 1.0 } else { -1.0 };
            let (mut edge, mut earlier, mut rest) = (false, f64::INFINITY, f64::INFINITY);
            for ny in y.saturating_sub(1)..(y + 2).min(h) {
                for nx in x.saturating_sub(1)..(x + 2).min(w) {
                    let j = ny * w + nx;
                    edge |= inside[j] != inside[i];
                    let d = distance2(&labs[i], &painted[ranks[j]]);
                    if ranks[j] < position {
                        earlier = earlier.min(d);
                    } else {
                        rest = rest.min(d);
                    }
                }
            }
            if !edge || earlier.is_infinite() || rest.is_infinite() {
                return side;
            }
            let margin = earlier.sqrt() - rest.sqrt();
            if inside[i] {
                margin.max(SIDE)
            } else {
                margin.min(-SIDE)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_finds_the_colours_there_are_and_no_more() {
        let colours = [[20.0, 10.0, -5.0], [70.0, -20.0, 30.0], [50.0, 40.0, 0.0]];
        let points: Vec<[f64; 3]> = (0..3000).map(|i| colours[i % 3]).collect();
        let mut found = kmeans(&points, 8);
        found.sort_by(|a, b| a[0].total_cmp(&b[0]));
        assert_eq!(found.len(), 3);
        let mut wanted = colours.to_vec();
        wanted.sort_by(|a, b| a[0].total_cmp(&b[0]));
        for (a, b) in found.iter().zip(&wanted) {
            assert!(distance2(a, b) < 1e-9, "{a:?} {b:?}");
        }
        assert_eq!(kmeans(&points, 8), kmeans(&points, 8));
    }

    /// A ground, a square on it, and a disc inside the square, in three colours.
    fn picture() -> (Vec<[f64; 3]>, usize, usize) {
        let (w, h) = (80, 60);
        let labs = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
                if (x - 25.0).hypot(y - 25.0) < 6.0 {
                    [80.0, 0.0, 0.0]
                } else if (10.0..40.0).contains(&x) && (10.0..40.0).contains(&y) {
                    [40.0, 30.0, 10.0]
                } else {
                    [60.0, -10.0, 20.0]
                }
            })
            .collect();
        (labs, w, h)
    }

    #[test]
    fn colours_go_largest_first_and_a_level_fills_holes_of_later_ones() {
        let (labs, w, h) = picture();
        let centres = [[80.0, 0.0, 0.0], [40.0, 30.0, 10.0], [60.0, -10.0, 20.0]];
        let stack = stack(&labs, w, h, &centres);
        // The ground is the canvas, then the square, then the disc.
        assert_eq!(stack.colours, [[60.0, -10.0, 20.0], [40.0, 30.0, 10.0], [80.0, 0.0, 0.0]]);
        // The square's level covers the disc too, so it has no hole.
        assert_eq!(stack.outlines.iter().map(Vec::len).collect::<Vec<_>>(), [1, 1]);
        let ranks: Vec<usize> = labs.iter().map(|p| closest(p, &stack.colours)).collect();
        let mask = level_mask(&ranks, w, h, 1);
        assert!(mask[25 * w + 25] && !mask[5 * w + 5] && !mask[25 * w + 60]);
    }

    #[test]
    fn an_outline_falls_where_the_nearest_colour_changes() {
        // Two colours meeting along a smooth ramp: the change is at x = 20.3.
        let (w, h) = (40, 20);
        let (dark, light) = ([30.0, 0.0, 0.0], [70.0, 0.0, 0.0]);
        let labs: Vec<[f64; 3]> = (0..w * h).map(|i| [50.0 + ((i % w) as f64 + 0.5 - 20.3) * 4.0, 0.0, 0.0]).collect();
        let stack = stack(&labs, w, h, &[dark, light]);
        let outline = &stack.outlines[0][0];
        let inner: Vec<f64> = outline.iter().filter(|p| p.im > 1.0 && p.im < h as f64 - 1.0).map(|p| p.re).collect();
        assert!(!inner.is_empty());
        for x in inner {
            assert!((x - 20.3).abs() < 0.05 || x > w as f64 - 0.6, "{x}");
        }
    }
}
