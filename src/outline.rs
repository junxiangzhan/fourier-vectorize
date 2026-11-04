//! From a smoothed channel to closed outlines: k-means levels, then sub-pixel
//! iso-lines by marching squares.
use crate::{
    preprocess::{kmeans_1d, levels_of},
    Complex,
};
use std::collections::{HashMap, HashSet};

/// Areas smaller than this (pixels) are dropped.
pub(crate) const MIN_AREA: usize = 16;

/// Shoelace area in screen coordinates: > 0 means screen-clockwise.
pub(crate) fn signed_area(outline: &[Complex]) -> f64 {
    (0..outline.len())
        .map(|i| {
            let (a, b) = (outline[i], outline[(i + 1) % outline.len()]);
            a.re * b.im - a.im * b.re
        })
        .sum::<f64>()
        / 2.0
}

/// Iso-lines of `field` (w x h values at pixel centres) at `level` by
/// marching squares: inside means value >= level, and each crossing is
/// placed by linear interpolation along the cell edge, so outlines are
/// sub-pixel instead of following the pixel staircase. Outside the image
/// counts as outside, its crossings on the image border. Loops keep the
/// inside on their right: outer boundaries come out
/// screen-clockwise and holes counter-clockwise. Where a cell has two
/// inside corners on a diagonal, its centre (mean of the corners) decides
/// whether they connect.
pub(crate) fn contours(field: &[f64], w: usize, h: usize, level: f64) -> Vec<Vec<Complex>> {
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

/// The smooth outlines of one channel: its levels (k-means on the smoothed
/// values), the thresholds between them (midpoints), and for each level
/// j >= 1 the sub-pixel iso-lines at threshold j - 1. Areas under MIN_AREA
/// pixels are dropped (islands vanish, small holes close); the count dropped
/// is returned last.
#[allow(clippy::type_complexity)]
pub(crate) fn smooth_outlines(values: &[f64], w: usize, h: usize, count: usize) -> (Vec<f64>, Vec<f64>, Vec<Vec<Vec<Complex>>>, usize) {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
