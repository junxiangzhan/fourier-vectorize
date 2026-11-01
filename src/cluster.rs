//! DBSCAN: density-based clustering. Groups points that are chained together
//! by neighbours closer than `eps`, so the number of clusters (here: the
//! number of separate outlines) is found, not chosen in advance.
use crate::{Complex, Error};
use std::collections::HashMap;

/// Returns one label per point: `Some(cluster)` numbered from 0 in order of
/// discovery, or `None` for noise. A point is a core point when at least
/// `min_points` points (itself included) lie within `eps` of it; clusters
/// are the core points reachable from each other plus their neighbours.
pub fn dbscan(
    points: &[Complex],
    eps: f64,
    min_points: usize,
) -> Result<Vec<Option<usize>>, Error> {
    if !eps.is_finite() || eps <= 0.0 || min_points == 0 {
        return Err(Error::InvalidParameter(
            "DBSCAN needs eps > 0 and min_points > 0",
        ));
    }
    if points
        .iter()
        .any(|p| !p.re.is_finite() || !p.im.is_finite())
    {
        return Err(Error::NonFinite);
    }
    // Bucket points into eps x eps cells: every neighbour of a point lies
    // in its own cell or one of the eight around it.
    let cell = |p: Complex| ((p.re / eps).floor() as i64, (p.im / eps).floor() as i64);
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, p) in points.iter().enumerate() {
        grid.entry(cell(*p)).or_default().push(i);
    }
    let neighbours = |i: usize| -> Vec<usize> {
        let (cx, cy) = cell(points[i]);
        let mut found = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(bucket) = grid.get(&(cx + dx, cy + dy)) {
                    found.extend(
                        bucket
                            .iter()
                            .filter(|&&j| (points[j] - points[i]).abs() <= eps),
                    );
                }
            }
        }
        found
    };
    let mut labels: Vec<Option<usize>> = vec![None; points.len()];
    let mut visited = vec![false; points.len()];
    let mut clusters = 0;
    for start in 0..points.len() {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let seeds = neighbours(start);
        if seeds.len() < min_points {
            continue; // noise for now; a later cluster may still claim it
        }
        // Grow a new cluster outward from this core point.
        let id = clusters;
        clusters += 1;
        labels[start] = Some(id);
        let mut queue = seeds;
        while let Some(j) = queue.pop() {
            if labels[j].is_none() {
                labels[j] = Some(id);
            }
            if visited[j] {
                continue;
            }
            visited[j] = true;
            let reach = neighbours(j);
            if reach.len() >= min_points {
                queue.extend(reach);
            }
        }
    }
    Ok(labels)
}
