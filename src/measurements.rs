//! Part 3-II(a): choose a pixel row that crosses the rotated star's
//! shallowest edge near one end, and plot coverage along that row.
use crate::{experiments::star_path, presentation::chart_svg};
use multimedia::{winding_at, Error, Image, Point2};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

/// x where segment a-b crosses the horizontal line y (same half-open rule as
/// the rasterizer), or None.
fn crossing(a: Point2, b: Point2, y: f64) -> Option<f64> {
    let (a, b) = if a.y() < b.y() { (a, b) } else { (b, a) };
    if y < a.y() || y >= b.y() {
        None
    } else {
        Some(a.x() + (b.x() - a.x()) * ((y - a.y()) / (b.y() - a.y())))
    }
}

pub fn run(output: &Path, size: usize, images: &[(usize, Image<f64>, f64)]) -> Result<(), Error> {
    let shape = star_path(size, 5.0_f64.to_radians()).flatten(0)?;
    let vertices = &shape.subpaths[0].vertices;
    let low = &images
        .iter()
        .find(|(k, _, _)| *k == 1)
        .ok_or(Error::InvalidParameter("missing k=1 image"))?
        .1;
    let high = &images
        .iter()
        .find(|(k, _, _)| *k == 8)
        .ok_or(Error::InvalidParameter("missing k=8 image"))?
        .1;
    // Shallowest edge = smallest |dy/dx|. Along such an edge one pixel row
    // spans many columns, which is where the staircase is longest.
    let edge = (0..vertices.len())
        .min_by(|&i, &j| {
            let a = vertices[i];
            let b = vertices[(i + 1) % vertices.len()];
            let c = vertices[j];
            let d = vertices[(j + 1) % vertices.len()];
            ((b.y() - a.y()) / (b.x() - a.x()))
                .abs()
                .total_cmp(&((d.y() - c.y()) / (d.x() - c.x())).abs())
        })
        .unwrap();
    let a = vertices[edge];
    let b = vertices[(edge + 1) % vertices.len()];
    let slope = ((b.y() - a.y()) / (b.x() - a.x())).abs();
    // The row crosses the edge over about 1/slope columns; plot that much
    // on each side of the crossing plus a small margin.
    let preferred_half = (1.0 / slope).ceil() + 3.0;
    let mut selection = None;
    // Try rows near either end of the edge only. In the middle of the edge
    // the star is filled on both sides (core and spike), so there is no
    // inside/outside transition to plot.
    for fraction in [0.10, 0.15, 0.20, 0.24, 0.90, 0.85, 0.80, 0.76] {
        let y = (a.y() + (b.y() - a.y()) * fraction).floor();
        if y < 0.0 || y >= size as f64 {
            continue;
        }
        let row = y as usize;
        let sy = y + 0.5;
        let Some(x) = crossing(a, b, sy) else {
            continue;
        };
        let actual_fraction = (sy - a.y()) / (b.y() - a.y());
        if actual_fraction.min(1.0 - actual_fraction) > 0.25 {
            continue;
        }
        // Keep the plotted stretch clear of other edges crossing this row.
        let nearest = (0..vertices.len())
            .filter(|i| *i != edge)
            .filter_map(|i| crossing(vertices[i], vertices[(i + 1) % vertices.len()], sy))
            .map(|other| (other - x).abs())
            .fold(f64::INFINITY, f64::min);
        let half = preferred_half.min((nearest - 2.0) * 0.75);
        if half < 3.0 {
            continue;
        }
        let start = (x - half).floor().clamp(0.0, size as f64) as usize;
        let end = (x + half).ceil().clamp(0.0, size as f64) as usize;
        if start >= end {
            continue;
        }
        // The k=8 stretch must reach both 0 and 1 and contain a real ramp.
        let values = &high.row(row).unwrap()[start..end];
        let partial = values.iter().filter(|v| **v > 0.0 && **v < 1.0).count();
        if !values.contains(&0.0) || !values.contains(&1.0) || partial < 3 {
            continue;
        }
        // Probe just off each side of the edge: exactly one side must be
        // inside the star, otherwise this is a filled-on-both-sides stretch.
        let delta = b - a;
        let normal = multimedia::Vector::from_array([-delta[(1, 0)], delta[(0, 0)]]) / delta.norm();
        let on_edge = Point2::new(x, sy);
        let side_a = winding_at(&shape, on_edge + normal * 0.7)? != 0;
        let side_b = winding_at(&shape, on_edge - normal * 0.7)? != 0;
        if side_a == side_b {
            continue;
        }
        selection = Some((row, start, end, x, actual_fraction, nearest, partial));
        break;
    }
    let (row, start, end, x, fraction, nearest, partial) = selection.ok_or(
        Error::InvalidParameter("no clean near-end row; increase canvas size"),
    )?;
    // Coverage is read from the unquantized f64 images, not from the BMPs.
    let mut csv = BufWriter::new(File::create(output.join("row_samples.csv"))?);
    writeln!(csv, "x,y,coverage_k1,coverage_k8")?;
    let mut series1 = Vec::new();
    let mut series8 = Vec::new();
    for column in start..end {
        let v1 = *low.get(column, row).unwrap();
        let v8 = *high.get(column, row).unwrap();
        writeln!(csv, "{column},{row},{v1:.9},{v8:.9}")?;
        series1.push((column as f64, v1));
        series8.push((column as f64, v8));
    }
    csv.flush()?;
    chart_svg(
        &output.join("row_plot.svg"),
        "Along-row transition near shallow edge endpoint",
        "Pixel column x",
        "Coverage",
        &[("k=1", "#1565c0", series1), ("k=8", "#e66b18", series8)],
    )?;
    std::fs::write(output.join("row_selection.json"),format!("{{\"edge_index\":{edge},\"edge_start\":[{:.12},{:.12}],\"edge_end\":[{:.12},{:.12}],\"row\":{row},\"x_start\":{start},\"x_end_exclusive\":{end},\"crossing_x\":{x:.12},\"fraction_from_start\":{fraction:.12},\"nearest_other_crossing_distance\":{nearest:.12},\"partial_pixels_k8\":{partial},\"source\":\"unquantized coverage\"}}\n",a.x(),a.y(),b.x(),b.y()))?;
    std::fs::write(output.join("measurements_answers.md"),format!("# Pixel row and cost\n\nRow {row}, columns {start} through {}, crosses shallow edge {edge} at fraction {fraction:.4} from its start. Its nearest other crossing is {nearest:.3} pixels away; the selected stretch excludes that crossing. The two sides of this edge are not both filled.\n\nk=1 changes abruptly between 0 and 1. k=8 has {partial} partially covered pixels across a gradual transition, because averaging integrates many subpositions within each pixel. This is an along-row measurement, not a perpendicular one-pixel probe.\n\nThe timing CSV records medians of three fresh renders per k on {size} x {size}, measured with a monotonic clock. Scaling, fill, allocation and averaging are included; image and chart output are excluded. The high-resolution image and its average traverse O(k^2) samples, while scanline-edge intersection work grows with the number of rows O(k). Larger-k costs therefore tend toward quadratic growth; k=1 omits the averaging pass and fixed overhead affects the smaller cases. Actual values and trial variation are retained.\n",end-1))?;
    Ok(())
}
