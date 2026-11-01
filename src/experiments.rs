//! The Part 1-3 experiments. Each function builds its scene, computes every
//! number and image with the library, and writes the raw data (CSV/JSON),
//! the images (BMP) and a short text answer into the output directory.
use crate::presentation::{chart_svg, geometry_svg, write_points};
use multimedia::{
    winding_at, Color, CubicBezier, Error, Image, Path as ShapePath, Point2, Rasterizer, Segment,
    Subpath,
};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

/// A 3 x 2 image of six known colors, so the audit can check that the BMP
/// encoder's channel order and row order decode correctly.
pub fn diagnostic(output: &Path) -> Result<(), Error> {
    let mut image = Image::new(3, 2, Color::rgb(0.0, 0.0, 0.0))?;
    let colors = [
        Color::rgb(1.0, 0.0, 0.0),
        Color::rgb(0.0, 1.0, 0.0),
        Color::rgb(0.0, 0.0, 1.0),
        Color::rgb(1.0, 1.0, 0.0),
        Color::rgb(0.0, 1.0, 1.0),
        Color::rgb(1.0, 0.0, 1.0),
    ];
    for (i, color) in colors.into_iter().enumerate() {
        *image.get_mut(i % 3, i / 3).unwrap() = color;
    }
    image.save_bmp(output.join("diagnostic.bmp"))
}

/// Part 1-1 and 1-2: the overlay at a shallow depth and the convergence table.
pub fn curves(output: &Path, size: usize) -> Result<(), Error> {
    // Control points as fractions of the canvas, so the arch spans most of it.
    let point = |x: f64, y: f64| Point2::new(x * size as f64, y * size as f64);
    let curve = CubicBezier::new([
        point(0.07, 0.82),
        point(0.12, 0.05),
        point(0.87, 0.05),
        point(0.93, 0.82),
    ]);
    // Construction A, sampled with < 0.1 px between consecutive points so
    // the measured gap is not dominated by our own sampling of A.
    let samples = curve.sample_dense(0.1)?;
    // Construction B at depth 2 (4 segments): still visibly different from A.
    let shallow = curve.flatten(2)?;
    write_points(&output.join("curve_controls.csv"), curve.controls())?;
    write_points(&output.join("curve_formula.csv"), &samples)?;
    write_points(&output.join("curve_depth2.csv"), &shallow.vertices)?;
    geometry_svg(
        &output.join("curve_overlay.svg"),
        size,
        "Formula (blue) / midpoint depth 2 (orange)",
        &[(&samples, "#1565c0"), (&shallow.vertices, "#e66b18")],
        &[curve.controls()],
    )?;
    let max_spacing = samples
        .windows(2)
        .map(|p| p[0].distance(p[1]))
        .fold(0.0_f64, f64::max);
    let mut table = BufWriter::new(File::create(output.join("convergence.csv"))?);
    writeln!(table, "depth,segments,max_gap_px,ratio_to_previous")?;
    let mut previous = None;
    let mut values = Vec::new();
    let mut threshold = None;
    // Two extra depths show the deep convergence trend after passing 0.5 px.
    for depth in 0..=multimedia::geometry::MAX_SUBDIVISION_DEPTH {
        let polyline = curve.flatten(depth)?;
        // Gap in the required direction: for every formula point, distance to
        // the nearest polyline segment; report the largest.
        let mut gap = 0.0_f64;
        for &p in &samples {
            gap = gap.max(polyline.distance_to(p)?);
        }
        // Ratio to the previous depth shows the per-level rule (about 1/4).
        let ratio = previous.map(|p: f64| if p == 0.0 { 0.0 } else { gap / p });
        writeln!(
            table,
            "{depth},{},{gap:.12},{}",
            polyline.vertices.len() - 1,
            ratio.map(|v| format!("{v:.12}")).unwrap_or_default()
        )?;
        values.push((depth as f64, gap));
        previous = Some(gap);
        if gap < 0.5 && threshold.is_none() {
            threshold = Some(depth);
        }
        if let Some(first) = threshold {
            if depth >= first + 2 {
                break;
            }
        }
    }
    table.flush()?;
    let threshold = threshold.ok_or(Error::InvalidParameter(
        "subdivision resource limit reached before convergence",
    ))?;
    chart_svg(
        &output.join("convergence.svg"),
        "Curve-to-polyline maximum gap",
        "Subdivision depth",
        "Gap (pixels)",
        &[("measured gap", "#1565c0", values)],
    )?;
    std::fs::write(output.join("curve_summary.json"), format!("{{\"canvas\":{size},\"shallow_depth\":2,\"formula_samples\":{},\"max_spacing_px\":{max_spacing:.12},\"first_depth_below_half_pixel\":{threshold},\"error_direction\":\"formula points to nearest polyline segments\"}}\n",samples.len()))?;
    std::fs::write(output.join("curve_answers.md"), format!("# Curve convergence\n\nCanvas: {size} x {size}. Formula samples: {}. Maximum adjacent spacing: {max_spacing:.9} px. First depth with maximum curve-to-polyline gap below 0.5 px: {threshold}.\n\nThe measured ratios approach 0.25: doubling the number of segments reduces the maximum error by about four. Shallow depths deviate more because a long chord does not describe local curvature as well. The convergence table contains the actual ratios, rather than assuming this law.\n\nMidpoint subdivision also permits locally adaptive refinement: subdivide only intervals whose control polygon is not flat enough. The construction offers this capability; this experiment deliberately uses uniform depth to measure convergence.\n",samples.len()))?;
    Ok(())
}

/// Part 1-3: an open path of three cubics with two joins. Join 0 is smooth
/// (direction lines collinear); join 1 is a corner (not collinear). Two
/// variants differ only in the length of the outgoing handle at join 0.
pub fn joins(output: &Path, size: usize) -> Result<(), Error> {
    let p = |x: f64, y: f64| Point2::new(x * size as f64, y * size as f64);
    let first = [p(0.06, 0.72), p(0.12, 0.28), p(0.24, 0.30), p(0.35, 0.40)];
    // third[0] is join 1. Its outgoing direction (third[1] - third[0]) is not
    // along the incoming one (third[0] - second[2]), which makes the corner.
    let third = [p(0.68, 0.48), p(0.80, 0.64), p(0.89, 0.18), p(0.94, 0.28)];
    let mut controls_file = BufWriter::new(File::create(output.join("joins_controls.csv"))?);
    let mut metrics = BufWriter::new(File::create(output.join("joins_metrics.csv"))?);
    writeln!(controls_file, "variant,segment,control,x,y")?;
    writeln!(
        metrics,
        "variant,join,incoming_length,outgoing_length,tangent_cross,tangent_angle_deg"
    )?;
    for (name, factor) in [("equal", 1.0), ("unequal", 3.0)] {
        // Join 0 = first[3]. Its outgoing handle continues the incoming
        // direction line (first[3] - first[2]) straight through the join, so
        // the three points stay collinear; `factor` sets only its length.
        let second = [
            first[3],
            first[3] + (first[3] - first[2]) * factor,
            p(0.56, 0.76),
            third[0],
        ];
        let controls = [first, second, third];
        let mut sub = Subpath::new(first[0]);
        for (i, c) in controls.iter().enumerate() {
            sub.cubic_to(c[1], c[2], c[3]);
            for (j, point) in c.iter().enumerate() {
                writeln!(
                    controls_file,
                    "{name},{i},{j},{:.12},{:.12}",
                    point.x(),
                    point.y()
                )?;
            }
        }
        let path = ShapePath::new(vec![sub]);
        let flat = path.flatten(7)?;
        write_points(
            &output.join(format!("joins_{name}.csv")),
            &flat.subpaths[0].vertices,
        )?;
        let segments: Vec<_> = controls
            .iter()
            .map(|c| CubicBezier::new(*c).sample_dense(0.1))
            .collect::<Result<_, _>>()?;
        let lines: Vec<_> = segments
            .iter()
            .zip(["#1565c0", "#0a875a", "#9b45bd"])
            .map(|(line, color)| (line.as_slice(), color))
            .collect();
        let control_lines: Vec<_> = controls.iter().map(|c| c.as_slice()).collect();
        geometry_svg(
            &output.join(format!("joins_{name}.svg")),
            size,
            &format!("Open path / {name} handles"),
            &lines,
            &control_lines,
        )?;
        // Measure each join: direction-line lengths, and the angle between
        // incoming and outgoing tangents (0 degrees = smooth, else a corner).
        for join in 0..2 {
            let incoming = controls[join][3] - controls[join][2];
            let outgoing = controls[join + 1][1] - controls[join + 1][0];
            let cross = incoming[(0, 0)] * outgoing[(1, 0)] - incoming[(1, 0)] * outgoing[(0, 0)];
            let angle = (incoming.dot(outgoing) / (incoming.norm() * outgoing.norm()))
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees();
            writeln!(
                metrics,
                "{name},{join},{:.12},{:.12},{cross:.12},{angle:.9}",
                incoming.norm(),
                outgoing.norm()
            )?;
        }
    }
    controls_file.flush()?;
    metrics.flush()?;
    std::fs::write(output.join("joins_answers.md"),"# Joins\n\nJoin 0 remains geometrically smooth in both versions: its incoming and outgoing tangents point along the same line. Unequal handles change tangent magnitude, not direction. Equal handle lengths additionally match the first derivatives when both segments use the same parameter interval (C1 continuity); collinearity with the same traversal direction gives G1 continuity.\n\nJoin 1 is a corner because its incoming and outgoing direction lines are not collinear; the metrics table records the turning angle. The unequal version changes only the outgoing handle at join 0, making it three times longer.\n")?;
    Ok(())
}

/// Part 2-2: a five-pointed star as ONE self-intersecting closed path of five
/// vertices, rotated by `angle` about the canvas center (Part 3 uses 5°).
pub fn star_path(size: usize, angle: f64) -> ShapePath {
    let center = Point2::new(size as f64 / 2.0, size as f64 / 2.0);
    // Five points evenly spaced on a circle, starting at the top.
    let vertices: Vec<_> = (0..5)
        .map(|i| {
            let a = -std::f64::consts::FRAC_PI_2 + i as f64 * std::f64::consts::TAU / 5.0;
            Point2::new(
                center.x() + 0.42 * size as f64 * a.cos(),
                center.y() + 0.42 * size as f64 * a.sin(),
            )
            .rotated_about(center, angle)
        })
        .collect();
    // "Every other one" order: each step skips one vertex, so the path wraps
    // the center twice and its center winding is +2 or -2, not 1.
    let order = [0, 2, 4, 1, 3];
    let mut sub = Subpath::new(vertices[order[0]]);
    for i in order.into_iter().skip(1) {
        sub.line_to(vertices[i]);
    }
    sub.close();
    ShapePath::new(vec![sub])
}
/// Writes coverage as a BMP: black shape on white, gray for partial coverage.
pub fn save_coverage(image: &Image<f64>, path: &Path) -> Result<(), Error> {
    image
        .map(|a| Color::rgb(1.0, 1.0, 1.0).mix(Color::rgb(0.0, 0.0, 0.0), *a))?
        .save_bmp(path)
}
/// Part 2-2: fill the unrotated star and report the winding at its center.
pub fn star(output: &Path, size: usize) -> Result<(), Error> {
    // The star has only line segments, so flattening depth does not matter.
    let path = star_path(size, 0.0).flatten(0)?;
    let image = Rasterizer::new(size, size)?.render(&path, 1)?;
    let center = Point2::new(size as f64 / 2.0, size as f64 / 2.0);
    let winding = winding_at(&path, center)?;
    save_coverage(&image, &output.join("star.bmp"))?;
    write_points(
        &output.join("star_vertices.csv"),
        &path.subpaths[0].vertices,
    )?;
    std::fs::write(output.join("star_summary.json"),format!("{{\"canvas\":{size},\"center_x\":{},\"center_y\":{},\"winding\":{winding},\"convention\":\"screen-clockwise positive\",\"vertex_order\":[0,2,4,1,3]}}\n",center.x(),center.y()))?;
    Ok(())
}

/// A circle as four cubic Bézier quarter arcs. Each arc's two direction points
/// sit 0.5523 * r along the tangent from its end points. With y down, the
/// order right -> bottom -> left -> top is screen-clockwise.
pub fn circle(center: Point2, radius: f64) -> Subpath {
    let x = center.x();
    let y = center.y();
    let h = 0.5523 * radius;
    let mut sub = Subpath::new(Point2::new(x + radius, y));
    sub.cubic_to(
        Point2::new(x + radius, y + h),
        Point2::new(x + h, y + radius),
        Point2::new(x, y + radius),
    );
    sub.cubic_to(
        Point2::new(x - h, y + radius),
        Point2::new(x - radius, y + h),
        Point2::new(x - radius, y),
    );
    sub.cubic_to(
        Point2::new(x - radius, y - h),
        Point2::new(x - h, y - radius),
        Point2::new(x, y - radius),
    );
    sub.cubic_to(
        Point2::new(x + h, y - radius),
        Point2::new(x + radius, y - h),
        Point2::new(x + radius, y),
    );
    sub.close();
    sub
}
/// All points of a sub-path, sorted, ignoring order and direction. Equal
/// results prove that reversing a contour moved no point.
fn positions(sub: &Subpath) -> Vec<Point2> {
    let mut points = vec![sub.start()];
    for segment in sub.segments() {
        match *segment {
            Segment::LineTo(end) => points.push(end),
            Segment::CubicTo {
                control1,
                control2,
                end,
            } => points.extend([control1, control2, end]),
        }
    }
    points.sort_by(|a, b| {
        a.x()
            .total_cmp(&b.x())
            .then_with(|| a.y().total_cmp(&b.y()))
    });
    points
}
/// Part 2-3: outer and inner circle as two sub-paths, rendered twice. The
/// only change between the two renders is the inner circle's direction.
pub fn rings(output: &Path, size: usize) -> Result<(), Error> {
    let center = Point2::new(size as f64 / 2.0, size as f64 / 2.0);
    let outer = circle(center, 0.35 * size as f64);
    let inner = circle(center, 0.15 * size as f64);
    let reversed = inner.reversed();
    let unchanged = positions(&inner) == positions(&reversed);
    if !unchanged {
        return Err(Error::InvalidParameter("reversal changed circle geometry"));
    }
    let raster = Rasterizer::new(size, size)?;
    let mut windings = Vec::new();
    let mut controls = BufWriter::new(File::create(output.join("ring_controls.csv"))?);
    writeln!(controls, "variant,subpath,point,x,y")?;
    for (name, inner) in [("same", inner), ("opposite", reversed)] {
        let path = ShapePath::new(vec![outer.clone(), inner]);
        for (j, sub) in path.subpaths().iter().enumerate() {
            for (i, point) in positions(sub).iter().enumerate() {
                writeln!(
                    controls,
                    "{name},{j},{i},{:.12},{:.12}",
                    point.x(),
                    point.y()
                )?;
            }
        }
        // Whole pipeline: Bézier circles -> Part 1 mid-point flattening
        // (depth 7, 128 segments per quarter) -> Part 2 non-zero fill.
        let flat = path.flatten(7)?;
        save_coverage(
            &raster.render(&flat, 1)?,
            &output.join(format!("ring_{name}.bmp")),
        )?;
        windings.push(winding_at(&flat, center)?);
    }
    controls.flush()?;
    std::fs::write(output.join("ring_summary.json"),format!("{{\"canvas\":{size},\"outer_radius\":{},\"inner_radius\":{},\"segments_per_circle\":4,\"handle_ratio\":0.5523,\"flatten_depth\":7,\"same_winding\":{},\"opposite_winding\":{},\"positions_unchanged\":{unchanged}}}\n",0.35*size as f64,0.15*size as f64,windings[0],windings[1]))?;
    std::fs::write(output.join("ring_answers.md"),format!("# Direction and the letter O\n\nWith both contours screen-clockwise, the center winding is {} and it is filled. Reversing only the inner contour gives winding {} and a hole. The coordinates are unchanged; direction is the only changed input.\n\nA font using non-zero winding must retain directed contours: the inner contour of O must oppose the outer contour for the counter to remain empty. A list of undirected outlines is insufficient to determine this fill.\n",windings[0],windings[1]))?;
    Ok(())
}

/// Writes the unquantized coverage of every pixel as CSV (x, y, coverage).
pub fn write_coverage(path: &Path, image: &Image<f64>) -> Result<(), Error> {
    let mut file = BufWriter::new(File::create(path)?);
    writeln!(file, "x,y,coverage")?;
    for y in 0..image.height() {
        for (x, value) in image.row(y).unwrap().iter().enumerate() {
            writeln!(file, "{x},{y},{value:.9}")?;
        }
    }
    file.flush()?;
    Ok(())
}

/// Part 3-I and 3-II(b): the star rotated by 5°, rendered at k = 1, 2, 4, 8,
/// with three timed renders per k. Returns the images for the row plot.
pub fn antialias(output: &Path, size: usize) -> Result<Vec<(usize, Image<f64>, f64)>, Error> {
    let path = star_path(size, 5.0_f64.to_radians()).flatten(0)?;
    let vertices = &path.subpaths[0].vertices;
    // Guard: after rotation no edge may be exactly horizontal or vertical,
    // since an axis-aligned edge never shows the staircase.
    let min_dx = (0..vertices.len())
        .map(|i| (vertices[(i + 1) % vertices.len()].x() - vertices[i].x()).abs())
        .fold(f64::INFINITY, f64::min);
    let min_dy = (0..vertices.len())
        .map(|i| (vertices[(i + 1) % vertices.len()].y() - vertices[i].y()).abs())
        .fold(f64::INFINITY, f64::min);
    if min_dx < 1e-9 || min_dy < 1e-9 {
        return Err(Error::InvalidParameter(
            "rotated star still has an axis-aligned edge",
        ));
    }
    write_points(&output.join("rotated_star_vertices.csv"), vertices)?;
    let raster = Rasterizer::new(size, size)?;
    let mut timings = BufWriter::new(File::create(output.join("timing.csv"))?);
    let mut trials = BufWriter::new(File::create(output.join("timing_trials.csv"))?);
    writeln!(timings, "k,width,height,median_render_ms")?;
    writeln!(trials, "k,trial,render_ms")?;
    let mut results = Vec::new();
    let mut chart = Vec::new();
    for k in [1, 2, 4, 8] {
        let mut elapsed = Vec::new();
        let mut result = None;
        // Time only the render (scale, allocate, fill, average), not file
        // output. The median of three runs damps one-off scheduler noise.
        for trial in 0..3 {
            let start = Instant::now();
            let image = raster.render(&path, k)?;
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            elapsed.push(ms);
            writeln!(trials, "{k},{trial},{ms:.9}")?;
            result = Some(image);
        }
        elapsed.sort_by(f64::total_cmp);
        let median = elapsed[1];
        let image = result.unwrap();
        save_coverage(&image, &output.join(format!("star_k{k}.bmp")))?;
        write_coverage(&output.join(format!("star_k{k}_coverage.csv")), &image)?;
        writeln!(timings, "{k},{size},{size},{median:.9}")?;
        chart.push((k as f64, median));
        results.push((k, image, median));
    }
    timings.flush()?;
    trials.flush()?;
    chart_svg(
        &output.join("timing.svg"),
        "Render cost (median of 3 trials)",
        "k",
        "Milliseconds",
        &[("measured", "#1565c0", chart)],
    )?;
    std::fs::write(output.join("aa_summary.json"),format!("{{\"canvas\":{size},\"angle_degrees\":5,\"min_abs_dx\":{min_dx:.12},\"min_abs_dy\":{min_dy:.12},\"k\":[1,2,4,8],\"timing_repeats\":3,\"timing_statistic\":\"median\",\"timing_scope\":\"geometry scaling, allocation, scanline fill, block averaging; excludes output\"}}\n"))?;
    Ok(results)
}
