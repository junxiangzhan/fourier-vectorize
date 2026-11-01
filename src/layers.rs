//! Part 4 vectorizer, extended to holes and full colour.
//!
//! Holes: every outline found by DBSCAN is traced on its own; outlines are
//! then nested by testing whether one lies inside another (the Part 2
//! winding test). Even depth = outer boundary, odd depth = hole. A shape's
//! holes are given the opposite direction to its outer boundary, so the
//! non-zero rule leaves them empty (the Part 2-3 conclusion about "O").
//!
//! Colour, two ways:
//! 1. Palette: find the flat colours, split every anti-aliased pixel between
//!    the two palette colours it is a blend of ("unmixing"), and vectorize one
//!    mask per colour.
//! 2. Channels: split the image into cyan / magenta / yellow ink, cut each
//!    channel into nested level masks, vectorize every mask, and put the three
//!    plates back together with a multiply blend, as in colour printing.
//!
//! All rendering and blending for comparison is done by our own code.
use crate::{
    experiments::circle,
    recover::{kind, primitive_row, recover, sample, svg_path, to_subpath, Primitive},
    vectorize::{descend, edge_points, initial_circle, B_HARMONICS, DENSE, K, SIZE},
};
use multimedia::{
    cluster::dbscan, winding_at, Color, Complex, Error, FlattenedPath, Image, Path as ShapePath,
    Point2, Polyline, Rasterizer, Subpath,
};
use std::{
    collections::HashMap,
    f64::consts::{FRAC_PI_2, TAU},
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

/// Same DBSCAN parameters as the single-colour recovery.
const EPS: f64 = 3.0;
const MIN_POINTS: usize = 3;
/// A colour (or ink level) must cover this many pixels to count as flat
/// colour rather than an anti-aliased blend along an edge.
const MIN_PIXELS: usize = 64;
/// Colours closer than this (CIE76 ΔE in Lab) are merged into one.
const MERGE_DELTA_E: f64 = 4.0;
/// Ink levels closer than this are merged into one.
const MERGE_LEVEL: f64 = 0.02;
/// A pixel counts as visibly wrong when a channel is off by more than this.
const VISIBLE: f64 = 0.1;

fn p(z: Complex) -> Point2 {
    Point2::new(z.re, z.im)
}

// ------------------------------------------------------------ tracing ----

/// One traced outline.
pub(crate) struct Contour {
    pub(crate) cluster: usize,
    pub(crate) edges: usize,
    pub(crate) primitives: Vec<Primitive>,
    /// Dense points along the recovered primitives.
    pub(crate) outline: Vec<Complex>,
    /// Shoelace area in screen coordinates: > 0 means screen-clockwise.
    pub(crate) area: f64,
}

pub(crate) fn signed_area(outline: &[Complex]) -> f64 {
    (0..outline.len())
        .map(|i| {
            let (a, b) = (outline[i], outline[(i + 1) % outline.len()]);
            a.re * b.im - a.im * b.re
        })
        .sum::<f64>()
        / 2.0
}

/// Edge points → DBSCAN → one Fourier outline per cluster → primitives.
pub(crate) fn trace(mask: &Image<f64>) -> Result<Vec<Contour>, Error> {
    let edges = edge_points(mask);
    let labels = dbscan(&edges, EPS, MIN_POINTS)?;
    let clusters = labels.iter().flatten().max().map_or(0, |m| m + 1);
    let mut contours = Vec::new();
    for id in 0..clusters {
        let targets: Vec<Complex> = edges
            .iter()
            .zip(&labels)
            .filter(|(_, l)| **l == Some(id))
            .map(|(q, _)| *q)
            .collect();
        let mut curve = initial_circle(&targets);
        for harmonics in B_HARMONICS {
            curve = curve.extended(harmonics);
            descend(&mut curve, &targets, |_, _| Ok(()))?;
        }
        let primitives = recover(&curve.sample(DENSE), &targets).primitives;
        let outline: Vec<Complex> = primitives.iter().flat_map(sample).collect();
        contours.push(Contour {
            cluster: id,
            edges: targets.len(),
            area: signed_area(&outline),
            primitives,
            outline,
        });
    }
    Ok(contours)
}

/// Is `point` inside the closed outline? Uses the Part 2 winding number.
fn inside(point: Complex, outline: &[Complex]) -> Result<bool, Error> {
    let path = FlattenedPath {
        subpaths: vec![Polyline {
            vertices: outline.iter().map(|z| p(*z)).collect(),
            closed: true,
        }],
    };
    Ok(winding_at(&path, p(point))? != 0)
}

/// Nesting of outlines: depth (how many others contain it) and parent (the
/// smallest one that does).
pub(crate) fn nest(contours: &[Contour]) -> Result<Vec<(usize, Option<usize>)>, Error> {
    let mut result = Vec::new();
    for (i, contour) in contours.iter().enumerate() {
        let mut containers = Vec::new();
        for (j, other) in contours.iter().enumerate() {
            if i != j && inside(contour.outline[0], &other.outline)? {
                containers.push(j);
            }
        }
        let parent = containers
            .iter()
            .copied()
            .min_by(|&a, &b| contours[a].area.abs().total_cmp(&contours[b].area.abs()));
        result.push((containers.len(), parent));
    }
    Ok(result)
}

/// A filled shape: an outer outline with its holes, outer screen-clockwise
/// and holes the other way. Returns the sub-paths and, per contour, whether
/// it was reversed to get that orientation.
pub(crate) struct Shape {
    pub(crate) outer: usize,
    pub(crate) holes: Vec<usize>,
    pub(crate) subpaths: Vec<Subpath>,
}

pub(crate) fn shapes(
    contours: &[Contour],
    nesting: &[(usize, Option<usize>)],
) -> (Vec<Shape>, Vec<bool>) {
    let wants_clockwise = |i: usize| nesting[i].0.is_multiple_of(2);
    let reversed: Vec<bool> = (0..contours.len())
        .map(|i| (contours[i].area > 0.0) != wants_clockwise(i))
        .collect();
    let oriented = |i: usize| {
        let sub = to_subpath(&contours[i].primitives);
        if reversed[i] {
            sub.reversed()
        } else {
            sub
        }
    };
    let result = (0..contours.len())
        .filter(|&i| wants_clockwise(i))
        .map(|outer| {
            let holes: Vec<usize> = (0..contours.len())
                .filter(|&h| !wants_clockwise(h) && nesting[h].1 == Some(outer))
                .collect();
            let mut subpaths = vec![oriented(outer)];
            subpaths.extend(holes.iter().map(|&h| oriented(h)));
            Shape {
                outer,
                holes,
                subpaths,
            }
        })
        .collect();
    (result, reversed)
}

pub(crate) fn svg_shape(shape: &Shape, fill: Color) -> String {
    let d: Vec<String> = shape.subpaths.iter().map(svg_path).collect();
    format!("  <path d=\"{}\" fill=\"{}\"/>\n", d.join(" "), hex(fill))
}

pub(crate) fn hex(color: Color) -> String {
    let c = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", c(color.r), c(color.g), c(color.b))
}

/// Writes one CSV row per contour and per primitive for the audit.
pub(crate) fn record(
    contours_file: &mut impl Write,
    primitives_file: &mut impl Write,
    layer: &str,
    contours: &[Contour],
    nesting: &[(usize, Option<usize>)],
    reversed: &[bool],
) -> std::io::Result<()> {
    for (i, contour) in contours.iter().enumerate() {
        let count = |k: &str| contour.primitives.iter().filter(|x| kind(x) == k).count();
        writeln!(
            contours_file,
            "{layer},{i},{},{},{},{},{:.6},{},{},{},{}",
            contour.cluster,
            contour.edges,
            nesting[i].0,
            nesting[i].1.map_or(String::new(), |v| v.to_string()),
            contour.area,
            reversed[i],
            count("line"),
            count("arc"),
            count("cubic")
        )?;
        for (index, primitive) in contour.primitives.iter().enumerate() {
            writeln!(
                primitives_file,
                "{layer},{i},{index},{},{}",
                kind(primitive),
                primitive_row(primitive)
            )?;
        }
    }
    Ok(())
}

// -------------------------------------------------------- compositing ----

pub(crate) fn paint(image: &mut Image<Color>, coverage: &Image<f64>, color: Color) {
    for y in 0..image.height() {
        for x in 0..image.width() {
            let pixel = image.get_mut(x, y).unwrap();
            *pixel = pixel.mix(color, *coverage.get(x, y).unwrap());
        }
    }
}

pub(crate) fn render(raster: &Rasterizer, subpaths: &[Subpath]) -> Result<Image<f64>, Error> {
    raster.render(&ShapePath::new(subpaths.to_vec()).flatten(7)?, K)
}

/// RMS over all channels, and pixels with some channel off by > VISIBLE.
pub(crate) fn color_error(a: &Image<Color>, b: &Image<Color>) -> (f64, usize) {
    let mut square = 0.0;
    let mut visible = 0;
    for (x, y) in a.pixels().iter().zip(b.pixels()) {
        let d = [x.r - y.r, x.g - y.g, x.b - y.b];
        square += d.iter().map(|v| v * v).sum::<f64>();
        if d.iter().any(|v| v.abs() > VISIBLE) {
            visible += 1;
        }
    }
    ((square / (3 * a.pixels().len()) as f64).sqrt(), visible)
}

/// Gray map of the per-pixel worst channel error (darker = larger).
pub(crate) fn error_map(a: &Image<Color>, b: &Image<Color>) -> Result<Image<Color>, Error> {
    let (w, h) = (a.width(), a.height());
    let mut map = Image::new(w, h, Color::rgb(1.0, 1.0, 1.0))?;
    for (i, (x, y)) in a.pixels().iter().zip(b.pixels()).enumerate() {
        let e = (x.r - y.r)
            .abs()
            .max((x.g - y.g).abs())
            .max((x.b - y.b).abs());
        *map.get_mut(i % w, i / w).unwrap() = Color::rgb(1.0 - e, 1.0 - e, 1.0 - e);
    }
    Ok(map)
}

pub(crate) fn write_rgb(path: &Path, image: &Image<Color>) -> Result<(), Error> {
    let mut file = BufWriter::new(File::create(path)?);
    writeln!(file, "x,y,r,g,b")?;
    for (i, c) in image.pixels().iter().enumerate() {
        writeln!(
            file,
            "{},{},{:.9},{:.9},{:.9}",
            i % image.width(),
            i / image.width(),
            c.r,
            c.g,
            c.b
        )?;
    }
    file.flush()?;
    Ok(())
}

// --------------------------------------------------------------- holes ----

/// An annulus as two circles; the inner one is reversed later by `shapes`,
/// whatever direction it was drawn in.
fn ring(center: Point2, outer: f64, inner: f64) -> Vec<Subpath> {
    vec![circle(center, outer), circle(center, inner).reversed()]
}

fn square(x0: f64, y0: f64, side: f64) -> Subpath {
    let mut s = Subpath::new(Point2::new(x0, y0));
    s.line_to(Point2::new(x0 + side, y0))
        .line_to(Point2::new(x0 + side, y0 + side))
        .line_to(Point2::new(x0, y0 + side))
        .close();
    s
}

fn holes(output: &Path, raster: &Rasterizer) -> Result<String, Error> {
    // An "O" with an island in its hole (depth 0, 1, 2) and a square frame.
    let mut scene = ring(Point2::new(72.0, 80.0), 46.0, 26.0);
    scene.push(circle(Point2::new(72.0, 80.0), 11.0));
    scene.push(square(140.0, 34.0, 84.0));
    scene.push(square(166.0, 60.0, 32.0).reversed());
    scene.extend(ring(Point2::new(128.0, 190.0), 40.0, 30.0));
    let original = render(raster, &scene)?;
    crate::experiments::save_coverage(&original, &output.join("holes_scene.bmp"))?;
    crate::experiments::write_coverage(&output.join("holes_scene_coverage.csv"), &original)?;
    let contours = trace(&original)?;
    let nesting = nest(&contours)?;
    let (found, reversed) = shapes(&contours, &nesting);
    let mut contours_file = BufWriter::new(File::create(output.join("holes_contours.csv"))?);
    writeln!(
        contours_file,
        "layer,contour,cluster,edges,depth,parent,area,reversed,lines,arcs,cubics"
    )?;
    let mut primitives_file = BufWriter::new(File::create(output.join("holes_primitives.csv"))?);
    writeln!(
        primitives_file,
        "layer,contour,index,kind,x0,y0,x1,y1,cx,cy,radius,sweep,q1x,q1y,q2x,q2y"
    )?;
    record(
        &mut contours_file,
        &mut primitives_file,
        "holes",
        &contours,
        &nesting,
        &reversed,
    )?;
    contours_file.flush()?;
    primitives_file.flush()?;
    let mut svg =
        format!("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {SIZE} {SIZE}\">\n");
    let mut all = Vec::new();
    for shape in &found {
        svg += &svg_shape(shape, Color::rgb(0.0, 0.0, 0.0));
        all.extend(shape.subpaths.iter().cloned());
    }
    svg += "</svg>\n";
    std::fs::write(output.join("holes_vector.svg"), svg)?;
    let rebuilt = render(raster, &all)?;
    crate::experiments::save_coverage(&rebuilt, &output.join("holes_rebuilt.bmp"))?;
    crate::experiments::write_coverage(&output.join("holes_rebuilt_coverage.csv"), &rebuilt)?;
    // The same outlines without the orientation fix, to show why it matters.
    let naive: Vec<Subpath> = contours.iter().map(|c| to_subpath(&c.primitives)).collect();
    let unfixed = render(raster, &naive)?;
    crate::experiments::save_coverage(&unfixed, &output.join("holes_unoriented.bmp"))?;
    crate::experiments::write_coverage(&output.join("holes_unoriented_coverage.csv"), &unfixed)?;
    let wrong = |img: &Image<f64>| {
        img.pixels()
            .iter()
            .zip(original.pixels())
            .filter(|(a, b)| (**a >= 0.5) != (**b >= 0.5))
            .count()
    };
    let shapes_json: Vec<String> = found
        .iter()
        .map(|s| format!("{{\"outer\":{},\"holes\":{:?}}}", s.outer, s.holes))
        .collect();
    Ok(format!(
        "\"holes\":{{\"contours\":{},\"shapes\":[{}],\"rebuilt_misclassified\":{},\"unoriented_misclassified\":{}}}",
        contours.len(),
        shapes_json.join(","),
        wrong(&rebuilt),
        wrong(&unfixed)
    ))
}

// -------------------------------------------------------------- colour ----

/// sRGB (0..1) to CIE Lab (D65), used only to judge how different two
/// colours look when merging the palette.
pub(crate) fn lab(c: Color) -> [f64; 3] {
    let linear = |v: f64| {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    let (r, g, b) = (linear(c.r), linear(c.g), linear(c.b));
    let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    let f = |t: f64| {
        if t > 0.008856 {
            t.cbrt()
        } else {
            7.787 * t + 16.0 / 116.0
        }
    };
    [
        116.0 * f(y) - 16.0,
        500.0 * (f(x) - f(y)),
        200.0 * (f(y) - f(z)),
    ]
}

pub(crate) fn delta_e(a: Color, b: Color) -> f64 {
    let (a, b) = (lab(a), lab(b));
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn quantize(v: f64) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Flat colours: 8-bit colours covering at least MIN_PIXELS pixels, merged
/// greedily (most common first) when within MERGE_DELTA_E of a kept one.
fn palette(image: &Image<Color>) -> Vec<(Color, usize)> {
    let mut counts: HashMap<(u8, u8, u8), usize> = HashMap::new();
    for c in image.pixels() {
        *counts
            .entry((quantize(c.r), quantize(c.g), quantize(c.b)))
            .or_default() += 1;
    }
    let mut candidates: Vec<((u8, u8, u8), usize)> = counts
        .into_iter()
        .filter(|(_, n)| *n >= MIN_PIXELS)
        .collect();
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut kept: Vec<(Color, usize)> = Vec::new();
    for ((r, g, b), n) in candidates {
        let color = Color::rgb(r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0);
        match kept
            .iter_mut()
            .find(|(k, _)| delta_e(*k, color) < MERGE_DELTA_E)
        {
            Some((k, total)) => {
                let w = n as f64 / (*total + n) as f64;
                *k = k.mix(color, w);
                *total += n;
            }
            None => kept.push((color, n)),
        }
    }
    kept
}

/// Splits a pixel between the two palette colours it is most likely a
/// blend of: the pair (a, b) and t minimizing |p - ((1-t)a + t b)|.
/// Returns (a, b, t).
fn unmix(p: Color, colors: &[Color]) -> (usize, usize, f64) {
    let mut best = (0, 0, 0.0, f64::INFINITY);
    for a in 0..colors.len() {
        for b in a + 1..colors.len() {
            let (ca, cb) = (colors[a], colors[b]);
            let d = [cb.r - ca.r, cb.g - ca.g, cb.b - ca.b];
            let e = [p.r - ca.r, p.g - ca.g, p.b - ca.b];
            let len = d.iter().map(|v| v * v).sum::<f64>();
            let t = if len > 0.0 {
                (d.iter().zip(&e).map(|(x, y)| x * y).sum::<f64>() / len).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let err: f64 = d.iter().zip(&e).map(|(x, y)| (y - x * t).powi(2)).sum();
            if err < best.3 {
                best = (a, b, t, err);
            }
        }
    }
    (best.0, best.1, best.2)
}

fn color_scene(raster: &Rasterizer) -> Result<Image<Color>, Error> {
    let mut image = Image::new(SIZE, SIZE, Color::rgb(1.0, 1.0, 1.0))?;
    let star = |center: Complex, r: f64| {
        let tip = |i: usize| {
            center + Complex::from_angle(-FRAC_PI_2 + i as f64 * TAU / 5.0 + 0.1).scale(r)
        };
        let mut s = Subpath::new(p(tip(0)));
        for i in [2, 4, 1, 3] {
            s.line_to(p(tip(i)));
        }
        s.close();
        s
    };
    let layers: Vec<(Vec<Subpath>, Color)> = vec![
        (
            vec![circle(Point2::new(86.0, 92.0), 54.0)],
            Color::rgb(0.9, 0.1, 0.1),
        ),
        (
            vec![star(Complex::new(150.0, 104.0), 54.0)],
            Color::rgb(0.1, 0.2, 0.85),
        ),
        (
            ring(Point2::new(190.0, 200.0), 40.0, 20.0),
            Color::rgb(0.1, 0.7, 0.3),
        ),
        (
            vec![
                square(28.0, 166.0, 64.0),
                square(48.0, 186.0, 24.0).reversed(),
            ],
            Color::rgb(0.95, 0.8, 0.1),
        ),
    ];
    for (subpaths, color) in layers {
        let coverage = render(raster, &subpaths)?;
        paint(&mut image, &coverage, color);
    }
    Ok(image)
}

fn colour(output: &Path, raster: &Rasterizer) -> Result<String, Error> {
    let original = color_scene(raster)?;
    original.save_bmp(output.join("color_scene.bmp"))?;
    write_rgb(&output.join("color_scene.csv"), &original)?;
    let mut contours_file = BufWriter::new(File::create(output.join("color_contours.csv"))?);
    writeln!(
        contours_file,
        "layer,contour,cluster,edges,depth,parent,area,reversed,lines,arcs,cubics"
    )?;
    let mut primitives_file = BufWriter::new(File::create(output.join("color_primitives.csv"))?);
    writeln!(
        primitives_file,
        "layer,contour,index,kind,x0,y0,x1,y1,cx,cy,radius,sweep,q1x,q1y,q2x,q2y"
    )?;

    // ---- Method 1: palette + unmixing, one mask per colour.
    let found = palette(&original);
    let colors: Vec<Color> = found.iter().map(|(c, _)| *c).collect();
    // Background: the palette colour of the four canvas corners' majority.
    let corners = [(0, 0), (SIZE - 1, 0), (0, SIZE - 1), (SIZE - 1, SIZE - 1)];
    let nearest = |c: Color| {
        (0..colors.len())
            .min_by(|&a, &b| delta_e(colors[a], c).total_cmp(&delta_e(colors[b], c)))
            .unwrap()
    };
    let mut votes = vec![0; colors.len()];
    for (x, y) in corners {
        votes[nearest(*original.get(x, y).unwrap())] += 1;
    }
    let background = (0..colors.len())
        .max_by_key(|&i| (votes[i], found[i].1))
        .unwrap();
    let mut masks: Vec<Image<f64>> = (0..colors.len())
        .map(|_| Image::new(SIZE, SIZE, 0.0))
        .collect::<Result<_, _>>()?;
    for (i, pixel) in original.pixels().iter().enumerate() {
        let (a, b, t) = unmix(*pixel, &colors);
        *masks[a].get_mut(i % SIZE, i / SIZE).unwrap() += 1.0 - t;
        *masks[b].get_mut(i % SIZE, i / SIZE).unwrap() += t;
    }
    // What the masks alone reproduce, before any vectorizing.
    let mut unmixed = Image::new(SIZE, SIZE, colors[background])?;
    for (index, mask) in masks.iter().enumerate() {
        if index != background {
            paint(&mut unmixed, mask, colors[index]);
        }
    }
    let mut svg1 = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {SIZE} {SIZE}\">\n  <rect width=\"{SIZE}\" height=\"{SIZE}\" fill=\"{}\"/>\n",
        hex(colors[background])
    );
    let mut rebuilt1 = Image::new(SIZE, SIZE, colors[background])?;
    let (mut contours1, mut paths1) = (0, 0);
    // Larger colour regions first, so smaller ones are painted on top.
    let mut order: Vec<usize> = (0..colors.len()).filter(|&i| i != background).collect();
    order.sort_by(|&a, &b| found[b].1.cmp(&found[a].1));
    for &index in &order {
        let contours = trace(&masks[index])?;
        let nesting = nest(&contours)?;
        let (shapes_found, reversed) = shapes(&contours, &nesting);
        record(
            &mut contours_file,
            &mut primitives_file,
            &format!("palette{index}"),
            &contours,
            &nesting,
            &reversed,
        )?;
        contours1 += contours.len();
        for shape in &shapes_found {
            svg1 += &svg_shape(shape, colors[index]);
            paths1 += 1;
            paint(
                &mut rebuilt1,
                &render(raster, &shape.subpaths)?,
                colors[index],
            );
        }
    }
    svg1 += "</svg>\n";
    std::fs::write(output.join("color_palette.svg"), &svg1)?;
    rebuilt1.save_bmp(output.join("color_palette_rebuilt.bmp"))?;
    write_rgb(&output.join("color_palette_rebuilt.csv"), &rebuilt1)?;
    error_map(&rebuilt1, &original)?.save_bmp(output.join("color_palette_error.bmp"))?;

    // ---- Method 2: CMY ink channels, nested level masks, multiply blend.
    let names = ["cyan", "magenta", "yellow"];
    let ink = |c: &Color, k: usize| 1.0 - [c.r, c.g, c.b][k];
    let mut plates = Vec::new();
    let mut svg2 = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {SIZE} {SIZE}\">\n  <rect width=\"{SIZE}\" height=\"{SIZE}\" fill=\"#ffffff\"/>\n"
    );
    let (mut contours2, mut paths2) = (0, 0);
    let mut level_json = Vec::new();
    let mut decomposed = [
        Image::new(SIZE, SIZE, 0.0)?,
        Image::new(SIZE, SIZE, 0.0)?,
        Image::new(SIZE, SIZE, 0.0)?,
    ];
    let mut vector_ink = [
        Image::new(SIZE, SIZE, 0.0)?,
        Image::new(SIZE, SIZE, 0.0)?,
        Image::new(SIZE, SIZE, 0.0)?,
    ];
    for channel in 0..3 {
        // Flat ink levels of this channel, lowest first; the lowest is the
        // paper (no ink).
        let mut counts: HashMap<u8, usize> = HashMap::new();
        for c in original.pixels() {
            *counts.entry(quantize(ink(c, channel))).or_default() += 1;
        }
        let mut values: Vec<(u8, usize)> = counts
            .into_iter()
            .filter(|(_, n)| *n >= MIN_PIXELS)
            .collect();
        values.sort();
        let mut levels: Vec<(f64, usize)> = Vec::new();
        for (v, n) in values {
            let v = v as f64 / 255.0;
            match levels.last_mut() {
                Some((l, total)) if v - *l < MERGE_LEVEL => {
                    *l = (*l * *total as f64 + v * n as f64) / (*total + n) as f64;
                    *total += n;
                }
                _ => levels.push((v, n)),
            }
        }
        let levels: Vec<f64> = levels.into_iter().map(|(l, _)| l).collect();
        level_json.push(format!("\"{}\":{:?}", names[channel], levels));
        svg2 += "  <g style=\"mix-blend-mode:multiply;isolation:isolate\">\n";
        for j in 1..levels.len() {
            // Threshold mask between level j-1 and level j: 0 below, 1 at or
            // above level j, a linear ramp in between. Painting the masks of
            // increasing level on top of each other gives the channel back.
            let (low, high) = (levels[j - 1], levels[j]);
            let mut mask = Image::new(SIZE, SIZE, 0.0)?;
            for (i, c) in original.pixels().iter().enumerate() {
                *mask.get_mut(i % SIZE, i / SIZE).unwrap() =
                    ((ink(c, channel) - low) / (high - low)).clamp(0.0, 1.0);
            }
            for (i, m) in mask.pixels().iter().enumerate() {
                let v = decomposed[channel].get_mut(i % SIZE, i / SIZE).unwrap();
                *v = *v * (1.0 - m) + high * m;
            }
            let contours = trace(&mask)?;
            let nesting = nest(&contours)?;
            let (shapes_found, reversed) = shapes(&contours, &nesting);
            record(
                &mut contours_file,
                &mut primitives_file,
                &format!("{}{j}", names[channel]),
                &contours,
                &nesting,
                &reversed,
            )?;
            contours2 += contours.len();
            let mut plate_color = [1.0, 1.0, 1.0];
            plate_color[channel] = 1.0 - high;
            let fill = Color::rgb(plate_color[0], plate_color[1], plate_color[2]);
            for shape in &shapes_found {
                svg2 += &svg_shape(shape, fill);
                paths2 += 1;
                let coverage = render(raster, &shape.subpaths)?;
                for (i, m) in coverage.pixels().iter().enumerate() {
                    let v = vector_ink[channel].get_mut(i % SIZE, i / SIZE).unwrap();
                    *v = *v * (1.0 - m) + high * m;
                }
            }
        }
        svg2 += "  </g>\n";
        // The plate as printed alone: white paper, this ink only.
        plates.push(vector_ink[channel].map(|v| {
            let mut c = [1.0, 1.0, 1.0];
            c[channel] = 1.0 - v;
            Color::rgb(c[0], c[1], c[2])
        })?);
    }
    svg2 += "</svg>\n";
    std::fs::write(output.join("color_channels.svg"), &svg2)?;
    for (plate, name) in plates.iter().zip(names) {
        plate.save_bmp(output.join(format!("color_plate_{name}.bmp")))?;
    }
    // Multiply the three plates on white paper: channel k = 1 - ink_k.
    let combine = |inks: &[Image<f64>; 3]| -> Result<Image<Color>, Error> {
        let mut out = Image::new(SIZE, SIZE, Color::rgb(1.0, 1.0, 1.0))?;
        for i in 0..SIZE * SIZE {
            let v = |k: usize| 1.0 - inks[k].pixels()[i];
            *out.get_mut(i % SIZE, i / SIZE).unwrap() = Color::rgb(v(0), v(1), v(2));
        }
        Ok(out)
    };
    let exact = combine(&decomposed)?;
    let rebuilt2 = combine(&vector_ink)?;
    rebuilt2.save_bmp(output.join("color_channels_rebuilt.bmp"))?;
    write_rgb(&output.join("color_channels_rebuilt.csv"), &rebuilt2)?;
    error_map(&rebuilt2, &original)?.save_bmp(output.join("color_channels_error.bmp"))?;
    contours_file.flush()?;
    primitives_file.flush()?;

    let (unmix_rms, unmix_visible) = color_error(&unmixed, &original);
    let (exact_rms, exact_visible) = color_error(&exact, &original);
    let (rms1, visible1) = color_error(&rebuilt1, &original);
    let (rms2, visible2) = color_error(&rebuilt2, &original);
    let palette_json: Vec<String> = found
        .iter()
        .map(|(c, n)| {
            format!(
                "{{\"rgb\":[{:.6},{:.6},{:.6}],\"pixels\":{n}}}",
                c.r, c.g, c.b
            )
        })
        .collect();
    Ok(format!(
        "\"color\":{{\"min_pixels\":{MIN_PIXELS},\"merge_delta_e\":{MERGE_DELTA_E},\"merge_level\":{MERGE_LEVEL},\"visible\":{VISIBLE},\"palette\":[{}],\"background\":{background},\"palette_method\":{{\"masks_rms\":{unmix_rms:.12},\"masks_visible\":{unmix_visible},\"contours\":{contours1},\"paths\":{paths1},\"svg_bytes\":{},\"rms\":{rms1:.12},\"visible\":{visible1}}},\"channel_method\":{{\"levels\":{{{}}},\"masks_rms\":{exact_rms:.12},\"masks_visible\":{exact_visible},\"contours\":{contours2},\"paths\":{paths2},\"svg_bytes\":{},\"rms\":{rms2:.12},\"visible\":{visible2}}}}}",
        palette_json.join(","),
        svg1.len(),
        level_json.join(","),
        svg2.len()
    ))
}

pub fn run(output: &Path) -> Result<(), Error> {
    let raster = Rasterizer::new(SIZE, SIZE)?;
    let holes_json = holes(output, &raster)?;
    let colour_json = colour(output, &raster)?;
    std::fs::write(
        output.join("layers_summary.json"),
        format!("{{\"canvas\":{SIZE},\"k\":{K},\"eps\":{EPS},\"min_points\":{MIN_POINTS},{holes_json},{colour_json}}}\n"),
    )?;
    Ok(())
}
