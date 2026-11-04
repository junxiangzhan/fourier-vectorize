//! A picture as Fourier series: drawing it as a bitmap or a browser SVG, and
//! reading and writing its coefficients.
use crate::{
    bezier::{fit_loop, Segment},
    color::{hex, lab_to_rgb, over_white, svg_lab_filter},
    fit::{harmonics, samples},
    packed::{is_packed, unpack},
    Color, Complex, Error, Image, Rasterizer,
};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

/// Supersampling of the fill.
pub(crate) const FILL_K: usize = 2;

/// One curve: outline points, harmonics m, coefficients c_-m..c_m.
pub(crate) type Curve = (usize, usize, Vec<Complex>);
/// One mask: plate, level, value, and its curves.
pub(crate) type Layer = (usize, usize, f64, Vec<Curve>);

/// A picture as Fourier series: canvas size, each plate's base level, and
/// its masks. Plates are L*, a*, b* and, for a picture with transparency,
/// alpha; masks come in level order within each plate.
pub(crate) struct Drawing {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) base: [f64; 3],
    /// The alpha plate's base level, if the picture has one.
    pub(crate) alpha: Option<f64>,
    pub(crate) masks: Vec<Layer>,
}

pub(crate) const PLATES: [&str; 4] = ["lightness", "a", "b", "alpha"];
/// The plate of `PLATES` that holds alpha.
pub(crate) const ALPHA: usize = 3;

/// How much detail an SVG's outlines keep.
#[derive(Clone, Copy, PartialEq)]
pub struct SvgDetail {
    /// How far simplifying may move an outline, in output pixels.
    pub tolerance: f64,
    /// Decimal places of the coordinates.
    pub decimals: u32,
    /// Straight lines only, instead of curves where those are shorter. Corners
    /// show when the SVG is enlarged.
    pub lines: bool,
}
impl Default for SvgDetail {
    fn default() -> Self {
        Self { tolerance: 0.1, decimals: 1, lines: false }
    }
}

/// Douglas-Peucker on a closed curve, from its first point around to that
/// point again: every dropped point stays within `tolerance` of the result.
fn simplify(z: &[Complex], tolerance: f64) -> Vec<Complex> {
    let n = z.len();
    let mut keep = vec![false; n];
    keep[0] = true;
    let mut spans = vec![(0, n)];
    while let Some((a, b)) = spans.pop() {
        let (p, q) = (z[a], z[b % n]);
        let (dx, dy) = (q.re - p.re, q.im - p.im);
        let squared = dx * dx + dy * dy;
        let mut worst = (0.0, a);
        for (i, v) in z.iter().enumerate().take(b).skip(a + 1) {
            let (x, y) = (v.re - p.re, v.im - p.im);
            let t = if squared > 0.0 { ((x * dx + y * dy) / squared).clamp(0.0, 1.0) } else { 0.0 };
            let distance = (x - t * dx).hypot(y - t * dy);
            if distance > worst.0 {
                worst = (distance, i);
            }
        }
        if worst.0 > tolerance {
            keep[worst.1] = true;
            spans.push((a, worst.1));
            spans.push((worst.1, b));
        }
    }
    z.iter().zip(&keep).filter(|(_, k)| **k).map(|(v, _)| *v).collect()
}

/// A closed subpath with `decimals` decimal places: an absolute start, then
/// relative steps, so rounding does not add up along the curve. Empty when
/// the curve rounds to fewer than three distinct corners and so encloses
/// nothing (a bare `l` would end the whole path in a browser).
fn path_data(z: &[Complex], decimals: u32) -> String {
    let segments: Vec<Segment> = z[1..].iter().map(|&end| Segment::Line { end }).collect();
    segment_data(z[0], &segments, decimals)
}

/// A number of a path, separated from the one before only where it has to be.
fn push_number(text: &mut String, v: i64, decimals: u32) {
    if !text.ends_with(['M', 'l', 'c']) && v >= 0 {
        text.push(' ');
    }
    let unit = 10i64.pow(decimals);
    let (whole, part) = (v.abs() / unit, v.abs() % unit);
    let sign = if v < 0 { "-" } else { "" };
    let part = format!("{part:0width$}", width = decimals as usize);
    let part = part.trim_end_matches('0');
    text.push_str(&match (whole, part.is_empty()) {
        (_, true) => format!("{sign}{whole}"),
        (0, false) => format!("{sign}.{part}"),
        _ => format!("{sign}{whole}.{part}"),
    });
}

/// A closed subpath from `start` along `segments`, as `path_data` describes. A
/// command letter is written only when it changes, and a closing line is left
/// to the `z`.
fn segment_data(start: Complex, segments: &[Segment], decimals: u32) -> String {
    let unit = 10i64.pow(decimals) as f64;
    let rounded = |q: Complex| ((q.re * unit).round() as i64, (q.im * unit).round() as i64);
    let origin = rounded(start);
    let mut text = String::from("M");
    push_number(&mut text, origin.0, decimals);
    push_number(&mut text, origin.1, decimals);
    let (mut last, mut letter, mut steps) = (origin, 'M', 0);
    for (i, segment) in segments.iter().enumerate() {
        let (command, points) = match *segment {
            Segment::Line { end } if i + 1 == segments.len() && rounded(end) == origin => continue,
            Segment::Line { end } => ('l', vec![rounded(end)]),
            Segment::Cubic { first, second, end } => ('c', vec![rounded(first), rounded(second), rounded(end)]),
        };
        if points.iter().all(|point| *point == last) {
            continue;
        }
        if command != letter {
            text.push(command);
            letter = command;
        }
        for point in &points {
            push_number(&mut text, point.0 - last.0, decimals);
            push_number(&mut text, point.1 - last.1, decimals);
        }
        last = points[points.len() - 1];
        steps += 1;
    }
    if steps < 2 {
        return String::new();
    }
    text.push('z');
    text
}

/// The curve's direction at each of its `count` samples, which is the series
/// differentiated: each c_k times i*k, up to a constant factor.
fn directions(coefficients: &[Complex], m: usize, count: usize) -> Result<Vec<Complex>, Error> {
    let turned: Vec<Complex> = coefficients
        .iter()
        .enumerate()
        .map(|(i, c)| *c * Complex::new(0.0, i as f64 - m as f64))
        .collect();
    samples(&turned, m, count)
}

/// One level's curves as the data of one path.
fn write_curves(file: &mut impl Write, curves: &[Curve], scale: usize, detail: SvgDetail) -> Result<(), Error> {
    for (_, m, c) in curves {
        let count = (4 * m).next_power_of_two().max(16) * scale.next_power_of_two();
        let k = scale as f64;
        let z: Vec<Complex> = samples(c, *m, count)?
            .iter()
            .map(|q| Complex { re: q.re * k, im: q.im * k })
            .collect();
        let data = if detail.lines {
            path_data(&simplify(&z, detail.tolerance), detail.decimals)
        } else {
            let tangents = directions(c, *m, count)?;
            segment_data(z[0], &fit_loop(&z, &tangents, detail.tolerance), detail.decimals)
        };
        write!(file, "{data}")?;
    }
    Ok(())
}

impl Drawing {
    /// Rendered `scale` times larger: each mask's curves are sampled as in
    /// the fits (s from the outline's points), times `scale`, and filled
    /// together with the non-zero rule, each level over the ones below. A
    /// bitmap has no transparency, so any alpha is drawn over white.
    pub(crate) fn bitmap(&self, scale: usize) -> Result<Image<Color>, Error> {
        let (w, h) = (self.w * scale, self.h * scale);
        let raster = Rasterizer::new(w, h)?;
        let mut plates = Vec::new();
        for base in self.base.into_iter().chain(self.alpha) {
            plates.push(Image::new(w, h, base)?);
        }
        for (channel, _, value, curves) in &self.masks {
            let contours: Vec<Vec<Complex>> = curves
                .iter()
                .map(|(n, m, c)| {
                    let s = (2 * n).max(4 * m + 2).next_power_of_two().max(64) * scale.next_power_of_two();
                    samples(c, *m, s).map(|z| z.iter().map(|q| q.scale(scale as f64)).collect())
                })
                .collect::<Result<_, _>>()?;
            let coverage = raster.render(&contours, FILL_K)?;
            let image = &mut plates[*channel];
            for (i, m) in coverage.pixels().iter().enumerate() {
                let v = image.get_mut(i % w, i / w).unwrap();
                *v = *v * (1.0 - m) + value * m;
            }
        }
        let mut image = Image::new(w, h, Color::rgb(0.0, 0.0, 0.0))?;
        for i in 0..w * h {
            let colour = lab_to_rgb(plates[0].pixels()[i], plates[1].pixels()[i], plates[2].pixels()[i]);
            *image.get_mut(i % w, i / w).unwrap() = match plates.get(ALPHA) {
                Some(alpha) => over_white(colour, alpha.pixels()[i]),
                None => colour,
            };
        }
        Ok(image)
    }

    /// The browser SVG, `scale` times larger: each plate lights only its own
    /// channel and the Lab filter turns the three channels back into RGB. A curve with m harmonics is sampled
    /// at 4m points (at least twice its highest frequency), times `scale`,
    /// then simplified as `detail` says; one non-zero path per level. An alpha
    /// plate is drawn in grey as a mask over the filtered picture.
    pub(crate) fn write_svg(&self, path: &Path, scale: usize, detail: SvgDetail) -> Result<(), Error> {
        let (w, h) = (self.w * scale, self.h * scale);
        let encode = |channel: usize, v: f64| {
            let mut c = [0.0; 3];
            c[channel] = if channel == 0 { v / 100.0 } else { (v + 128.0) / 255.0 };
            hex(Color::rgb(c[0], c[1], c[2]))
        };
        let mut file = BufWriter::new(File::create(path)?);
        write!(
            file,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" width=\"{w}\" height=\"{h}\">\n{}",
            svg_lab_filter(w, h)
        )?;
        if let Some(base) = self.alpha {
            let grey = |v: f64| hex(Color::rgb(v, v, v));
            writeln!(
                file,
                "  <mask id=\"alpha\" maskUnits=\"userSpaceOnUse\" x=\"0\" y=\"0\" width=\"{w}\" height=\"{h}\">\n    <rect width=\"{w}\" height=\"{h}\" fill=\"{}\"/>",
                grey(base)
            )?;
            for (_, _, value, curves) in self.masks.iter().filter(|m| m.0 == ALPHA) {
                write!(file, "    <path fill=\"{}\" d=\"", grey(*value))?;
                write_curves(&mut file, curves, scale, detail)?;
                writeln!(file, "\"/>")?;
            }
            writeln!(file, "  </mask>\n  <g mask=\"url(#alpha)\">")?;
        }
        writeln!(file, "  <g filter=\"url(#lab)\">\n    <rect width=\"{w}\" height=\"{h}\" fill=\"#000000\"/>")?;
        for (channel, base) in self.base.iter().enumerate() {
            writeln!(
                file,
                "    <g style=\"mix-blend-mode:screen;isolation:isolate\">\n    <rect width=\"{w}\" height=\"{h}\" fill=\"{}\"/>",
                encode(channel, *base)
            )?;
            for (_, _, value, curves) in self.masks.iter().filter(|m| m.0 == channel) {
                write!(file, "    <path fill=\"{}\" d=\"", encode(channel, *value))?;
                write_curves(&mut file, curves, scale, detail)?;
                writeln!(file, "\"/>")?;
            }
            writeln!(file, "    </g>")?;
        }
        writeln!(file, "  </g>")?;
        if self.alpha.is_some() {
            writeln!(file, "  </g>")?;
        }
        writeln!(file, "</svg>")?;
        file.flush()?;
        Ok(())
    }

    /// Every c_k as one CSV row: plate, level, value, outline, points, k, re,
    /// im. With the summary's size and base levels this is the whole picture.
    /// `sparse` leaves out the rows of coefficients that are zero, as pruned
    /// ones are.
    pub(crate) fn write_coefficients(&self, path: &Path, sparse: bool) -> Result<(), Error> {
        let mut file = BufWriter::new(File::create(path)?);
        writeln!(file, "plate,level,value,outline,points,k,re,im")?;
        for (channel, level, value, curves) in &self.masks {
            for (outline, (n, m, c)) in curves.iter().enumerate() {
                for (i, ck) in c.iter().enumerate() {
                    if sparse && ck.re == 0.0 && ck.im == 0.0 {
                        continue;
                    }
                    let k = i as i64 - *m as i64;
                    writeln!(file, "{},{level},{value:.9},{outline},{n},{k},{:.9},{:.9}", PLATES[*channel], ck.re, ck.im)?;
                }
            }
        }
        file.flush()?;
        Ok(())
    }

    /// Reads a coefficients CSV and the size and base levels from its JSON
    /// summary (the same stem, ending .json instead of _coefficients.csv).
    pub(crate) fn load(path: &Path) -> Result<Self, Error> {
        let bytes = std::fs::read(path)?;
        if is_packed(&bytes) {
            unpack(&bytes)
        } else {
            Self::read(path)
        }
    }

    pub(crate) fn read(path: &Path) -> Result<Self, Error> {
        let bad = || Error::InvalidParameter("not a coefficients file written by `fit`");
        let name = path.file_name().and_then(|n| n.to_str()).ok_or_else(bad)?;
        let stem = name.strip_suffix("_coefficients.csv").ok_or_else(bad)?;
        let summary = std::fs::read_to_string(path.with_file_name(format!("{stem}.json")))?;
        // The hand-written summary has fixed keys, so a plain search suffices.
        let numbers_after = |key: &str| -> Result<Vec<f64>, Error> {
            let start = summary.find(key).ok_or_else(bad)? + key.len();
            let end = start + summary[start..].find(']').ok_or_else(bad)?;
            summary[start..end].split(',').map(|v| v.trim().parse::<f64>().map_err(|_| bad())).collect()
        };
        let size = numbers_after("\"size\":[")?;
        let mut base = [0.0; 3];
        for (channel, plate) in PLATES.iter().enumerate().take(ALPHA) {
            base[channel] = *numbers_after(&format!("\"{plate}\":{{\"levels\":["))?.first().ok_or_else(bad)?;
        }
        // Only a picture with transparency has this plate.
        let key = format!("\"{}\":{{\"levels\":[", PLATES[ALPHA]);
        let alpha = if summary.contains(&key) { Some(*numbers_after(&key)?.first().ok_or_else(bad)?) } else { None };
        let mut drawing = Drawing { w: size[0] as usize, h: size[1] as usize, base, alpha, masks: Vec::new() };
        let expected = summary
            .split("\"coefficients\":")
            .nth(1)
            .and_then(|t| t.split(',').next())
            .and_then(|v| v.parse::<usize>().ok())
            .ok_or_else(bad)?;
        let text = std::fs::read_to_string(path)?;
        let (mut key, mut current, mut rows) = (None, None, 0);
        for line in text.lines().skip(1) {
            let f: Vec<&str> = line.split(',').collect();
            if f.len() != 8 {
                return Err(bad());
            }
            let channel = PLATES.iter().position(|p| *p == f[0]).ok_or_else(bad)?;
            if channel == ALPHA && alpha.is_none() {
                return Err(bad());
            }
            let number = |s: &str| s.parse::<f64>().map_err(|_| bad());
            let (level, value) = (f[1].parse::<usize>().map_err(|_| bad())?, number(f[2])?);
            let (outline, points, k) = (f[3].parse::<usize>().map_err(|_| bad())?, f[4].parse::<usize>().map_err(|_| bad())?, f[5].parse::<i64>().map_err(|_| bad())?);
            if key != Some((channel, level)) {
                drawing.masks.push((channel, level, value, Vec::new()));
                key = Some((channel, level));
                current = None;
            }
            let curves = &mut drawing.masks.last_mut().unwrap().3;
            // A coefficient with no row is zero.
            if current != Some(outline) {
                let m = harmonics(points);
                curves.push((points, m, vec![Complex::ZERO; 2 * m + 1]));
                current = Some(outline);
            }
            let (n, m, c) = curves.last_mut().unwrap();
            let index = k + *m as i64;
            if *n != points || index < 0 || index >= c.len() as i64 {
                return Err(bad());
            }
            c[index as usize] = Complex::new(number(f[6])?, number(f[7])?);
            rows += 1;
        }
        // A cut-off file would otherwise read as a pruned one.
        if rows != expected {
            return Err(bad());
        }
        Ok(drawing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplify_drops_straight_runs_and_stays_within_tolerance() {
        // A square with many points per side keeps just its corners.
        let mut square = Vec::new();
        for (x, y, dx, dy) in [(0.0, 0.0, 1.0, 0.0), (100.0, 0.0, 0.0, 1.0), (100.0, 100.0, -1.0, 0.0), (0.0, 100.0, 0.0, -1.0)] {
            square.extend((0..100).map(|i| Complex { re: x + dx * i as f64, im: y + dy * i as f64 }));
        }
        assert_eq!(simplify(&square, 0.1).len(), 4);
        // A circle loses most points, and each lost one stays near the result.
        let circle: Vec<Complex> = (0..1024)
            .map(|i| {
                let a = i as f64 / 1024.0 * std::f64::consts::TAU;
                Complex { re: 50.0 * a.cos(), im: 50.0 * a.sin() }
            })
            .collect();
        let kept = simplify(&circle, 0.1);
        assert!(kept.len() <= 64, "{}", kept.len());
        for q in &circle {
            let nearest = (0..kept.len())
                .map(|i| {
                    let (p, r) = (kept[i], kept[(i + 1) % kept.len()]);
                    let (dx, dy, x, y) = (r.re - p.re, r.im - p.im, q.re - p.re, q.im - p.im);
                    let t = ((x * dx + y * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
                    (x - t * dx).hypot(y - t * dy)
                })
                .fold(f64::INFINITY, f64::min);
            assert!(nearest <= 0.1 + 1e-9, "{nearest}");
        }
    }

    #[test]
    fn segment_data_uses_a_letter_only_when_it_changes_and_leaves_the_closing_line_to_z() {
        let at = |re: f64, im: f64| Complex { re, im };
        let line = |re, im| Segment::Line { end: at(re, im) };
        let curve = |a, b, c, d, e, f| Segment::Cubic { first: at(a, b), second: at(c, d), end: at(e, f) };
        let segments = [
            line(1.0, 0.0),
            line(2.0, 0.0),
            curve(2.0, 1.0, 3.0, 2.0, 3.0, 3.0),
            curve(3.0, 3.5, 2.5, 4.0, 2.0, 4.0),
            line(0.0, 4.0),
            line(0.0, 0.0),
        ];
        assert_eq!(segment_data(at(0.0, 0.0), &segments, 1), "M0 0l1 0 1 0c0 1 1 2 1 3 0 .5-.5 1-1 1l-2 0z");
        // Nothing to enclose once rounded: no subpath at all.
        let tiny = [line(0.01, 0.0), curve(0.01, 0.01, 0.02, 0.01, 0.0, 0.01), line(0.0, 0.0)];
        assert_eq!(segment_data(at(0.0, 0.0), &tiny, 1), "");
    }

    #[test]
    fn svgs_use_curves_unless_told_to_use_lines() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join(format!("svg-curves-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // One circle, of radius 50 at (60, 60): c_0 and c_1 of m = 2.
        let c = vec![Complex::ZERO, Complex::ZERO, Complex::new(60.0, 60.0), Complex::new(50.0, 0.0), Complex::ZERO];
        let drawing = Drawing { w: 120, h: 120, base: [0.0; 3], alpha: None, masks: vec![(0, 1, 50.0, vec![(6, 2, c)])] };
        let path_of = |lines: bool, scale: usize| {
            let file = dir.join("circle.svg");
            drawing.write_svg(&file, scale, SvgDetail { lines, ..SvgDetail::default() }).unwrap();
            let text = std::fs::read_to_string(&file).unwrap();
            let start = text.find(" d=\"").unwrap() + 4;
            text[start..start + text[start..].find('"').unwrap()].to_string()
        };
        let (curves, lines) = (path_of(false, 1), path_of(true, 1));
        assert!(curves.contains('c') && !curves.contains('l'), "{curves}");
        assert!(lines.contains('l') && !lines.contains('c'), "{lines}");
        assert!(curves.len() < lines.len(), "{curves} vs {lines}");
        // Enlarged, a few more curves do where lines need many more.
        let (curves, lines) = (path_of(false, 8), path_of(true, 8));
        assert!(curves.len() * 3 < lines.len(), "{} vs {}", curves.len(), lines.len());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_alpha_plate_masks_the_svg_and_lays_the_bitmap_over_white() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join(format!("svg-alpha-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Black all over, opaque only inside a circle of radius 50 at (60, 60).
        let c = vec![Complex::ZERO, Complex::ZERO, Complex::new(60.0, 60.0), Complex::new(50.0, 0.0), Complex::ZERO];
        let circle = vec![(ALPHA, 1, 1.0, vec![(6, 2, c)])];
        let write = |drawing: &Drawing| {
            let file = dir.join("alpha.svg");
            drawing.write_svg(&file, 1, SvgDetail::default()).unwrap();
            std::fs::read_to_string(&file).unwrap()
        };
        let masked = Drawing { w: 120, h: 120, base: [0.0; 3], alpha: Some(0.0), masks: circle };
        let text = write(&masked);
        let (mask, group, filter) = (text.find("<mask id=\"alpha\"").unwrap(), text.find("<g mask=\"url(#alpha)\">").unwrap(), text.find("<g filter=\"url(#lab)\">").unwrap());
        assert!(mask < group && group < filter, "{text}");
        assert_eq!(text.matches("<g").count(), text.matches("</g>").count());
        assert!(text[mask..group].contains("fill=\"#000000\"") && text[mask..group].contains("<path fill=\"#ffffff\""), "{text}");
        let image = masked.bitmap(1).unwrap();
        assert_eq!(*image.get(2, 2).unwrap(), Color::rgb(1.0, 1.0, 1.0));
        assert!(image.get(60, 60).unwrap().r < 0.01);
        // Without a plate, nothing of it appears.
        let opaque = Drawing { w: 120, h: 120, base: [0.0; 3], alpha: None, masks: Vec::new() };
        assert!(!write(&opaque).contains("mask"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn path_data_steps_back_to_the_rounded_points() {
        let points = [(12.34, 5.0), (12.34, 5.0), (0.26, -3.2), (-7.0, 8.06), (100.0, 0.5)];
        let z: Vec<Complex> = points.iter().map(|&(re, im)| Complex { re, im }).collect();
        let text = path_data(&z, 1);
        assert!(text.starts_with('M') && text.ends_with('z') && text.contains('l'), "{text}");
        let numbers: Vec<f64> = text
            .replace(['M', 'l', 'z'], " ")
            .replace('-', " -")
            .split_whitespace()
            .map(|n| n.parse().unwrap())
            .collect();
        let mut at = (numbers[0], numbers[1]);
        let mut drawn = vec![at];
        for step in numbers[2..].chunks(2) {
            at = (at.0 + step[0], at.1 + step[1]);
            drawn.push(at);
        }
        // The repeated point is gone; the rest land on their rounded places.
        let wanted = [(12.3, 5.0), (0.3, -3.2), (-7.0, 8.1), (100.0, 0.5)];
        assert_eq!(drawn.len(), wanted.len(), "{text}");
        for (d, w) in drawn.iter().zip(wanted) {
            assert!((d.0 - w.0).abs() < 1e-9 && (d.1 - w.1).abs() < 1e-9, "{text}");
        }
        // Nothing to enclose: no subpath at all rather than a bare `l`.
        let dot = [Complex { re: 1.0, im: 1.0 }, Complex { re: 1.01, im: 1.0 }];
        assert_eq!(path_data(&dot, 1), "");
        // Other precisions: whole pixels, and hundredths with zeros trimmed.
        let z = [(1.26, 2.0), (3.5, 4.04), (0.0, 9.1)].map(|(re, im)| Complex { re, im });
        assert_eq!(path_data(&z, 0), "M1 2l3 2-4 5z");
        assert_eq!(path_data(&z, 2), "M1.26 2l2.24 2.04-3.5 5.06z");
    }
}
