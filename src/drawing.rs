//! A picture as Fourier series: drawing it as a bitmap or a browser SVG, and
//! reading and writing its coefficients.
use crate::{
    color::{hex, lab_to_rgb, svg_lab_filter},
    fit::{harmonics, samples},
    packed::{unpack, MAGIC},
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
/// its masks. Plates are L*, a*, b*; masks come in level order within each
/// plate.
pub(crate) struct Drawing {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) base: [f64; 3],
    pub(crate) masks: Vec<Layer>,
}

pub(crate) const PLATES: [&str; 3] = ["lightness", "a", "b"];

/// How much detail an SVG's outlines keep.
#[derive(Clone, Copy, PartialEq)]
pub struct SvgDetail {
    /// How far simplifying may move an outline, in output pixels.
    pub tolerance: f64,
    /// Decimal places of the coordinates.
    pub decimals: u32,
}
impl Default for SvgDetail {
    fn default() -> Self {
        Self { tolerance: 0.1, decimals: 1 }
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
    let unit = 10i64.pow(decimals);
    let number = |text: &mut String, v: i64| {
        if !text.ends_with(['M', 'l']) && v >= 0 {
            text.push(' ');
        }
        let (whole, part) = (v.abs() / unit, v.abs() % unit);
        let sign = if v < 0 { "-" } else { "" };
        let part = format!("{part:0width$}", width = decimals as usize);
        let part = part.trim_end_matches('0');
        text.push_str(&match (whole, part.is_empty()) {
            (_, true) => format!("{sign}{whole}"),
            (0, false) => format!("{sign}.{part}"),
            _ => format!("{sign}{whole}.{part}"),
        });
    };
    let rounded = |q: &Complex| ((q.re * unit as f64).round() as i64, (q.im * unit as f64).round() as i64);
    let mut text = String::from("M");
    let mut last = rounded(&z[0]);
    number(&mut text, last.0);
    number(&mut text, last.1);
    text.push('l');
    let mut steps = 0;
    for q in &z[1..] {
        let next = rounded(q);
        if next != last {
            number(&mut text, next.0 - last.0);
            number(&mut text, next.1 - last.1);
            last = next;
            steps += 1;
        }
    }
    if steps < 2 {
        return String::new();
    }
    text.push('z');
    text
}

impl Drawing {
    /// Rendered `scale` times larger: each mask's curves are sampled as in
    /// the fits (s from the outline's points), times `scale`, and filled
    /// together with the non-zero rule, each level over the ones below.
    pub(crate) fn bitmap(&self, scale: usize) -> Result<Image<Color>, Error> {
        let (w, h) = (self.w * scale, self.h * scale);
        let raster = Rasterizer::new(w, h)?;
        let mut plates = Vec::new();
        for base in self.base {
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
            *image.get_mut(i % w, i / w).unwrap() =
                lab_to_rgb(plates[0].pixels()[i], plates[1].pixels()[i], plates[2].pixels()[i]);
        }
        Ok(image)
    }

    /// The browser SVG, `scale` times larger: each plate lights only its own
    /// channel and the Lab filter turns the three channels back into RGB. A curve with m harmonics is sampled
    /// at 4m points (at least twice its highest frequency), times `scale`,
    /// then simplified as `detail` says; one non-zero path per level.
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
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" width=\"{w}\" height=\"{h}\">\n{}  <g filter=\"url(#lab)\">\n    <rect width=\"{w}\" height=\"{h}\" fill=\"#000000\"/>\n",
            svg_lab_filter(w, h)
        )?;
        for (channel, base) in self.base.iter().enumerate() {
            writeln!(
                file,
                "    <g style=\"mix-blend-mode:screen;isolation:isolate\">\n    <rect width=\"{w}\" height=\"{h}\" fill=\"{}\"/>",
                encode(channel, *base)
            )?;
            for (_, _, value, curves) in self.masks.iter().filter(|m| m.0 == channel) {
                write!(file, "    <path fill=\"{}\" d=\"", encode(channel, *value))?;
                for (_, m, c) in curves {
                    let count = (4 * m).next_power_of_two().max(16) * scale.next_power_of_two();
                    let k = scale as f64;
                    let z: Vec<Complex> = samples(c, *m, count)?
                        .iter()
                        .map(|q| Complex { re: q.re * k, im: q.im * k })
                        .collect();
                    write!(file, "{}", path_data(&simplify(&z, detail.tolerance), detail.decimals))?;
                }
                writeln!(file, "\"/>")?;
            }
            writeln!(file, "    </g>")?;
        }
        writeln!(file, "  </g>\n</svg>")?;
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
        if bytes.starts_with(MAGIC) {
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
        for (channel, plate) in PLATES.iter().enumerate() {
            base[channel] = *numbers_after(&format!("\"{plate}\":{{\"levels\":["))?.first().ok_or_else(bad)?;
        }
        let mut drawing = Drawing { w: size[0] as usize, h: size[1] as usize, base, masks: Vec::new() };
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
