//! Packed coefficients: a compact binary form of a coefficients file that also
//! holds the canvas size, base levels and any colour table, so it stands alone.
//!
//! Quantizing is what makes it small, and the only lossy step. Each curve has
//! one step, from its count of kept coefficients K: every coefficient rounded
//! to a step s moves the curve's points by an RMS of s * sqrt(K / 6), so
//! s = movement * sqrt(6 / K) moves every curve by the same amount, however
//! long. Measured on two paintings, that beat one step for all curves, and
//! beat rounding small values to zero more readily. c_0, where the curve sits,
//! is rounded to half of `movement` on every curve instead, so that the c_0 of
//! curves close together differ by little.
//!
//! The file, with varints for the numbers not otherwise stated:
//!
//! ```text
//! "FVC3"      the format and its version
//! u8          compression: only LZMA2 so far
//! varint × 2  canvas width and height
//! u32         LZMA2 dictionary size, and from here on one LZMA2 stream of layers:
//!   u8        mode: MMMA SSSS
//!   u8        flags: FLAG_SPARSE; any other bit is refused
//!   f32       movement
//!   a table layer: u8 the table's colours, MMMA SSSS as for a mode; only sRGB
//!     varint  number of colours, then f32 × 3 for each
//!   for each plate the mode has: f32 base level, varint number of levels
//!     for each level: f32 value, varint number of curves
//!     (a table's own plate: varint indices into the table, not f32 values)
//!   for each curve: varint top, so k = -top..=top (FLAG_SPARSE: varint K)
//!   FLAG_SPARSE, for each curve: K - 1 gaps between the kept positions after 0
//!   for each curve: c_0, re and im, as differences from the level's last
//!   for each band of positions after 0, for each curve: its coefficients there
//! ```
//!
//! Coefficients go with |k| rising and as zigzag varints.
//!
//! A layer's mode says how many plates follow and what they are. MMM is the
//! colour model: 000 a joint colour table, 001 RGB, 010 CMY, 011 Lab. A adds an
//! alpha plate after the model's own. SSSS is the colour space within the model:
//! up to 15 named ones, with 1111 for one whose parameters follow the flags. So
//! `0b0111_0000` is four plates: L*, a*, b* and alpha, in CIELAB with a D65
//! white. A table layer (000) has one plate of its own, whose base and levels
//! are colours of the table, each painted over the ones before it; the table
//! names its own space, so the mode's SSSS is 0000. Only Lab with a D65 white
//! and tables of sRGB colours (`0b0010_0000`, without alpha) are laid out so
//! far; all other modes are refused.
//!
//! Layers run to the end of the stream, each painted over those before it and
//! otherwise independent of them, so a file does not hold their count: its stream
//! has an end of its own. A picture needs only one, and more are refused for now.
//!
//! A level's curves are filled together, so sorting them by where they sit
//! changes nothing drawn. How finely to sample a curve follows from its length,
//! so the file does not hold its count of outline points.
//!
//! Against one curve after another with that count stored, on two paintings:
//! the bands saved about 3%, sorting with the c_0 grid 2%, leaving out the count
//! 2%, and the LZMA settings under 1%. A model of an adaptive arithmetic coder
//! would save about 3% more, for which `compression` leaves room.
use crate::{
    drawing::{Curve, Drawing, Level, ALPHA},
    fit::{samples, MAX_HARMONICS},
    Color, Complex, Error,
};
use lzma_rust2::{Lzma2Options, Lzma2Reader, Lzma2Writer};
use std::io::{Read, Write};

/// The start of a packed file: the format and its version.
const MAGIC: &[u8; 4] = b"FVC3";
/// A layer's mode: Lab, CIELAB with a D65 white, and no alpha plate.
const LAB_LAYER: u8 = 0b0110_0000;
/// A layer's mode: a joint colour table, and no alpha plate.
const TABLE_LAYER: u8 = 0b0000_0000;
/// The bit of a layer's mode that adds an alpha plate.
const WITH_ALPHA: u8 = 0b0001_0000;
/// The colours of a table: sRGB, without alpha.
const SRGB: u8 = 0b0010_0000;
/// The most colours a table may claim.
const MAX_COLOURS: usize = 1 << 16;
/// Some curve of the layer keeps other than all of its lowest harmonics.
const FLAG_SPARSE: u8 = 1;
const LZMA2: u8 = 0;
const PRESET: u32 = 9;
/// Varints of mostly noise have no alignment to model.
const LITERAL_CONTEXT: u32 = 4;
const POSITION_BITS: u32 = 0;
/// The largest dictionary a file may ask a reader to allocate.
const MAX_DICT: u32 = 64 << 20;
const MIN_DICT: u32 = 1 << 16;
/// The largest canvas a file may claim, in pixels.
const MAX_PIXELS: usize = 1 << 30;
/// Pixels between the rows that curves are sorted in.
const SORT_ROW: f64 = 32.0;
/// Where each band of positions after 0 ends.
const BANDS: [usize; 3] = [5, 17, usize::MAX];
/// Outline points per pixel of a curve's length, about what tracing one gives.
const POINTS_PER_PIXEL: f64 = 1.25;
/// The most outline points a curve's length can stand for.
const MAX_POINTS: f64 = (1 << 20) as f64;

/// Whether these bytes claim to be a packed file of any version.
pub(crate) fn is_packed(bytes: &[u8]) -> bool {
    bytes.starts_with(b"FVC") && bytes.get(3).is_some_and(u8::is_ascii_digit)
}

fn bad() -> Error {
    Error::InvalidParameter("not a valid packed coefficients file")
}

fn other_version() -> Error {
    Error::InvalidParameter("this packed file was written by a different version of the program; fit the picture again")
}

fn step(movement: f64, kept: usize) -> f64 {
    movement * (6.0 / kept.max(1) as f64).sqrt()
}

/// The step of every curve's c_0.
fn grid(movement: f64) -> f64 {
    movement / 2.0
}

/// The harmonic at a position of the order 0, -1, 1, -2, 2, ...
fn harmonic(position: usize) -> i64 {
    let half = position.div_ceil(2) as i64;
    if position % 2 == 1 {
        -half
    } else {
        half
    }
}

fn put(out: &mut Vec<u8>, mut value: u64) {
    while value >= 128 {
        out.push((value & 127) as u8 | 128);
        value >>= 7;
    }
    out.push(value as u8);
}

fn put_signed(out: &mut Vec<u8>, value: i64) {
    put(out, ((value << 1) ^ (value >> 63)) as u64);
}

fn put_float(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(&(value as f32).to_le_bytes());
}

/// One curve as it goes in a file: the positions it keeps, c_0's always first.
struct Kept<'a> {
    m: i64,
    c: &'a [Complex],
    positions: Vec<usize>,
}
impl Kept<'_> {
    fn at(&self, position: usize) -> Complex {
        self.c[(harmonic(position) + self.m) as usize]
    }
    /// The highest |k| kept.
    fn top(&self) -> usize {
        self.positions.last().map_or(0, |p| p.div_ceil(2))
    }
    fn is_whole_band(&self) -> bool {
        self.positions.len() == 2 * self.top() + 1
    }
}

/// A level's curves as they go in a file, sorted by where they sit.
fn kept(curves: &[Curve]) -> Vec<Kept<'_>> {
    let mut all: Vec<Kept> = curves
        .iter()
        .map(|(_, m, c)| {
            let mut curve = Kept { m: *m as i64, c, positions: vec![0] };
            let live: Vec<usize> = (1..c.len())
                .filter(|&p| {
                    let v = curve.at(p);
                    v.re != 0.0 || v.im != 0.0
                })
                .collect();
            curve.positions.extend(live);
            curve
        })
        .collect();
    let row = |curve: &Kept| (curve.at(0).im / SORT_ROW).floor();
    all.sort_by(|a, b| row(a).total_cmp(&row(b)).then(a.at(0).re.total_cmp(&b.at(0).re)));
    all
}

/// `drawing` as a packed file's bytes; a coefficient that is exactly zero is
/// left out, as pruned ones are.
pub(crate) fn pack(drawing: &Drawing, movement: f64) -> Result<Vec<u8>, Error> {
    // The reader gets it as an f32, and its steps must be these.
    let movement = f64::from(movement as f32);
    let colours = if drawing.table.is_some() { 1 } else { ALPHA };
    let order: Vec<usize> = (0..colours).chain(drawing.alpha.map(|_| ALPHA)).collect();
    let plates: Vec<Vec<&Level>> = order.iter().map(|&plate| drawing.masks.iter().filter(|level| level.0 == plate).collect()).collect();
    let layers: Vec<Vec<Kept>> = plates.iter().flatten().map(|(_, _, _, curves)| kept(curves)).collect();
    let sparse = layers.iter().flatten().any(|curve| !curve.is_whole_band());
    let model = if drawing.table.is_some() { TABLE_LAYER } else { LAB_LAYER };
    let mut body = vec![model | if drawing.alpha.is_some() { WITH_ALPHA } else { 0 }, if sparse { FLAG_SPARSE } else { 0 }];
    put_float(&mut body, movement);
    if let Some(table) = &drawing.table {
        body.push(SRGB);
        put(&mut body, table.len() as u64);
        for c in table {
            for v in [c.r, c.g, c.b] {
                put_float(&mut body, v);
            }
        }
    }
    // A table's own plate holds indices into it.
    let value = |body: &mut Vec<u8>, plate: usize, v: f64| -> Result<(), Error> {
        match (&drawing.table, plate) {
            (Some(table), 0) if v >= 0.0 && (v as usize) < table.len() => put(body, v as u64),
            (Some(_), 0) => return Err(Error::InvalidParameter("a colour index past the table")),
            _ => put_float(body, v),
        }
        Ok(())
    };
    let mut sorted = layers.iter();
    for (&plate, levels) in order.iter().zip(&plates) {
        let base = drawing.base.get(plate).copied().or(drawing.alpha).unwrap_or_default();
        value(&mut body, plate, base)?;
        put(&mut body, levels.len() as u64);
        for (level, curves) in levels.iter().zip(sorted.by_ref()) {
            value(&mut body, plate, level.2)?;
            put(&mut body, curves.len() as u64);
        }
    }
    for curve in layers.iter().flatten() {
        put(&mut body, if sparse { curve.positions.len() } else { curve.top() } as u64);
    }
    if sparse {
        for curve in layers.iter().flatten() {
            for pair in curve.positions.windows(2) {
                put(&mut body, (pair[1] - pair[0] - 1) as u64);
            }
        }
    }
    for curves in &layers {
        let mut last = (0, 0);
        for curve in curves {
            let v = curve.at(0);
            let here = ((v.re / grid(movement)).round() as i64, (v.im / grid(movement)).round() as i64);
            put_signed(&mut body, here.0 - last.0);
            put_signed(&mut body, here.1 - last.1);
            last = here;
        }
    }
    let mut start = 1;
    for end in BANDS {
        for curve in layers.iter().flatten() {
            let s = step(movement, curve.positions.len());
            for &p in curve.positions.iter().filter(|&&p| (start..end).contains(&p)) {
                let v = curve.at(p);
                put_signed(&mut body, (v.re / s).round() as i64);
                put_signed(&mut body, (v.im / s).round() as i64);
            }
        }
        start = end;
    }
    let mut options = Lzma2Options::with_preset(PRESET);
    let dict = u32::try_from(body.len()).unwrap_or(MAX_DICT).clamp(MIN_DICT, MAX_DICT);
    options.lzma_options.dict_size = dict;
    options.lzma_options.lc = LITERAL_CONTEXT;
    options.lzma_options.pb = POSITION_BITS;
    let mut out = MAGIC.to_vec();
    out.push(LZMA2);
    put(&mut out, drawing.w as u64);
    put(&mut out, drawing.h as u64);
    out.extend_from_slice(&dict.to_le_bytes());
    let mut writer = Lzma2Writer::new(out, options);
    writer.write_all(&body)?;
    Ok(writer.finish()?)
}

/// Reads the numbers of an unpacked body, refusing anything out of place.
struct Body<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl Body<'_> {
    fn byte(&mut self) -> Result<u8, Error> {
        let b = *self.bytes.get(self.at).ok_or_else(bad)?;
        self.at += 1;
        Ok(b)
    }
    fn number(&mut self) -> Result<u64, Error> {
        let (mut value, mut shift) = (0u64, 0);
        loop {
            let b = self.byte()?;
            if shift >= 64 || (shift == 63 && b > 1) {
                return Err(bad());
            }
            value |= u64::from(b & 127) << shift;
            if b < 128 {
                return Ok(value);
            }
            shift += 7;
        }
    }
    fn size(&mut self) -> Result<usize, Error> {
        usize::try_from(self.number()?).map_err(|_| bad())
    }
    fn signed(&mut self) -> Result<i64, Error> {
        let z = self.number()?;
        Ok((z >> 1) as i64 ^ -((z & 1) as i64))
    }
    fn take<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        let end = self.at.checked_add(N).ok_or_else(bad)?;
        let bytes = self.bytes.get(self.at..end).ok_or_else(bad)?;
        self.at = end;
        bytes.try_into().map_err(|_| bad())
    }
    fn short_float(&mut self) -> Result<f64, Error> {
        Some(f64::from(f32::from_le_bytes(self.take()?))).filter(|v| v.is_finite()).ok_or_else(bad)
    }
    /// The bytes left, which is at most how many numbers are.
    fn left(&self) -> usize {
        self.bytes.len() - self.at
    }
}

/// What an LZMA2 stream holds, from the dictionary size before it.
fn inflate(rest: &[u8]) -> Result<Vec<u8>, Error> {
    let (dict, stream) = rest.split_first_chunk::<4>().ok_or_else(bad)?;
    let dict = u32::from_le_bytes(*dict);
    if !(MIN_DICT..=MAX_DICT).contains(&dict) {
        return Err(bad());
    }
    let mut unpacked = Vec::new();
    Lzma2Reader::new(stream, dict, None).read_to_end(&mut unpacked).map_err(|_| bad())?;
    Ok(unpacked)
}

/// As many outline points as a curve this long would have, which is all that
/// drawing it asks of them: how finely to sample.
fn points_along(c: &[Complex], m: usize) -> Result<usize, Error> {
    let z = samples(c, m, (8 * m).next_power_of_two().max(64))?;
    let length: f64 = z.iter().zip(z.iter().cycle().skip(1)).map(|(a, b)| (*a - *b).abs()).sum();
    Ok((length * POINTS_PER_PIXEL).clamp(1.0, MAX_POINTS) as usize)
}

/// The drawing a packed file holds.
pub(crate) fn unpack(bytes: &[u8]) -> Result<Drawing, Error> {
    if is_packed(bytes) && !bytes.starts_with(MAGIC) {
        return Err(other_version());
    }
    let rest = bytes.strip_prefix(MAGIC).ok_or_else(bad)?;
    let mut header = Body { bytes: rest, at: 0 };
    if header.byte()? != LZMA2 {
        return Err(other_version());
    }
    let (w, h) = (header.size()?, header.size()?);
    if w == 0 || h == 0 || w.checked_mul(h).is_none_or(|pixels| pixels > MAX_PIXELS) {
        return Err(bad());
    }
    let unpacked = inflate(&rest[header.at..])?;
    let mut body = Body { bytes: &unpacked, at: 0 };
    let mode = body.byte()?;
    let flags = body.byte()?;
    let model = mode & !WITH_ALPHA;
    if (model != LAB_LAYER && model != TABLE_LAYER) || flags & !FLAG_SPARSE != 0 {
        return Err(other_version());
    }
    let sparse = flags & FLAG_SPARSE != 0;
    let movement = body.short_float()?;
    if movement <= 0.0 {
        return Err(bad());
    }
    let table = if model == TABLE_LAYER {
        if body.byte()? != SRGB {
            return Err(other_version());
        }
        let count = body.size()?;
        if count == 0 || count > MAX_COLOURS || count * 12 > body.left() {
            return Err(bad());
        }
        let mut colours = Vec::with_capacity(count);
        for _ in 0..count {
            colours.push(Color::rgb(body.short_float()?, body.short_float()?, body.short_float()?));
        }
        Some(colours)
    } else {
        None
    };
    let colours = table.as_ref().map_or(ALPHA, |_| 1);
    let order: Vec<usize> = (0..colours).chain((mode & WITH_ALPHA != 0).then_some(ALPHA)).collect();
    let (mut bases, mut layers) = ([0.0; ALPHA + 1], Vec::new());
    for &plate in &order {
        let value = |body: &mut Body| match (&table, plate) {
            (Some(table), 0) => Some(body.size()?).filter(|&i| i < table.len()).map(|i| i as f64).ok_or_else(bad),
            _ => body.short_float(),
        };
        bases[plate] = value(&mut body)?;
        for _ in 0..body.size()? {
            layers.push((plate, value(&mut body)?, body.size()?));
        }
    }
    // Each curve's count of kept coefficients, each at least a byte to come.
    let (mut counts, mut total) = (Vec::new(), 0usize);
    for &(_, _, curves) in &layers {
        for _ in 0..curves {
            let size = body.size()?;
            let kept = if sparse { size } else { size.checked_mul(2).and_then(|d| d.checked_add(1)).ok_or_else(bad)? };
            if kept == 0 || kept > 2 * MAX_HARMONICS + 1 {
                return Err(bad());
            }
            total += kept;
            if total > body.left() {
                return Err(bad());
            }
            counts.push(kept);
        }
    }
    let mut positions: Vec<Vec<usize>> = Vec::with_capacity(counts.len());
    for &kept in &counts {
        let mut here: Vec<usize> = Vec::with_capacity(kept);
        here.push(0);
        for _ in 1..kept {
            let p = match (sparse, here.last()) {
                (true, Some(&last)) => last.checked_add(body.size()?).and_then(|p| p.checked_add(1)).ok_or_else(bad)?,
                _ => here.len(),
            };
            if p > 2 * MAX_HARMONICS {
                return Err(bad());
            }
            here.push(p);
        }
        positions.push(here);
    }
    let mut curves: Vec<(usize, Vec<Complex>)> = positions
        .iter()
        .map(|kept| {
            let m = kept.last().map_or(1, |p| p.div_ceil(2)).max(1);
            (m, vec![Complex::ZERO; 2 * m + 1])
        })
        .collect();
    let mut next = curves.iter_mut();
    for &(_, _, count) in &layers {
        let mut last = (0i64, 0i64);
        for (m, c) in next.by_ref().take(count) {
            last = (last.0.wrapping_add(body.signed()?), last.1.wrapping_add(body.signed()?));
            c[*m] = Complex::new(last.0 as f64 * grid(movement), last.1 as f64 * grid(movement));
        }
    }
    let mut start = 1;
    for end in BANDS {
        for (kept, (m, c)) in positions.iter().zip(&mut curves) {
            let s = step(movement, kept.len());
            for &p in kept.iter().filter(|&&p| (start..end).contains(&p)) {
                let (re, im) = (body.signed()?, body.signed()?);
                c[(harmonic(p) + *m as i64) as usize] = Complex::new(re as f64 * s, im as f64 * s);
            }
        }
        start = end;
    }
    // Another layer, which nothing writes yet.
    if body.left() != 0 {
        return Err(other_version());
    }
    let alpha = (mode & WITH_ALPHA != 0).then_some(bases[ALPHA]);
    let mut drawing = Drawing { w, h, base: [bases[0], bases[1], bases[2]], alpha, table, masks: Vec::new() };
    let mut curves = curves.into_iter();
    let mut levels = [0; ALPHA + 1];
    for (plate, value, count) in layers {
        levels[plate] += 1;
        let mut layer = Vec::with_capacity(count);
        for (m, c) in curves.by_ref().take(count) {
            layer.push((points_along(&c, m)?, m, c));
        }
        drawing.masks.push((plate, levels[plate], value, layer));
    }
    Ok(drawing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::harmonics;

    fn wave(m: usize, gap: usize) -> Vec<Complex> {
        (0..2 * m + 1)
            .map(|i| {
                let k = i as i64 - m as i64;
                if k.unsigned_abs() as usize > 1 && i % gap != 0 {
                    return Complex::ZERO;
                }
                let size = 40.0 / (1.0 + k.abs() as f64);
                Complex::new(size * (k as f64 * 1.7).cos(), size * (k as f64 * 0.9).sin()) + Complex::new(0.001, 0.0)
            })
            .collect()
    }

    /// One curve of each kind: wavy and whole, and with most harmonics gone.
    fn drawing() -> Drawing {
        let (whole, sparse) = (600, 300);
        let mut thinned = wave(harmonics(sparse), 5);
        thinned[harmonics(sparse)] = Complex::new(120.0, 80.0);
        Drawing {
            w: 200,
            h: 120,
            base: [12.5, -3.25, 40.0],
            alpha: None,
            table: None, masks: vec![
                (0, 1, 33.25, vec![(whole, harmonics(whole), wave(harmonics(whole), 1))]),
                (2, 1, -8.5, vec![(sparse, harmonics(sparse), thinned), (whole, harmonics(whole), wave(harmonics(whole), 1))]),
            ],
        }
    }

    /// Only whole curves, of different lengths and places.
    fn whole_bands() -> Drawing {
        let curve = |m: usize, x: f64, y: f64| {
            let mut c = wave(m, 1);
            c[m] = Complex::new(x, y);
            (3 * m, m, c)
        };
        Drawing { w: 200, h: 120, base: [12.5, -3.25, 40.0], alpha: None, table: None, masks: vec![(0, 1, 33.25, vec![curve(40, 150.0, 90.0), curve(9, 20.0, 90.0), curve(100, 70.0, 10.0)])] }
    }

    /// A file with this compression and stream, as `pack` would lay them out.
    fn crafted(compression: u8, body: &[u8]) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&[compression, 10, 10]);
        out.extend_from_slice(&MIN_DICT.to_le_bytes());
        let mut writer = Lzma2Writer::new(out, Lzma2Options::with_preset(1));
        writer.write_all(body).unwrap();
        writer.finish().unwrap()
    }

    /// The start of a layer: its mode, its flags and a movement.
    fn start_of_layer(mode: u8, flags: u8) -> Vec<u8> {
        let mut body = vec![mode, flags];
        put_float(&mut body, 0.2);
        body
    }

    /// A plate of these base and levels, each level without curves.
    fn put_plate(body: &mut Vec<u8>, base: f64, levels: &[f64]) {
        put_float(body, base);
        put(body, levels.len() as u64);
        for value in levels {
            put_float(body, *value);
            put(body, 0);
        }
    }

    /// A layer of this mode, with `plates` plates and no curves.
    fn bare_layer(mode: u8, plates: usize) -> Vec<u8> {
        let mut body = start_of_layer(mode, 0);
        for plate in 0..plates {
            put_plate(&mut body, plate as f64, &[]);
        }
        body
    }

    /// Where the dictionary size is, after the magic, the compression and a
    /// canvas of 200 by 120.
    const DICT_AT: usize = 4 + 1 + 2 + 1;

    /// A packed file's stream, unpacked.
    fn inflated(bytes: &[u8]) -> Vec<u8> {
        inflate(&bytes[DICT_AT..]).unwrap()
    }

    /// c_k of a curve, whichever length of array it is held in.
    fn at_k((_, m, c): &Curve, k: i64) -> Complex {
        c.get((k + *m as i64) as usize).copied().unwrap_or(Complex::ZERO)
    }

    /// Each original curve and what came back for it, whose order may differ.
    fn pairs<'a>(a: &'a [Curve], b: &'a [Curve]) -> Vec<(&'a Curve, &'a Curve)> {
        assert_eq!(a.len(), b.len());
        a.iter()
            .map(|x| {
                let near = |y: &&Curve| (at_k(x, 0) - at_k(y, 0)).abs();
                (x, b.iter().min_by(|p, q| near(p).total_cmp(&near(q))).unwrap())
            })
            .collect()
    }

    #[test]
    fn the_rising_order_covers_every_harmonic_once() {
        assert_eq!((0..7).map(harmonic).collect::<Vec<_>>(), [0, -1, 1, -2, 2, -3, 3]);
        let order: Vec<i64> = (0..81).map(harmonic).collect();
        for k in -40..=40 {
            assert_eq!(order.iter().filter(|h| **h == k).count(), 1);
        }
    }

    #[test]
    fn numbers_survive_their_encoding() {
        let mut out = Vec::new();
        let values = [0, 1, -1, 63, -64, 64, 300, -70_000, i64::MAX / 2, i64::MIN / 2];
        for v in values {
            put_signed(&mut out, v);
        }
        let mut body = Body { bytes: &out, at: 0 };
        for v in values {
            assert_eq!(body.signed().unwrap(), v);
        }
        // Too long to be a number at all.
        let mut body = Body { bytes: &[0xFF; 11], at: 0 };
        assert!(body.number().is_err());
    }

    /// Everything in `original` came back, and within half a step.
    fn assert_restored(original: &Drawing, back: &Drawing, movement: f64) {
        let movement = f64::from(movement as f32);
        assert_eq!((back.w, back.h), (original.w, original.h));
        assert_eq!(back.base, original.base);
        assert_eq!(back.masks.len(), original.masks.len());
        for (a, b) in original.masks.iter().zip(&back.masks) {
            assert_eq!((a.0, a.2), (b.0, b.2));
            for (x, y) in pairs(&a.3, &b.3) {
                let kept = 1 + x.2.iter().enumerate().filter(|(i, v)| *i != x.1 && (v.re != 0.0 || v.im != 0.0)).count();
                let half = step(movement, kept) / 2.0 + 1e-9;
                let top = x.1 as i64;
                for k in -top..=top {
                    let (u, v) = (at_k(x, k), at_k(y, k));
                    if k == 0 {
                        assert!((u.re - v.re).abs() <= grid(movement) / 2.0 + 1e-9 && (u.im - v.im).abs() <= grid(movement) / 2.0 + 1e-9);
                        continue;
                    }
                    // A dropped harmonic comes back exactly zero.
                    if u.re == 0.0 && u.im == 0.0 {
                        assert_eq!((v.re, v.im), (0.0, 0.0));
                    }
                    assert!((u.re - v.re).abs() <= half && (u.im - v.im).abs() <= half, "k = {k}");
                }
            }
        }
    }

    #[test]
    fn unpacking_restores_the_structure_and_stays_within_half_a_step() {
        let (original, movement) = (drawing(), 0.2);
        let bytes = pack(&original, movement).unwrap();
        assert_eq!(inflated(&bytes)[1], FLAG_SPARSE);
        assert_restored(&original, &unpack(&bytes).unwrap(), movement);
        let original = whole_bands();
        let bytes = pack(&original, movement).unwrap();
        assert_eq!(inflated(&bytes)[1], 0);
        let back = unpack(&bytes).unwrap();
        assert_restored(&original, &back, movement);
        // A curve is held in no more room than it keeps.
        for (_, m, c) in &back.masks[0].3 {
            assert_eq!(c.len(), 2 * m + 1);
        }
    }

    #[test]
    fn a_levels_curves_go_by_where_they_sit_and_draw_the_same() {
        let original = whole_bands();
        let order: Vec<(f64, f64)> = kept(&original.masks[0].3).iter().map(|c| (c.at(0).re, c.at(0).im)).collect();
        assert_eq!(order, [(70.0, 10.0), (20.0, 90.0), (150.0, 90.0)]);
        let back = unpack(&pack(&original, 0.01).unwrap()).unwrap();
        let (a, b) = (original.bitmap(1).unwrap(), back.bitmap(1).unwrap());
        let worst = a.pixels().iter().zip(b.pixels()).map(|(p, q)| (p.r - q.r).abs().max((p.g - q.g).abs()).max((p.b - q.b).abs())).fold(0.0, f64::max);
        assert!(worst < 0.02, "{worst}");
    }

    #[test]
    fn a_curves_length_stands_in_for_its_points() {
        // A circle of radius 100, about 628 pixels around.
        let c = vec![Complex::ZERO, Complex::new(60.0, 60.0), Complex::new(100.0, 0.0)];
        let points = points_along(&c, 1).unwrap() as f64;
        assert!((points / (628.0 * POINTS_PER_PIXEL) - 1.0).abs() < 0.01, "{points}");
        // Absurd sizes stay bounded and never fail.
        let huge = vec![Complex::ZERO, Complex::ZERO, Complex::new(1e300, 0.0)];
        assert!(points_along(&huge, 1).unwrap() as f64 <= MAX_POINTS);
    }

    #[test]
    fn the_mode_says_whether_an_alpha_plate_follows() {
        let mut original = drawing();
        let without = pack(&original, 0.2).unwrap();
        assert_eq!(inflated(&without)[0], 0b0110_0000);
        assert_eq!(unpack(&without).unwrap().alpha, None);
        original.alpha = Some(0.25);
        original.masks.push((ALPHA, 1, 1.0, original.masks[0].3.clone()));
        let with = pack(&original, 0.2).unwrap();
        assert_eq!(inflated(&with)[0], 0b0111_0000);
        let back = unpack(&with).unwrap();
        assert_eq!(back.alpha, Some(0.25));
        assert_eq!((back.masks[2].0, back.masks[2].2), (ALPHA, 1.0));
    }

    #[test]
    fn a_layer_holds_as_many_plates_as_its_mode_says_in_order() {
        let mut body = start_of_layer(LAB_LAYER | WITH_ALPHA, 0);
        put_plate(&mut body, 1.0, &[10.0, 20.0]);
        put_plate(&mut body, 2.0, &[]);
        put_plate(&mut body, 3.0, &[30.0]);
        let three = body.clone();
        put_plate(&mut body, 0.5, &[1.0]);
        let back = unpack(&crafted(LZMA2, &body)).unwrap();
        assert_eq!((back.base, back.alpha), ([1.0, 2.0, 3.0], Some(0.5)));
        let levels: Vec<(usize, usize, f64)> = back.masks.iter().map(|m| (m.0, m.1, m.2)).collect();
        assert_eq!(levels, [(0, 1, 10.0), (0, 2, 20.0), (2, 1, 30.0), (ALPHA, 1, 1.0)]);
        // The alpha plate the mode promised is not there.
        assert!(unpack(&crafted(LZMA2, &three)).is_err());
        // Without alpha in the mode, a fourth plate would be a layer's worth of nonsense.
        body[0] = LAB_LAYER;
        assert!(unpack(&crafted(LZMA2, &body)).is_err());
        assert!(unpack(&crafted(LZMA2, &bare_layer(LAB_LAYER, 3))).unwrap().masks.is_empty());
    }

    /// A picture in a table of three colours: the second as the canvas, the
    /// third and then the first painted over it.
    fn in_a_table() -> Drawing {
        let mut drawing = drawing();
        drawing.base = [1.0, 0.0, 0.0];
        drawing.table = Some(vec![Color::rgb(0.875, 0.0625, 0.5), Color::rgb(0.0, 0.5, 1.0), Color::rgb(0.25, 0.75, 0.125)]);
        drawing.masks[0].2 = 2.0;
        drawing.masks[1] = (0, 2, 0.0, drawing.masks[1].3.clone());
        drawing
    }

    #[test]
    fn a_table_layer_keeps_its_colours_and_indices() {
        let original = in_a_table();
        let bytes = pack(&original, 0.2).unwrap();
        let stream = inflated(&bytes);
        assert_eq!((stream[0], stream[6]), (TABLE_LAYER, SRGB));
        let back = unpack(&bytes).unwrap();
        assert_eq!(back.table, original.table);
        assert_eq!((back.base[0], back.alpha), (1.0, None));
        let levels: Vec<(usize, usize, f64)> = back.masks.iter().map(|m| (m.0, m.1, m.2)).collect();
        assert_eq!(levels, [(0, 1, 2.0), (0, 2, 0.0)]);
        assert_restored(&original, &back, 0.2);
        // With an alpha plate after the table's own.
        let mut faded = in_a_table();
        faded.alpha = Some(0.5);
        faded.masks.push((ALPHA, 1, 1.0, faded.masks[0].3.clone()));
        let bytes = pack(&faded, 0.2).unwrap();
        assert_eq!(inflated(&bytes)[0], TABLE_LAYER | WITH_ALPHA);
        let back = unpack(&bytes).unwrap();
        assert_eq!((back.alpha, back.masks[2].0, back.masks[2].2), (Some(0.5), ALPHA, 1.0));
        // An index past the table is not written.
        let mut past = in_a_table();
        past.masks[0].2 = 3.0;
        assert!(pack(&past, 0.2).is_err());
    }

    #[test]
    fn a_table_is_refused_when_it_cannot_be_drawn() {
        let other = |bytes: &[u8]| matches!(unpack(bytes), Err(e) if e.to_string().contains("different version"));
        let table = |space: u8, colours: usize, base: u64| {
            let mut body = start_of_layer(TABLE_LAYER, 0);
            body.push(space);
            put(&mut body, colours as u64);
            for _ in 0..3 * colours {
                put_float(&mut body, 0.5);
            }
            put(&mut body, base);
            put(&mut body, 0);
            body
        };
        assert_eq!(unpack(&crafted(LZMA2, &table(SRGB, 2, 1))).unwrap().table.unwrap().len(), 2);
        // Colours of another space, or with alpha.
        assert!(other(&crafted(LZMA2, &table(LAB_LAYER, 2, 1))));
        assert!(other(&crafted(LZMA2, &table(SRGB | WITH_ALPHA, 2, 1))));
        // No colours, or an index past them.
        assert!(unpack(&crafted(LZMA2, &table(SRGB, 0, 0))).is_err());
        assert!(unpack(&crafted(LZMA2, &table(SRGB, 2, 2))).is_err());
        // More colours claimed than there are bytes for.
        let mut greedy = start_of_layer(TABLE_LAYER, 0);
        greedy.push(SRGB);
        put(&mut greedy, MAX_COLOURS as u64);
        assert!(unpack(&crafted(LZMA2, &greedy)).is_err());
    }

    #[test]
    fn what_is_not_known_is_refused_by_name() {
        let other = |bytes: &[u8]| matches!(unpack(bytes), Err(e) if e.to_string().contains("different version"));
        let lab = bare_layer(LAB_LAYER, 3);
        assert!(unpack(&crafted(LZMA2, &lab)).is_ok());
        // Earlier and later versions alike.
        assert!(other(b"FVC1 and whatever comes after"));
        assert!(other(b"FVC2 and whatever comes after"));
        assert!(other(b"FVC9 and whatever comes after"));
        assert!(other(&crafted(1, &lab)));
        // RGB, CMY, a model past those, and other spaces of Lab and of a table.
        for mode in [0x20, 0x30, 0x40, 0x50, 0x80, 0xE0, 0x61, 0x6F, 0x7F, 0x01, 0x1F] {
            assert!(other(&crafted(LZMA2, &bare_layer(mode, 3))), "{mode:#x}");
        }
        assert!(other(&crafted(LZMA2, &[start_of_layer(LAB_LAYER, 2).as_slice(), &lab[6..]].concat())));
        // A second layer.
        assert!(other(&crafted(LZMA2, &[lab.as_slice(), &lab].concat())));
    }

    #[test]
    fn the_curve_moves_by_about_the_asked_amount() {
        let original = drawing();
        let curve = &original.masks[0].3[0];
        let (n, m, c) = curve;
        let s = (2 * n).max(4 * m + 2).next_power_of_two();
        for movement in [0.05, 0.2, 0.8] {
            let back = unpack(&pack(&original, movement).unwrap()).unwrap();
            let (_, mb, y) = &back.masks[0].3[0];
            let (a, b) = (samples(c, *m, s).unwrap(), samples(y, *mb, s).unwrap());
            let rms = (a.iter().zip(&b).map(|(p, q)| (*p - *q).norm_squared()).sum::<f64>() / s as f64).sqrt();
            assert!(rms > movement * 0.6 && rms < movement * 1.4, "{rms} for {movement}");
        }
    }

    #[test]
    fn coarser_files_are_smaller() {
        let original = drawing();
        let size = |movement| pack(&original, movement).unwrap().len();
        assert!(size(0.8) < size(0.2) && size(0.2) < size(0.05));
    }

    #[test]
    fn damaged_files_are_refused_without_a_panic() {
        let good = pack(&drawing(), 0.2).unwrap();
        assert!(unpack(&good).is_ok());
        assert!(unpack(b"").is_err());
        assert!(unpack(b"FVC3").is_err());
        assert!(unpack(&good[..6]).is_err());
        // An absurd dictionary is not even allocated.
        let dict = DICT_AT..DICT_AT + 4;
        assert_eq!(good[dict.clone()], u32::try_from(inflated(&good).len()).unwrap().clamp(MIN_DICT, MAX_DICT).to_le_bytes());
        let mut huge = good.clone();
        huge[dict].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(unpack(&huge).is_err());
        // Nor is an absurd canvas believed.
        let mut wide = MAGIC.to_vec();
        wide.push(LZMA2);
        put(&mut wide, 1 << 40);
        put(&mut wide, 1 << 40);
        wide.extend_from_slice(&MIN_DICT.to_le_bytes());
        assert!(unpack(&wide).is_err());
        for cut in [9, good.len() / 2, good.len() - 1] {
            assert!(unpack(&good[..cut]).is_err(), "cut at {cut}");
        }
        // Damage anywhere either fails cleanly or yields some valid drawing.
        for i in (4..good.len()).step_by(good.len() / 97 + 1) {
            let mut flipped = good.clone();
            flipped[i] ^= 0x55;
            let _ = unpack(&flipped);
        }
        // Valid LZMA2 holding the wrong body.
        assert!(unpack(&crafted(LZMA2, &[LAB_LAYER, 0, 9, 9, 9])).is_err());
        // More curves claimed than there are bytes for is found before making room for them.
        let mut greedy = start_of_layer(LAB_LAYER, 0);
        put_float(&mut greedy, 1.0);
        put(&mut greedy, 1);
        put_float(&mut greedy, 1.0);
        put(&mut greedy, 2);
        put_plate(&mut greedy, 2.0, &[]);
        put_plate(&mut greedy, 3.0, &[]);
        put(&mut greedy, MAX_HARMONICS as u64);
        put(&mut greedy, MAX_HARMONICS as u64);
        assert!(unpack(&crafted(LZMA2, &greedy)).is_err());
    }
}
