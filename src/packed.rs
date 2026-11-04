//! Packed coefficients: a compact binary form of a coefficients file that also
//! holds the canvas size and base levels, so it stands alone.
//!
//! Quantizing is what makes it small, and the only lossy step. Each curve has
//! one step, from its count of kept coefficients K: every coefficient rounded
//! to a step s moves the curve's points by an RMS of s * sqrt(K / 6), so
//! s = movement * sqrt(6 / K) moves every curve by the same amount, however
//! long. Measured on two paintings, that beat one step for all curves, and
//! beat rounding small values to zero more readily.
//!
//! The numbers are varints with |k| rising, compressed with LZMA2. Other byte
//! layouts changed the size by about 1%, and a model of an adaptive arithmetic
//! coder by at most 2%, so nothing more elaborate is used.
use crate::{drawing::Drawing, fit::harmonics, Complex, Error};
use lzma_rust2::{Lzma2Options, Lzma2Reader, Lzma2Writer};
use std::io::{Read, Write};

/// The start of every packed file: the format and its version.
pub(crate) const MAGIC: &[u8; 4] = b"FVC1";
const PRESET: u32 = 9;
/// The largest dictionary a file may ask a reader to allocate.
const MAX_DICT: u32 = 64 << 20;
const MIN_DICT: u32 = 1 << 16;

fn bad() -> Error {
    Error::InvalidParameter("not a valid packed coefficients file")
}

fn step(movement: f64, kept: usize) -> f64 {
    movement * (6.0 / kept.max(1) as f64).sqrt()
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
    out.extend_from_slice(&value.to_le_bytes());
}

/// `drawing` as a packed file's bytes; a coefficient that is exactly zero is
/// left out, as pruned ones are.
pub(crate) fn pack(drawing: &Drawing, movement: f64) -> Result<Vec<u8>, Error> {
    let (mut head, mut gaps, mut values) = (Vec::new(), Vec::new(), Vec::new());
    put(&mut head, drawing.w as u64);
    put(&mut head, drawing.h as u64);
    for base in drawing.base {
        put_float(&mut head, base);
    }
    put_float(&mut head, movement);
    put(&mut head, drawing.masks.len() as u64);
    for (channel, level, value, curves) in &drawing.masks {
        head.push(*channel as u8);
        put(&mut head, *level as u64);
        put_float(&mut head, *value);
        put(&mut head, curves.len() as u64);
        for (n, m, c) in curves {
            let live: Vec<usize> = (0..c.len())
                .filter(|&p| {
                    let v = c[(harmonic(p) + *m as i64) as usize];
                    v.re != 0.0 || v.im != 0.0
                })
                .collect();
            put(&mut head, *n as u64);
            put(&mut head, live.len() as u64);
            if live.len() != c.len() {
                let mut next = 0;
                for &p in &live {
                    put(&mut gaps, (p - next) as u64);
                    next = p + 1;
                }
            }
            let s = step(movement, live.len());
            for &p in &live {
                let v = c[(harmonic(p) + *m as i64) as usize];
                put_signed(&mut values, (v.re / s).round() as i64);
                put_signed(&mut values, (v.im / s).round() as i64);
            }
        }
    }
    head.extend_from_slice(&gaps);
    head.extend_from_slice(&values);
    let mut options = Lzma2Options::with_preset(PRESET);
    let dict = u32::try_from(head.len()).unwrap_or(MAX_DICT).clamp(MIN_DICT, MAX_DICT);
    options.lzma_options.dict_size = dict;
    let mut out = MAGIC.to_vec();
    out.extend_from_slice(&dict.to_le_bytes());
    let mut writer = Lzma2Writer::new(out, options);
    writer.write_all(&head)?;
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
    fn float(&mut self) -> Result<f64, Error> {
        let end = self.at.checked_add(8).ok_or_else(bad)?;
        let bytes = self.bytes.get(self.at..end).ok_or_else(bad)?;
        self.at = end;
        let value = f64::from_le_bytes(bytes.try_into().map_err(|_| bad())?);
        if value.is_finite() {
            Ok(value)
        } else {
            Err(bad())
        }
    }
}

/// The drawing a packed file holds.
pub(crate) fn unpack(bytes: &[u8]) -> Result<Drawing, Error> {
    let rest = bytes.strip_prefix(MAGIC).ok_or_else(bad)?;
    let (dict, stream) = rest.split_first_chunk::<4>().ok_or_else(bad)?;
    let dict = u32::from_le_bytes(*dict);
    if !(MIN_DICT..=MAX_DICT).contains(&dict) {
        return Err(bad());
    }
    let mut unpacked = Vec::new();
    Lzma2Reader::new(stream, dict, None).read_to_end(&mut unpacked).map_err(|_| bad())?;
    let mut body = Body { bytes: &unpacked, at: 0 };
    let (w, h) = (body.size()?, body.size()?);
    if w == 0 || h == 0 {
        return Err(bad());
    }
    let base = [body.float()?, body.float()?, body.float()?];
    let movement = body.float()?;
    if movement <= 0.0 {
        return Err(bad());
    }
    let mut drawing = Drawing { w, h, base, masks: Vec::new() };
    // Curves whose kept harmonics come later: mask, curve and kept count.
    let mut shape = Vec::new();
    for i in 0..body.size()? {
        let channel = usize::from(body.byte()?);
        if channel > 2 {
            return Err(bad());
        }
        let (level, value) = (body.size()?, body.float()?);
        let mut curves = Vec::new();
        for j in 0..body.size()? {
            let (n, kept) = (body.size()?, body.size()?);
            let m = harmonics(n);
            if n == 0 || kept > 2 * m + 1 {
                return Err(bad());
            }
            curves.push((n, m, vec![Complex::ZERO; 2 * m + 1]));
            shape.push((i, j, kept));
        }
        drawing.masks.push((channel, level, value, curves));
    }
    let mut live = Vec::with_capacity(shape.len());
    for &(i, j, kept) in &shape {
        let total = drawing.masks[i].3[j].2.len();
        let mut positions = Vec::with_capacity(kept);
        if kept == total {
            positions.extend(0..total);
        } else {
            let mut next = 0usize;
            for _ in 0..kept {
                let p = next.checked_add(body.size()?).ok_or_else(bad)?;
                if p >= total {
                    return Err(bad());
                }
                positions.push(p);
                next = p + 1;
            }
        }
        live.push(positions);
    }
    for (&(i, j, kept), positions) in shape.iter().zip(&live) {
        let s = step(movement, kept);
        let (_, m, c) = &mut drawing.masks[i].3[j];
        for &p in positions {
            let (re, im) = (body.signed()?, body.signed()?);
            c[(harmonic(p) + *m as i64) as usize] = Complex::new(re as f64 * s, im as f64 * s);
        }
    }
    if body.at != unpacked.len() {
        return Err(bad());
    }
    Ok(drawing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::samples;

    /// One curve of each kind: wavy and whole, and with most harmonics gone.
    fn drawing() -> Drawing {
        let wave = |m: usize, gap: usize| -> Vec<Complex> {
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
        };
        let (whole, sparse) = (600, 300);
        Drawing {
            w: 200,
            h: 120,
            base: [12.5, -3.25, 40.0],
            masks: vec![
                (0, 1, 33.25, vec![(whole, harmonics(whole), wave(harmonics(whole), 1))]),
                (2, 7, -8.5, vec![(sparse, harmonics(sparse), wave(harmonics(sparse), 5)), (whole, harmonics(whole), wave(harmonics(whole), 1))]),
            ],
        }
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

    #[test]
    fn unpacking_restores_the_structure_and_stays_within_half_a_step() {
        let (original, movement) = (drawing(), 0.2);
        let back = unpack(&pack(&original, movement).unwrap()).unwrap();
        assert_eq!((back.w, back.h, back.base), (original.w, original.h, original.base));
        assert_eq!(back.masks.len(), original.masks.len());
        for (a, b) in original.masks.iter().zip(&back.masks) {
            assert_eq!((a.0, a.1, a.2, a.3.len()), (b.0, b.1, b.2, b.3.len()));
            for ((n, m, x), (nb, mb, y)) in a.3.iter().zip(&b.3) {
                assert_eq!((n, m, x.len()), (nb, mb, y.len()));
                let kept = x.iter().filter(|v| v.re != 0.0 || v.im != 0.0).count();
                let half = step(movement, kept) / 2.0 + 1e-12;
                for (u, v) in x.iter().zip(y) {
                    // A dropped harmonic comes back exactly zero.
                    if u.re == 0.0 && u.im == 0.0 {
                        assert_eq!((v.re, v.im), (0.0, 0.0));
                    }
                    assert!((u.re - v.re).abs() <= half && (u.im - v.im).abs() <= half);
                }
            }
        }
    }

    #[test]
    fn the_curve_moves_by_about_the_asked_amount() {
        let original = drawing();
        let (n, m, c) = &original.masks[0].3[0];
        let s = (2 * n).max(4 * m + 2).next_power_of_two();
        for movement in [0.05, 0.2, 0.8] {
            let back = unpack(&pack(&original, movement).unwrap()).unwrap();
            let (a, b) = (samples(c, *m, s).unwrap(), samples(&back.masks[0].3[0].2, *m, s).unwrap());
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
        assert!(unpack(b"FVC2....").is_err());
        assert!(unpack(&good[..6]).is_err());
        // An absurd dictionary is not even allocated.
        let mut huge = good.clone();
        huge[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(unpack(&huge).is_err());
        for cut in [9, good.len() / 2, good.len() - 1] {
            assert!(unpack(&good[..cut]).is_err(), "cut at {cut}");
        }
        // Damage anywhere either fails cleanly or yields some valid drawing.
        for i in (8..good.len()).step_by(good.len() / 97 + 1) {
            let mut flipped = good.clone();
            flipped[i] ^= 0x55;
            let _ = unpack(&flipped);
        }
        // Valid LZMA2 holding the wrong body.
        let mut writer = Lzma2Writer::new(b"FVC1\x00\x00\x01\x00".to_vec(), Lzma2Options::with_preset(1));
        writer.write_all(&[1, 0, 9, 9, 9]).unwrap();
        assert!(unpack(&writer.finish().unwrap()).is_err());
    }
}
