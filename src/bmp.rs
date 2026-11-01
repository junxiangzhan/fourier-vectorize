//! Explicit little-endian 24-bit BI_RGB encoder and decoder; no native
//! struct serialization.
use crate::{Color, Error, Image};
use std::io::Write;

/// Reads an uncompressed 24-bit BMP (bottom-up or top-down) into floating
/// RGB in [0, 1]. Used to load the Part 4 paintings.
pub fn read(bytes: &[u8]) -> Result<Image<Color>, Error> {
    let u32_at = |i: usize| -> Result<u32, Error> {
        bytes
            .get(i..i + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .ok_or(Error::InvalidParameter("truncated BMP header"))
    };
    let u16_at = |i: usize| -> Result<u16, Error> {
        bytes
            .get(i..i + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .ok_or(Error::InvalidParameter("truncated BMP header"))
    };
    if bytes.get(0..2) != Some(b"BM") {
        return Err(Error::InvalidParameter("not a BMP file"));
    }
    let offset = u32_at(10)? as usize;
    let width = u32_at(18)? as i32;
    let height = u32_at(22)? as i32;
    // Only the format this program itself writes: 24 bits, no compression.
    if u16_at(28)? != 24 || u32_at(30)? != 0 || width <= 0 || height == 0 {
        return Err(Error::InvalidParameter(
            "only uncompressed 24-bit BMP is supported",
        ));
    }
    let (width, rows) = (width as usize, height.unsigned_abs() as usize);
    // Positive height stores the bottom row first.
    let bottom_up = height > 0;
    let stride = (width * 3 + 3) & !3;
    let mut image = Image::new(width, rows, Color::rgb(0.0, 0.0, 0.0))?;
    for row in 0..rows {
        let start = offset + row * stride;
        let line = bytes
            .get(start..start + width * 3)
            .ok_or(Error::InvalidParameter("truncated BMP pixel data"))?;
        let y = if bottom_up { rows - 1 - row } else { row };
        for (x, bgr) in line.chunks_exact(3).enumerate() {
            *image.get_mut(x, y).unwrap() = Color::rgb(
                bgr[2] as f64 / 255.0,
                bgr[1] as f64 / 255.0,
                bgr[0] as f64 / 255.0,
            );
        }
    }
    Ok(image)
}

/// Checks that the image fits the BMP format and returns
/// (pixel-data size, total file size) in bytes.
pub(crate) fn validate(image: &Image<Color>) -> Result<(u32, u32), Error> {
    if image.width() > i32::MAX as usize || image.height() > i32::MAX as usize {
        return Err(Error::SizeOverflow);
    }
    // Each row is 3 bytes per pixel, rounded up to a multiple of 4.
    let stride = image
        .width()
        .checked_mul(3)
        .and_then(|n| n.checked_add(3))
        .map(|n| n & !3)
        .ok_or(Error::SizeOverflow)?;
    let pixels = stride
        .checked_mul(image.height())
        .ok_or(Error::SizeOverflow)?;
    // 54 = 14-byte file header + 40-byte BITMAPINFOHEADER.
    let file = pixels.checked_add(54).ok_or(Error::SizeOverflow)?;
    let pixels = u32::try_from(pixels).map_err(|_| Error::SizeOverflow)?;
    let file = u32::try_from(file).map_err(|_| Error::SizeOverflow)?;
    if image.pixels().iter().any(|color| !color.is_finite()) {
        return Err(Error::NonFinite);
    }
    Ok((pixels, file))
}

pub fn write(image: &Image<Color>, writer: &mut impl Write) -> Result<(), Error> {
    let (pixel_size, file_size) = validate(image)?;
    // File header: signature, file size, two reserved u16, pixel-data offset.
    writer.write_all(b"BM")?;
    writer.write_all(&file_size.to_le_bytes())?;
    writer.write_all(&[0; 4])?;
    writer.write_all(&54u32.to_le_bytes())?;
    // Info header: header size, width, height (positive = bottom-up rows),
    // 1 color plane, 24 bits per pixel, BI_RGB (uncompressed), data size.
    writer.write_all(&40u32.to_le_bytes())?;
    writer.write_all(&(image.width() as i32).to_le_bytes())?;
    writer.write_all(&(image.height() as i32).to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&24u16.to_le_bytes())?;
    writer.write_all(&0u32.to_le_bytes())?;
    writer.write_all(&pixel_size.to_le_bytes())?;
    // Resolution (2 x i32) and palette counts (2 x u32): all zero.
    writer.write_all(&[0; 16])?;
    let padding = (4 - image.width() * 3 % 4) % 4;
    // Quantization happens only here: [0, 1] float -> 0..255 byte.
    let channel = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    // Bottom-up row order; each pixel is written as B, G, R.
    for y in (0..image.height()).rev() {
        for c in image.row(y).unwrap() {
            writer.write_all(&[channel(c.b), channel(c.g), channel(c.r)])?;
        }
        writer.write_all(&[0; 3][..padding])?;
    }
    Ok(())
}
