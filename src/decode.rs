//! Reads a picture file of any supported format into floating RGB.
use crate::{bmp, Color, Error, Image};
use ::image::{ImageDecoder, ImageFormat, ImageReader};
use std::io::Cursor;

/// Decodes a BMP, PNG or JPEG, told apart by their contents rather than the
/// file name, into its colours as stored and, if any pixel is not fully opaque,
/// its alpha. EXIF orientation is applied; embedded colour profiles are ignored.
pub(crate) fn decode(bytes: &[u8]) -> Result<(Image<Color>, Option<Image<f64>>), Error> {
    if bytes.starts_with(b"BM") {
        return Ok((bmp::read(bytes)?, None));
    }
    match ::image::guess_format(bytes) {
        Ok(ImageFormat::Png) => {}
        Ok(ImageFormat::Jpeg) if jpeg_is_complete(bytes) => {}
        Ok(ImageFormat::Jpeg) => return Err(Error::InvalidParameter("truncated JPEG: it has no end-of-image marker")),
        _ => return Err(Error::InvalidParameter("unsupported image format; fit reads BMP, PNG and JPEG")),
    }
    let failed = |e: ::image::ImageError| Error::Decode(e.to_string());
    let reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let orientation = reader.into_decoder().and_then(|mut d| d.orientation()).map_err(failed)?;
    let mut decoded = ::image::load_from_memory(bytes).map_err(failed)?;
    decoded.apply_orientation(orientation);
    // 16 bits per channel keeps a 16-bit PNG's precision. An 8-bit value v
    // widens to v * 257, and v * 257 / 65535 is the same f64 as the BMP
    // reader's v / 255.
    let decoded = decoded.into_rgba16();
    let mut image = Image::new(decoded.width() as usize, decoded.height() as usize, Color::rgb(0.0, 0.0, 0.0))?;
    let mut alpha = Image::new(image.width(), image.height(), 1.0)?;
    let mut transparent = false;
    let channel = |v: u16| v as f64 / 65535.0;
    for (x, y, pixel) in decoded.enumerate_pixels() {
        let [r, g, b, a] = pixel.0;
        transparent |= a < u16::MAX;
        *image.get_mut(x as usize, y as usize).unwrap() = Color::rgb(channel(r), channel(g), channel(b));
        *alpha.get_mut(x as usize, y as usize).unwrap() = channel(a);
    }
    Ok((image, transparent.then_some(alpha)))
}

/// Whether a JPEG reaches its end-of-image marker. The decoder is lenient and
/// fills in whatever a cut-off file lacks, so it cannot tell us this itself.
fn jpeg_is_complete(bytes: &[u8]) -> bool {
    // Segments before the first scan may hold anything, even a thumbnail with
    // its own end marker, so step over them by their lengths.
    let mut at = 2;
    loop {
        match bytes.get(at..at + 4) {
            Some([0xFF, 0xDA, ..]) => break,
            Some([0xFF, 0xFF, ..]) => at += 1,
            Some([0xFF, _, high, low]) => at += 2 + u16::from_be_bytes([*high, *low]) as usize,
            Some(_) => at += 1,
            None => return false,
        }
    }
    // Scan data escapes every 0xFF byte, so this pair can only be the marker.
    bytes[at..].windows(2).any(|pair| pair == [0xFF, 0xD9])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::image::{ExtendedColorType, ImageEncoder};

    /// Every 8-bit value, in each channel, as a 256 x 3 picture.
    fn all_values() -> Vec<u8> {
        (0..3).flat_map(|c: usize| (0..=255u8).flat_map(move |v| [v, 255 - v, v.wrapping_mul(7)].map(|t| t.wrapping_add(c as u8)))).collect()
    }

    fn png(pixels: &[u8], color: ExtendedColorType) -> Vec<u8> {
        let mut bytes = Vec::new();
        ::image::codecs::png::PngEncoder::new(&mut bytes).write_image(pixels, 256, 3, color).unwrap();
        bytes
    }

    fn bmp_of(pixels: &[u8]) -> Vec<u8> {
        let mut picture = Image::new(256, 3, Color::rgb(0.0, 0.0, 0.0)).unwrap();
        for (i, rgb) in pixels.chunks_exact(3).enumerate() {
            *picture.get_mut(i % 256, i / 256).unwrap() = Color::rgb(rgb[0] as f64 / 255.0, rgb[1] as f64 / 255.0, rgb[2] as f64 / 255.0);
        }
        let mut bytes = Vec::new();
        bmp::write(&picture, &mut bytes).unwrap();
        bytes
    }

    #[test]
    fn png_pixels_equal_bmp_pixels_exactly() {
        let pixels = all_values();
        let expected = decode(&bmp_of(&pixels)).unwrap();
        assert!(expected.1.is_none());
        assert_eq!(decode(&png(&pixels, ExtendedColorType::Rgb8)).unwrap(), expected);
        // Opaque alpha and the same values widened to 16 bits change nothing.
        let rgba: Vec<u8> = pixels.chunks_exact(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect();
        assert_eq!(decode(&png(&rgba, ExtendedColorType::Rgba8)).unwrap(), expected);
        // v * 257 is v in both bytes, whichever byte order the encoder expects.
        let wide: Vec<u8> = pixels.iter().flat_map(|v| [*v, *v]).collect();
        assert_eq!(decode(&png(&wide, ExtendedColorType::Rgb16)).unwrap(), expected);
    }

    #[test]
    fn a_16_bit_png_keeps_its_precision() {
        // The encoder takes samples in native byte order.
        let (r, g, b) = (0x0123u16, 0x4567u16, 0x89abu16);
        let mut wide = vec![0u8; 256 * 3 * 3 * 2];
        for (i, sample) in [r, g, b].iter().enumerate() {
            wide[i * 2..i * 2 + 2].copy_from_slice(&sample.to_ne_bytes());
        }
        let (image, _) = decode(&png(&wide, ExtendedColorType::Rgb16)).unwrap();
        let expected = Color::rgb(r as f64 / 65535.0, g as f64 / 65535.0, b as f64 / 65535.0);
        assert_eq!(*image.get(0, 0).unwrap(), expected);
    }

    fn jpeg() -> Vec<u8> {
        let mut bytes = Vec::new();
        ::image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 100).write_image(&all_values(), 256, 3, ExtendedColorType::Rgb8).unwrap();
        bytes
    }

    #[test]
    fn jpeg_decodes_to_the_same_size() {
        let (image, _) = decode(&jpeg()).unwrap();
        assert_eq!((image.width(), image.height()), (256, 3));
    }

    #[test]
    fn a_jpeg_cut_off_anywhere_is_refused() {
        let bytes = jpeg();
        for len in 0..bytes.len() - 1 {
            assert!(decode(&bytes[..len]).is_err(), "accepted the first {len} of {} bytes", bytes.len());
        }
    }

    #[test]
    fn a_thumbnail_end_marker_does_not_make_a_cut_off_jpeg_complete() {
        let whole = jpeg();
        // An APP1 segment holding an end marker, right after the start marker.
        let mut bytes = whole[..2].to_vec();
        bytes.extend_from_slice(&[0xFF, 0xE1, 0x00, 0x04, 0xFF, 0xD9]);
        bytes.extend_from_slice(&whole[2..]);
        assert!(decode(&bytes).is_ok());
        assert!(decode(&bytes[..bytes.len() - 2]).is_err());
    }

    /// A TIFF block holding only an EXIF orientation.
    fn exif(orientation: u8) -> Vec<u8> {
        let mut block = b"MM\0*\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x01\0".to_vec();
        block.extend_from_slice(&[orientation, 0, 0, 0, 0, 0, 0]);
        block
    }

    /// Black at the top left, white everywhere else.
    fn corner(x: u32, y: u32) -> [u8; 3] {
        if (x, y) == (0, 0) {
            [0; 3]
        } else {
            [255; 3]
        }
    }

    /// Where the black corner ended up, and the size of the picture.
    fn black_at(image: &Image<Color>) -> (usize, usize, usize, usize) {
        let at = image.pixels().iter().position(|c| c.r < 0.5).unwrap();
        (at % image.width(), at / image.width(), image.width(), image.height())
    }

    #[test]
    fn exif_orientation_turns_a_png() {
        let pixels: Vec<u8> = (0..3).flat_map(|y| (0..2).flat_map(move |x| corner(x, y))).collect();
        // EXIF value: where the black top left corner of a 2 x 3 picture goes.
        for (value, expected) in [
            (1, (0, 0, 2, 3)),
            (2, (1, 0, 2, 3)),
            (3, (1, 2, 2, 3)),
            (4, (0, 2, 2, 3)),
            (5, (0, 0, 3, 2)),
            (6, (2, 0, 3, 2)),
            (7, (2, 1, 3, 2)),
            (8, (0, 1, 3, 2)),
        ] {
            let mut bytes = Vec::new();
            let mut encoder = ::image::codecs::png::PngEncoder::new(&mut bytes);
            encoder.set_exif_metadata(exif(value)).unwrap();
            encoder.write_image(&pixels, 2, 3, ExtendedColorType::Rgb8).unwrap();
            assert_eq!(black_at(&decode(&bytes).unwrap().0), expected, "orientation {value}");
        }
    }

    #[test]
    fn exif_orientation_turns_a_jpeg() {
        // Large enough blocks that the corner survives compression.
        let (w, h) = (32u32, 48u32);
        let pixels: Vec<u8> = (0..h).flat_map(|y| (0..w).flat_map(move |x| corner(x / 16, y / 16))).collect();
        let mut plain = Vec::new();
        ::image::codecs::jpeg::JpegEncoder::new_with_quality(&mut plain, 95).write_image(&pixels, w, h, ExtendedColorType::Rgb8).unwrap();
        let (upright, _) = decode(&plain).unwrap();
        assert_eq!((upright.width(), upright.height()), (32, 48));
        assert!(upright.get(4, 4).unwrap().r < 0.5);
        // The way a camera stores it: an APP1 segment right after the start.
        let block = [b"Exif\0\0".to_vec(), exif(6)].concat();
        let mut bytes = plain[..2].to_vec();
        bytes.extend_from_slice(&[0xFF, 0xE1]);
        bytes.extend_from_slice(&(block.len() as u16 + 2).to_be_bytes());
        bytes.extend_from_slice(&block);
        bytes.extend_from_slice(&plain[2..]);
        let (turned, _) = decode(&bytes).unwrap();
        assert_eq!((turned.width(), turned.height()), (48, 32));
        assert!(turned.get(43, 4).unwrap().r < 0.5);
        assert!(turned.get(4, 4).unwrap().r > 0.5);
    }

    #[test]
    fn alpha_comes_back_beside_the_stored_colours() {
        // Clear black, half-clear red, opaque blue, opaque white.
        let pixels = [0, 0, 0, 0, 255, 0, 0, 51, 0, 0, 255, 255, 255, 255, 255, 255];
        let mut bytes = Vec::new();
        ::image::codecs::png::PngEncoder::new(&mut bytes).write_image(&pixels, 2, 2, ExtendedColorType::Rgba8).unwrap();
        let (image, alpha) = decode(&bytes).unwrap();
        let alpha = alpha.unwrap();
        assert_eq!(alpha.pixels(), [0.0, 0.2, 1.0, 1.0]);
        assert_eq!(*image.get(1, 0).unwrap(), Color::rgb(1.0, 0.0, 0.0));
        // Grey with alpha takes the same path.
        let mut bytes = Vec::new();
        ::image::codecs::png::PngEncoder::new(&mut bytes).write_image(&[255, 0], 1, 1, ExtendedColorType::La8).unwrap();
        let (image, alpha) = decode(&bytes).unwrap();
        assert_eq!(alpha.unwrap().pixels(), [0.0]);
        assert_eq!(*image.get(0, 0).unwrap(), Color::rgb(1.0, 1.0, 1.0));
    }

    #[test]
    fn anything_else_is_refused() {
        let png = png(&all_values(), ExtendedColorType::Rgb8);
        for bytes in [&b""[..], b"BM", b"GIF89a and more", b"plain text", &png[..png.len() / 2]] {
            assert!(decode(bytes).is_err(), "{bytes:?}");
        }
    }
}
