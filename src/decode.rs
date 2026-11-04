//! Reads a picture file of any supported format into floating RGB.
use crate::{bmp, Color, Error, Image};
use ::image::ImageFormat;

/// Decodes a BMP, PNG or JPEG, told apart by their contents rather than the
/// file name. PNG and JPEG are taken as stored: alpha, embedded colour
/// profiles and EXIF orientation are all ignored.
pub(crate) fn decode(bytes: &[u8]) -> Result<Image<Color>, Error> {
    if bytes.starts_with(b"BM") {
        return bmp::read(bytes);
    }
    match ::image::guess_format(bytes) {
        Ok(ImageFormat::Png) => {}
        Ok(ImageFormat::Jpeg) if jpeg_is_complete(bytes) => {}
        Ok(ImageFormat::Jpeg) => return Err(Error::InvalidParameter("truncated JPEG: it has no end-of-image marker")),
        _ => return Err(Error::InvalidParameter("unsupported image format; fit reads BMP, PNG and JPEG")),
    }
    let decoded = ::image::load_from_memory(bytes).map_err(|e| Error::Decode(e.to_string()))?;
    // 16 bits per channel keeps a 16-bit PNG's precision. An 8-bit value v
    // widens to v * 257, and v * 257 / 65535 is the same f64 as the BMP
    // reader's v / 255.
    let decoded = decoded.into_rgb16();
    let mut image = Image::new(decoded.width() as usize, decoded.height() as usize, Color::rgb(0.0, 0.0, 0.0))?;
    let channel = |v: u16| v as f64 / 65535.0;
    for (x, y, pixel) in decoded.enumerate_pixels() {
        let [r, g, b] = pixel.0;
        *image.get_mut(x as usize, y as usize).unwrap() = Color::rgb(channel(r), channel(g), channel(b));
    }
    Ok(image)
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
        let image = decode(&png(&wide, ExtendedColorType::Rgb16)).unwrap();
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
        let image = decode(&jpeg()).unwrap();
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

    #[test]
    fn anything_else_is_refused() {
        let png = png(&all_values(), ExtendedColorType::Rgb8);
        for bytes in [&b""[..], b"BM", b"GIF89a and more", b"plain text", &png[..png.len() / 2]] {
            assert!(decode(bytes).is_err(), "{bytes:?}");
        }
    }
}
