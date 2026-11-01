use multimedia::{bmp, Color, Image, Point2};
use std::io;

#[test]
fn user_can_convert_coverage_to_color_and_write_a_padded_bitmap() {
    assert_eq!(
        Color::rgb(1.0, 0.0, 0.0).mix(Color::rgb(0.0, 0.0, 1.0), 0.5),
        Color::rgb(0.5, 0.0, 0.5)
    );
    let mut coverage = Image::new(1, 2, 0.0).unwrap();
    *coverage.get_mut(0, 1).unwrap() = 1.0;
    let colors = coverage
        .map(|a| Color::rgb(1.0, 0.0, 0.0).mix(Color::rgb(0.0, 0.0, 1.0), *a))
        .unwrap();
    let mut bytes = Vec::new();
    bmp::write(&colors, &mut bytes).unwrap();
    assert_eq!(&bytes[0..2], b"BM");
    assert_eq!(bytes.len(), 62);
    // Bottom blue row, then top red row; each has one padding byte.
    assert_eq!(&bytes[54..], &[255, 0, 0, 0, 0, 0, 255, 0]);
    assert_eq!(coverage.row(1).unwrap(), &[1.0]);
    assert!(coverage.get(1, 0).is_none());
}

#[test]
fn invalid_images_and_failed_output_are_reported_without_panics() {
    assert!(Image::new(0, 2, 0.0).is_err());
    assert!(Image::new(usize::MAX, 2, 0.0).is_err());
    let bad = Image::new(1, 1, Color::rgb(f64::NAN, 0.0, 0.0)).unwrap();
    let mut bytes = Vec::new();
    assert!(bmp::write(&bad, &mut bytes).is_err());
    assert!(bytes.is_empty());
    struct FailingWriter;
    impl io::Write for FailingWriter {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("full"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let image = Image::new(1, 1, Color::rgb(-1.0, 0.5, 2.0)).unwrap();
    assert!(bmp::write(&image, &mut FailingWriter).is_err());
    bmp::write(&image, &mut bytes).unwrap();
    assert_eq!(&bytes[54..], &[255, 128, 0, 0]);
}

#[test]
fn numeric_images_average_without_quantizing_and_reject_invalid_blocks() {
    let mut image = Image::new(2, 2, 0.0).unwrap();
    *image.get_mut(0, 0).unwrap() = 1.0;
    *image.get_mut(1, 1).unwrap() = 1.0;
    assert_eq!(image.mean_downsample(2).unwrap().pixels(), &[0.5]);
    assert_eq!(image.mean_downsample(1).unwrap(), image);
    assert!(image.mean_downsample(0).is_err());
    assert!(image.mean_downsample(3).is_err());
    *image.get_mut(0, 0).unwrap() = f64::INFINITY;
    assert!(image.mean_downsample(2).is_err());
}

#[test]
fn bitmap_rotation_moves_pixels_on_grid_and_interpolates_between_them() {
    let center = Point2::new(2.0, 2.0);
    let mut image = Image::new(4, 4, 0.0).unwrap();
    *image.get_mut(0, 0).unwrap() = 1.0;
    // Zero angle is the identity.
    let same = image.rotated_bilinear(center, 0.0).unwrap();
    assert!(same
        .pixels()
        .iter()
        .zip(image.pixels())
        .all(|(a, b)| (a - b).abs() < 1e-12));
    // A quarter turn maps pixel centers onto pixel centers: screen-clockwise,
    // the top-left pixel lands in the top-right corner.
    let turned = image
        .rotated_bilinear(center, std::f64::consts::FRAC_PI_2)
        .unwrap();
    for (i, value) in turned.pixels().iter().enumerate() {
        let expected = if i == 3 { 1.0 } else { 0.0 };
        assert!((value - expected).abs() < 1e-12, "pixel {i}: {value}");
    }
    // Off-grid angles blend the lit pixel into several neighbours.
    let mut dot = Image::new(5, 5, 0.0).unwrap();
    *dot.get_mut(2, 1).unwrap() = 1.0;
    let blurred = dot.rotated_bilinear(Point2::new(2.5, 2.5), 0.3).unwrap();
    assert!(blurred.pixels().iter().filter(|v| **v > 0.0).count() > 1);
    assert!(image.rotated_bilinear(center, f64::NAN).is_err());
}

#[test]
fn bitmaps_written_by_the_encoder_read_back_exactly() {
    let mut image = Image::new(3, 2, Color::rgb(0.0, 0.0, 0.0)).unwrap();
    for (i, v) in [0.0, 51.0, 102.0, 153.0, 204.0, 255.0].iter().enumerate() {
        *image.get_mut(i % 3, i / 3).unwrap() = Color::rgb(v / 255.0, 1.0 - v / 255.0, 0.2);
    }
    let mut bytes = Vec::new();
    bmp::write(&image, &mut bytes).unwrap();
    let back = bmp::read(&bytes).unwrap();
    assert_eq!((back.width(), back.height()), (3, 2));
    for (a, b) in back.pixels().iter().zip(image.pixels()) {
        assert!((a.r - b.r).abs() < 1e-12 && (a.g - b.g).abs() < 1e-12);
        assert!((a.b - (51.0 / 255.0)).abs() < 1e-12);
    }
    assert!(bmp::read(b"not a bitmap").is_err());
    assert!(bmp::read(&bytes[..60]).is_err());
}
