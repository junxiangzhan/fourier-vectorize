//! FFT, fill and bitmap behaviour, through the public library.
use fourier_vectorize::{bmp, fft, Color, Complex, Image, Rasterizer};
use std::f64::consts::TAU;

fn contour(points: &[(f64, f64)]) -> Vec<Complex> {
    points.iter().map(|&(x, y)| Complex::new(x, y)).collect()
}

#[test]
fn fft_matches_the_direct_sum_and_inverts_itself() {
    let input: Vec<Complex> = (0..8).map(|n| Complex::new((n * n) as f64 * 0.5 - 3.0, n as f64 % 3.0)).collect();
    let mut fast = input.clone();
    fft(&mut fast, false).unwrap();
    // X[f] = sum_n x[n] e^(-2 pi i f n / N), summed directly.
    for (f, value) in fast.iter().enumerate() {
        let direct = input.iter().enumerate().fold(Complex::ZERO, |sum, (n, x)| sum + *x * Complex::from_angle(-TAU * (f * n) as f64 / 8.0));
        assert!((*value - direct).abs() < 1e-12, "bin {f}");
    }
    fft(&mut fast, true).unwrap();
    for (a, b) in fast.iter().zip(&input) {
        assert!((*a - *b).abs() < 1e-12);
    }
    assert!(fft(&mut [Complex::ZERO; 6], false).is_err());
    assert!(fft(&mut [Complex::new(f64::NAN, 0.0)], false).is_err());
}

#[test]
fn nonzero_winding_fills_a_self_intersecting_star_center() {
    let star = contour(&[(3.0, 0.0), (4.76336, 5.42705), (0.14683, 2.07295), (5.85317, 2.07295), (1.23664, 5.42705)]);
    let image = Rasterizer::new(6, 6).unwrap().render(&[star], 1).unwrap();
    assert_eq!(*image.get(3, 3).unwrap(), 1.0);
}

#[test]
fn pixel_centers_shared_vertices_and_off_canvas_edges_have_consistent_fill() {
    let triangle = contour(&[(-1.0, -1.0), (4.0, -1.0), (4.0, 4.0)]);
    let image = Rasterizer::new(3, 3).unwrap().render(std::slice::from_ref(&triangle), 1).unwrap();
    assert_eq!(image.pixels(), &[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    let surround = contour(&[(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)]);
    assert_eq!(Rasterizer::new(2, 2).unwrap().render(&[surround], 1).unwrap().pixels(), &[1.0; 4]);
    let pixel_edge = contour(&[(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)]);
    assert_eq!(Rasterizer::new(3, 3).unwrap().render(&[pixel_edge], 1).unwrap().pixels(), &[1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    assert!(Rasterizer::new(0, 2).is_err());
    assert!(Rasterizer::new(usize::MAX, 2).is_err());
    let mut non_finite = triangle;
    non_finite[0] = Complex::new(f64::NAN, 0.0);
    assert!(Rasterizer::new(2, 2).unwrap().render(&[non_finite], 1).is_err());
    let repeated = contour(&[(0.0, 0.0), (2.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]);
    assert_eq!(Rasterizer::new(2, 2).unwrap().render(&[repeated], 1).unwrap().pixels(), &[1.0; 4]);
}

#[test]
fn reversing_only_the_inner_contour_changes_the_hole() {
    let outer = contour(&[(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0)]);
    let inner = contour(&[(1.0, 1.0), (5.0, 1.0), (5.0, 5.0), (1.0, 5.0)]);
    let reversed: Vec<Complex> = inner.iter().rev().copied().collect();
    let raster = Rasterizer::new(6, 6).unwrap();
    assert_eq!(*raster.render(&[outer.clone(), inner], 1).unwrap().get(2, 2).unwrap(), 1.0);
    assert_eq!(*raster.render(&[outer, reversed], 1).unwrap().get(2, 2).unwrap(), 0.0);
}

#[test]
fn supersampling_returns_area_fraction_at_original_dimensions() {
    let quarter = [contour(&[(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)])];
    let raster = Rasterizer::new(1, 1).unwrap();
    assert_eq!(raster.render(&quarter, 1).unwrap().pixels(), &[0.0]);
    for k in [2, 4, 8] {
        let image = raster.render(&quarter, k).unwrap();
        assert_eq!((image.width(), image.height()), (1, 1));
        assert_eq!(image.pixels(), &[0.25]);
    }
    assert!(raster.render(&quarter, 0).is_err());
    assert!(raster.render(&quarter, usize::MAX).is_err());
}

#[test]
fn bitmaps_are_padded_and_read_back_exactly() {
    let mut image = Image::new(1, 2, Color::rgb(1.0, 0.0, 0.0)).unwrap();
    *image.get_mut(0, 1).unwrap() = Color::rgb(0.0, 0.0, 1.0);
    let mut bytes = Vec::new();
    bmp::write(&image, &mut bytes).unwrap();
    assert_eq!(&bytes[0..2], b"BM");
    assert_eq!(bytes.len(), 62);
    // Bottom blue row, then top red row; each has one padding byte.
    assert_eq!(&bytes[54..], &[255, 0, 0, 0, 0, 0, 255, 0]);
    let back = bmp::read(&bytes).unwrap();
    assert_eq!(back.pixels(), image.pixels());
    assert!(bmp::read(b"not a bitmap").is_err());
    assert!(bmp::read(&bytes[..58]).is_err());
    let bad = Image::new(1, 1, Color::rgb(f64::NAN, 0.0, 0.0)).unwrap();
    assert!(bmp::write(&bad, &mut Vec::new()).is_err());
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
