//! The program as a user runs it.
use fourier_vectorize::{Color, Image};
use std::{
    path::{Path, PathBuf},
    process::Command,
};

fn run(args: &[&str]) -> (bool, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_fourier-vectorize")).args(args).output().unwrap();
    (output.status.success(), String::from_utf8_lossy(&output.stderr).into_owned())
}

#[test]
fn fit_then_render_writes_the_named_files() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("cli-test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut picture = Image::new(40, 32, Color::rgb(0.9, 0.85, 0.7)).unwrap();
    for y in 6..26 {
        for x in 8..30 {
            *picture.get_mut(x, y).unwrap() = Color::rgb(0.1, 0.2, 0.7);
        }
    }
    let image = dir.join("square.bmp");
    picture.save_bmp(&image).unwrap();
    let (image, out) = (image.to_str().unwrap(), dir.to_str().unwrap());
    let (ok, error) = run(&["fit", image, "--out", out, "--factor", "1", "--levels", "4,3,3", "--method", "dft", "--keep", "0.5", "--rounds", "2", "--prune-steps", "10"]);
    assert!(ok, "{error}");
    let csv = dir.join("square_dft_keep0.5_largest_r2_s10_coefficients.csv");
    assert!(csv.exists());
    assert!(dir.join("square_dft_keep0.5_largest_r2_s10.json").exists());
    let (ok, error) = run(&["render", csv.to_str().unwrap(), "--out", out, "--tolerance", "0.5", "--decimals", "0"]);
    assert!(ok, "{error}");
    assert!(dir.join("square_dft_keep0.5_largest_r2_s10_t0.5_d0.svg").exists());
    assert!(dir.join("square_dft_keep0.5_largest_r2_s10.bmp").exists());
}

/// A fresh directory under target/ for one test.
fn fresh_dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A blue square on a cream ground, with a slight ramp so that the pixels are
/// not all round numbers.
fn square() -> image::RgbImage {
    image::RgbImage::from_fn(40, 32, |x, y| {
        let ramp = (x + y) as u8;
        if (8..30).contains(&x) && (6..26).contains(&y) {
            image::Rgb([25 + ramp, 50, 180 - ramp])
        } else {
            image::Rgb([230, 215 - ramp, 180])
        }
    })
}

/// Fits `file` with cheap settings into its own directory, then returns the
/// coefficients and the summary without its timing.
fn fit_cheaply(file: &Path, out: &Path) -> (String, String) {
    std::fs::create_dir_all(out).unwrap();
    let (ok, error) = run(&["fit", file.to_str().unwrap(), "--out", out.to_str().unwrap(), "--factor", "1", "--levels", "8,4,4", "--method", "dft"]);
    assert!(ok, "{file:?}: {error}");
    let coefficients = std::fs::read_to_string(out.join("square_dft_coefficients.csv")).unwrap();
    let summary = std::fs::read_to_string(out.join("square_dft.json")).unwrap();
    let summary = summary.split("\"seconds\"").next().unwrap().to_owned();
    (coefficients, summary)
}

#[test]
fn png_and_bmp_of_the_same_pixels_give_identical_output() {
    let dir = fresh_dir("cli-test-png");
    let pixels = square();
    let mut picture = Image::new(40, 32, Color::rgb(0.0, 0.0, 0.0)).unwrap();
    for (x, y, p) in pixels.enumerate_pixels() {
        let v = |c: u8| c as f64 / 255.0;
        *picture.get_mut(x as usize, y as usize).unwrap() = Color::rgb(v(p[0]), v(p[1]), v(p[2]));
    }
    let bmp = dir.join("square.bmp");
    picture.save_bmp(&bmp).unwrap();
    let png = dir.join("square.png");
    pixels.save(&png).unwrap();
    let expected = fit_cheaply(&bmp, &dir.join("from-bmp"));
    assert!(expected.0.lines().count() > 1, "the fit found no outlines");
    assert_eq!(fit_cheaply(&png, &dir.join("from-png")), expected);
}

#[test]
fn jpeg_fits_and_renders() {
    let dir = fresh_dir("cli-test-jpeg");
    let jpeg = dir.join("square.jpg");
    square().save(&jpeg).unwrap();
    let (coefficients, _) = fit_cheaply(&jpeg, &dir);
    assert!(coefficients.lines().count() > 1, "the fit found no outlines");
    let csv = dir.join("square_dft_coefficients.csv");
    let (ok, error) = run(&["render", csv.to_str().unwrap(), "--out", dir.to_str().unwrap()]);
    assert!(ok, "{error}");
    assert!(dir.join("square_dft.svg").exists());
    let redrawn = fourier_vectorize::bmp::read(&std::fs::read(dir.join("square_dft.bmp")).unwrap()).unwrap();
    assert_eq!((redrawn.width(), redrawn.height()), (40, 32));
}

#[test]
fn the_format_comes_from_the_contents_not_the_extension() {
    let dir = fresh_dir("cli-test-extension");
    let mut bytes = std::io::Cursor::new(Vec::new());
    square().write_to(&mut bytes, image::ImageFormat::Png).unwrap();
    let misnamed = dir.join("square.bmp");
    std::fs::write(&misnamed, bytes.into_inner()).unwrap();
    fit_cheaply(&misnamed, &dir);
}

#[test]
fn unsupported_or_corrupt_files_are_refused() {
    let dir = fresh_dir("cli-test-corrupt");
    let encoded = |format| {
        let mut bytes = std::io::Cursor::new(Vec::new());
        square().write_to(&mut bytes, format).unwrap();
        bytes.into_inner()
    };
    let (png, jpeg) = (encoded(image::ImageFormat::Png), encoded(image::ImageFormat::Jpeg));
    let mut flipped_png = png.clone();
    let middle = flipped_png.len() / 2;
    flipped_png[middle] ^= 0xFF;
    for (name, bytes) in [
        ("empty.png", &[][..]),
        ("text.png", b"not a picture"),
        ("animation.gif", b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"),
        ("cut.png", &png[..png.len() / 2]),
        ("flipped.png", &flipped_png),
        ("cut.jpg", &jpeg[..jpeg.len() / 2]),
        ("header.jpg", &jpeg[..jpeg.len().min(20)]),
        ("cut.bmp", b"BM\x00\x00"),
    ] {
        let file = dir.join(name);
        std::fs::write(&file, bytes).unwrap();
        let (ok, error) = run(&["fit", file.to_str().unwrap(), "--out", dir.to_str().unwrap()]);
        assert!(!ok && error.starts_with("Error:"), "{name}: {error}");
    }
}

#[test]
fn pack_then_render_needs_only_the_packed_file() {
    let dir = fresh_dir("cli-test-pack");
    let png = dir.join("square.png");
    square().save(&png).unwrap();
    fit_cheaply(&png, &dir);
    let (csv, out) = (dir.join("square_dft_coefficients.csv"), dir.to_str().unwrap());
    let (ok, error) = run(&["pack", csv.to_str().unwrap(), "--out", out, "--movement", "0.1"]);
    assert!(ok, "{error}");
    let packed = dir.join("square_dft_m0.1.fvc");
    assert!(packed.exists());
    std::fs::remove_file(&csv).unwrap();
    std::fs::remove_file(dir.join("square_dft.json")).unwrap();
    let (ok, error) = run(&["render", packed.to_str().unwrap(), "--out", out]);
    assert!(ok, "{error}");
    assert!(dir.join("square_dft_m0.1.svg").exists());
    assert!(dir.join("square_dft_m0.1.bmp").exists());
    // A damaged one is refused.
    let bytes = std::fs::read(&packed).unwrap();
    let cut = dir.join("cut.fvc");
    std::fs::write(&cut, &bytes[..bytes.len() / 2]).unwrap();
    let (ok, error) = run(&["render", cut.to_str().unwrap(), "--out", out]);
    assert!(!ok && error.starts_with("Error:"), "{error}");
}

#[test]
fn bad_arguments_are_refused() {
    for args in [
        &["frobnicate"][..],
        &["pack"],
        &["pack", "a_coefficients.csv", "--movement", "0"],
        &["pack", "a_coefficients.csv", "--movement", "-1"],
        &["pack", "a_coefficients.csv", "--movement", "inf"],
        &["pack", "a_coefficients.csv", "--scale", "2"],
        &["pack", "does-not-exist_coefficients.csv"],
        &["render", "a_coefficients.csv", "--movement", "0.2"],
        &["fit", "a.bmp", "--movement", "0.2"],
        &["fit"],
        &["fit", "a.bmp", "--keep", "1.5"],
        &["fit", "a.bmp", "--keep", "0"],
        &["fit", "a.bmp", "--prune", "lowpass"],
        &["fit", "a.bmp", "--keep", "0.5", "--rounds", "0"],
        &["fit", "a.bmp", "--tolerance", "1"],
        &["fit", "a.bmp", "--levels", "1,2,3"],
        &["fit", "a.bmp", "--method", "magic"],
        &["render", "a_coefficients.csv", "--keep", "0.5"],
        &["render", "a_coefficients.csv", "--bmp", "--decimals", "2"],
        &["render", "a_coefficients.csv", "--decimals", "9"],
        &["render", "a_coefficients.csv", "--tolerance", "-1"],
        &["render", "a_coefficients.csv", "--scale", "99"],
        &["fit", "does-not-exist.bmp"],
    ] {
        let (ok, error) = run(args);
        assert!(!ok && error.starts_with("Error:"), "{args:?}: {error}");
    }
    assert!(run(&["--help"]).0);
}
