//! The program as a user runs it.
use fourier_vectorize::{Color, Image};
use std::{path::Path, process::Command};

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

#[test]
fn bad_arguments_are_refused() {
    for args in [
        &["frobnicate"][..],
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
