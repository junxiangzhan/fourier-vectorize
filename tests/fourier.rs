use multimedia::{fft, hilbert, Complex};
use std::f64::consts::TAU;

#[test]
fn fft_matches_the_direct_sum_and_inverts_itself() {
    let input: Vec<Complex> = (0..8)
        .map(|n| Complex::new((n * n) as f64 * 0.5 - 3.0, n as f64 % 3.0))
        .collect();
    let mut fast = input.clone();
    fft(&mut fast, false).unwrap();
    // X[f] = sum_n x[n] e^(-2 pi i f n / N), summed directly.
    for (f, value) in fast.iter().enumerate() {
        let direct = input.iter().enumerate().fold(Complex::ZERO, |sum, (n, x)| {
            sum + *x * Complex::from_angle(-TAU * (f * n) as f64 / 8.0)
        });
        assert!((*value - direct).abs() < 1e-12, "bin {f}");
    }
    fft(&mut fast, true).unwrap();
    for (a, b) in fast.iter().zip(&input) {
        assert!((*a - *b).abs() < 1e-12);
    }
    // An impulse has a flat spectrum.
    let mut impulse = vec![Complex::ZERO; 16];
    impulse[0] = Complex::new(1.0, 0.0);
    fft(&mut impulse, false).unwrap();
    assert!(impulse
        .iter()
        .all(|c| (*c - Complex::new(1.0, 0.0)).abs() < 1e-12));
    assert!(fft(&mut [Complex::ZERO; 6], false).is_err());
    assert!(fft(&mut [Complex::new(f64::NAN, 0.0)], false).is_err());
}

#[test]
fn hilbert_curve_visits_every_cell_once_in_unit_steps() {
    let n = 16;
    let mut seen = vec![false; n * n];
    let mut previous: Option<(usize, usize)> = None;
    for d in 0..n * n {
        let (x, y) = hilbert::d2xy(n, d).unwrap();
        assert!(x < n && y < n && !seen[y * n + x]);
        seen[y * n + x] = true;
        if let Some((px, py)) = previous {
            assert_eq!(px.abs_diff(x) + py.abs_diff(y), 1, "step {d}");
        }
        previous = Some((x, y));
    }
    assert_eq!(hilbert::d2xy(n, 0).unwrap(), (0, 0));
    assert!(hilbert::d2xy(6, 0).is_err());
    assert!(hilbert::d2xy(n, n * n).is_err());
}
