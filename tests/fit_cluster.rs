use multimedia::{
    cluster::dbscan,
    fit::{
        cubic_eval, fit_circle, fit_cubic, fit_line, intersect_line_circle, intersect_lines, Circle,
    },
    Complex,
};
use std::f64::consts::TAU;

fn c(x: f64, y: f64) -> Complex {
    Complex::new(x, y)
}

#[test]
fn dbscan_finds_the_number_of_chains_and_marks_isolated_points_as_noise() {
    let mut points: Vec<Complex> = (0..20).map(|i| c(i as f64, 0.0)).collect();
    points.extend((0..20).map(|i| c(i as f64 * 1.4, 50.0)));
    points.push(c(100.0, 100.0));
    let labels = dbscan(&points, 2.0, 3).unwrap();
    assert!(labels[..20].iter().all(|l| *l == Some(0)));
    assert!(labels[20..40].iter().all(|l| *l == Some(1)));
    assert_eq!(labels[40], None);
    assert!(dbscan(&points, 0.0, 3).is_err());
    assert!(dbscan(&[c(f64::NAN, 0.0)], 1.0, 1).is_err());
}

#[test]
fn line_and_circle_fits_recover_exact_geometry() {
    let along: Vec<Complex> = (0..10)
        .map(|i| c(3.0 + 2.0 * i as f64, 1.0 + i as f64))
        .collect();
    let line = fit_line(&along).unwrap();
    assert!(along.iter().all(|p| line.distance(*p) < 1e-9));
    // Direction is (2, 1) normalized, up to sign.
    let expected = c(2.0, 1.0).scale(1.0 / 5f64.sqrt());
    let dot = line.direction.re * expected.re + line.direction.im * expected.im;
    assert!((dot.abs() - 1.0).abs() < 1e-12);
    // A quarter of a circle is enough to recover it.
    let arc: Vec<Complex> = (0..30)
        .map(|i| c(40.0, -7.0) + Complex::from_angle(i as f64 * TAU / 120.0).scale(12.5))
        .collect();
    let circle = fit_circle(&arc).unwrap();
    assert!((circle.center - c(40.0, -7.0)).abs() < 1e-9);
    assert!((circle.radius - 12.5).abs() < 1e-9);
    assert!(fit_circle(&along).is_none_or(|fit| fit.radius > 1e6));
}

#[test]
fn cubic_fit_reproduces_a_sampled_cubic_and_intersections_rebuild_corners() {
    let q = [c(0.0, 0.0), c(10.0, 20.0), c(30.0, 20.0), c(40.0, 0.0)];
    let samples: Vec<Complex> = (0..=50).map(|i| cubic_eval(&q, i as f64 / 50.0)).collect();
    let fit = fit_cubic(&samples, q[0], q[3]).unwrap();
    // Chord-length parameters differ from the true t, so compare shapes: every
    // sample must lie close to the fitted curve.
    let fitted: Vec<Complex> = (0..=1000)
        .map(|i| cubic_eval(&fit, i as f64 / 1000.0))
        .collect();
    for s in &samples {
        let gap = fitted
            .iter()
            .map(|f| (*f - *s).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(gap < 0.5, "gap {gap}");
    }
    let a = fit_line(&[c(0.0, 0.0), c(1.0, 1.0)]).unwrap();
    let b = fit_line(&[c(0.0, 4.0), c(1.0, 3.0)]).unwrap();
    assert!((intersect_lines(&a, &b).unwrap() - c(2.0, 2.0)).abs() < 1e-12);
    assert!(intersect_lines(&a, &a).is_none());
    let circle = Circle {
        center: c(0.0, 0.0),
        radius: 5.0,
    };
    let horizontal = fit_line(&[c(-1.0, 3.0), c(1.0, 3.0)]).unwrap();
    let hits = intersect_line_circle(&horizontal, &circle);
    assert_eq!(hits.len(), 2);
    assert!(hits
        .iter()
        .all(|p| (p.abs() - 5.0).abs() < 1e-12 && (p.im - 3.0).abs() < 1e-12));
    let far = fit_line(&[c(-1.0, 9.0), c(1.0, 9.0)]).unwrap();
    assert!(intersect_line_circle(&far, &circle).is_empty());
}
