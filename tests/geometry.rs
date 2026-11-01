use multimedia::{distance_to_segment, CubicBezier, Matrix, Point2, Vector};

#[test]
fn formula_and_midpoint_construction_describe_the_same_curve() {
    let curve = CubicBezier::new([
        Point2::new(0.0, 0.0),
        Point2::new(0.0, 2.0),
        Point2::new(2.0, 2.0),
        Point2::new(2.0, 0.0),
    ]);
    assert_eq!(curve.eval(0.5), Point2::new(1.0, 1.5));
    assert_eq!(
        curve.flatten(1).unwrap().vertices,
        vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.5),
            Point2::new(2.0, 0.0)
        ]
    );
    let (left, right) = curve.split_midpoint();
    for t in [0.0, 0.1, 0.4, 0.9, 1.0] {
        assert!(left.eval(t).distance(curve.eval(t * 0.5)) < 1e-12);
        assert!(right.eval(t).distance(curve.eval(0.5 + t * 0.5)) < 1e-12);
    }
}

#[test]
fn geometry_measurements_have_independent_worked_results() {
    let a = Matrix::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let b = Matrix::from_rows([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
    assert_eq!(a * b, Matrix::from_rows([[58.0, 64.0], [139.0, 154.0]]));
    assert_eq!(a.transpose().rows(), &[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    let vector = Vector::from_array([3.0, 4.0]);
    assert_eq!(vector.norm(), 5.0);
    assert_eq!(vector.dot(Vector::from_array([2.0, -1.0])), 2.0);
    assert_eq!(Matrix::<2, 2>::identity() * vector, vector);
    assert_eq!(
        distance_to_segment(
            Point2::new(1.0, 3.0),
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0)
        ),
        3.0
    );
    assert_eq!(
        distance_to_segment(
            Point2::new(5.0, 4.0),
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0)
        ),
        5.0
    );
    assert_eq!(
        distance_to_segment(
            Point2::new(3.0, 4.0),
            Point2::new(0.0, 0.0),
            Point2::new(0.0, 0.0)
        ),
        5.0
    );
    let curve = CubicBezier::new([
        Point2::new(0.0, 0.0),
        Point2::new(0.0, 2.0),
        Point2::new(2.0, 2.0),
        Point2::new(2.0, 0.0),
    ]);
    let samples = curve.sample_dense(0.1).unwrap();
    assert!(samples.windows(2).all(|p| p[0].distance(p[1]) < 0.1));
    let chord = curve.flatten(0).unwrap();
    assert_eq!(chord.distance_to(Point2::new(1.0, 1.5)).unwrap(), 1.5);
    assert!(curve.flatten(usize::MAX).is_err());
    assert!(curve.sample_dense(0.0).is_err());
    let invalid = CubicBezier::new([Point2::new(f64::NAN, 0.0); 4]);
    assert!(invalid.flatten(1).is_err());
}
