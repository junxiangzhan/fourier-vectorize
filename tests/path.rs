use multimedia::{Path, Point2, Subpath};

#[test]
fn open_curve_paths_preserve_joins_and_subpath_state() {
    let mut sub = Subpath::new(Point2::new(0.0, 0.0));
    sub.cubic_to(
        Point2::new(0.0, 2.0),
        Point2::new(2.0, 2.0),
        Point2::new(2.0, 0.0),
    );
    sub.line_to(Point2::new(3.0, 0.0));
    let path = Path::new(vec![sub]);
    let flat = path.flatten(1).unwrap();
    assert_eq!(
        flat.subpaths[0].vertices,
        vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.5),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.0)
        ]
    );
    assert!(!flat.subpaths[0].closed);
}

#[test]
fn reversing_curves_changes_traversal_without_moving_geometry() {
    let mut sub = Subpath::new(Point2::new(0.0, 0.0));
    sub.cubic_to(
        Point2::new(0.0, 2.0),
        Point2::new(2.0, 2.0),
        Point2::new(2.0, 0.0),
    );
    let path = Path::new(vec![sub]);
    let a = path.flatten(3).unwrap();
    let b = path.reversed().flatten(3).unwrap();
    assert_eq!(
        a.subpaths[0]
            .vertices
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>(),
        b.subpaths[0].vertices
    );
}

#[test]
fn transforms_preserve_arbitrary_center_and_closed_state() {
    let center = Point2::new(2.0, 2.0);
    let rotated = Point2::new(4.0, 2.0).rotated_about(center, std::f64::consts::FRAC_PI_2);
    assert!(rotated.distance(Point2::new(2.0, 4.0)) < 1e-12);
    let mut sub = Subpath::new(Point2::new(4.0, 2.0));
    sub.line_to(Point2::new(4.0, 3.0))
        .line_to(Point2::new(3.0, 3.0))
        .close();
    let path = Path::new(vec![sub]);
    let scaled = path.scaled_about(center, 2.0).flatten(0).unwrap();
    assert_eq!(scaled.subpaths[0].vertices[0], Point2::new(6.0, 2.0));
    assert!(path.rotated_about(center, 0.1).flatten(0).unwrap().subpaths[0].closed);
    assert!(scaled.subpaths[0].closed);
}
