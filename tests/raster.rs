use multimedia::{winding_at, Path, Point2, Rasterizer, Subpath};

fn polygon(points: &[(f64, f64)]) -> Subpath {
    let mut sub = Subpath::new(Point2::new(points[0].0, points[0].1));
    for &(x, y) in &points[1..] {
        sub.line_to(Point2::new(x, y));
    }
    sub.close();
    sub
}
#[test]
fn nonzero_winding_fills_a_self_intersecting_star_center() {
    let vertices = [
        (3.0, 0.0),
        (4.76336, 5.42705),
        (0.14683, 2.07295),
        (5.85317, 2.07295),
        (1.23664, 5.42705),
    ];
    let path = Path::new(vec![polygon(&vertices)]).flatten(0).unwrap();
    let image = Rasterizer::new(6, 6).unwrap().render(&path, 1).unwrap();
    assert_eq!(winding_at(&path, Point2::new(3.0, 3.0)).unwrap(), 2);
    assert_eq!(*image.get(3, 3).unwrap(), 1.0);
}

#[test]
fn pixel_centers_shared_vertices_and_off_canvas_edges_have_consistent_fill() {
    let triangle = Path::new(vec![polygon(&[(-1.0, -1.0), (4.0, -1.0), (4.0, 4.0)])])
        .flatten(0)
        .unwrap();
    let image = Rasterizer::new(3, 3).unwrap().render(&triangle, 1).unwrap();
    assert_eq!(
        image.pixels(),
        &[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    );
    let surround = Path::new(vec![polygon(&[
        (-10.0, -10.0),
        (10.0, -10.0),
        (10.0, 10.0),
        (-10.0, 10.0),
    ])])
    .flatten(0)
    .unwrap();
    assert_eq!(
        Rasterizer::new(2, 2)
            .unwrap()
            .render(&surround, 1)
            .unwrap()
            .pixels(),
        &[1.0; 4]
    );
    let pixel_edge = Path::new(vec![polygon(&[
        (0.5, 0.5),
        (2.5, 0.5),
        (2.5, 2.5),
        (0.5, 2.5),
    ])])
    .flatten(0)
    .unwrap();
    assert_eq!(
        Rasterizer::new(3, 3)
            .unwrap()
            .render(&pixel_edge, 1)
            .unwrap()
            .pixels(),
        &[1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(winding_at(&pixel_edge, Point2::new(0.5, 0.5)).unwrap(), 1);
    assert_eq!(winding_at(&pixel_edge, Point2::new(2.5, 1.5)).unwrap(), 0);
    let mut open = Subpath::new(Point2::new(0.0, 0.0));
    open.line_to(Point2::new(1.0, 1.0));
    assert!(Rasterizer::new(2, 2)
        .unwrap()
        .render(&Path::new(vec![open]).flatten(0).unwrap(), 1)
        .is_err());
    assert!(Rasterizer::new(0, 2).is_err());
    assert!(Rasterizer::new(usize::MAX, 2).is_err());
    let mut non_finite = triangle.clone();
    non_finite.subpaths[0].vertices[0] = Point2::new(f64::NAN, 0.0);
    assert!(Rasterizer::new(2, 2)
        .unwrap()
        .render(&non_finite, 1)
        .is_err());
    assert!(winding_at(&triangle, Point2::new(0.0, f64::INFINITY)).is_err());
    let repeated = Path::new(vec![polygon(&[
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, 0.0),
        (2.0, 2.0),
        (0.0, 2.0),
    ])])
    .flatten(0)
    .unwrap();
    assert_eq!(
        Rasterizer::new(2, 2)
            .unwrap()
            .render(&repeated, 1)
            .unwrap()
            .pixels(),
        &[1.0; 4]
    );
}

#[test]
fn reversing_only_the_inner_contour_changes_the_hole() {
    let outer = polygon(&[(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0)]);
    let inner = polygon(&[(1.0, 1.0), (5.0, 1.0), (5.0, 5.0), (1.0, 5.0)]);
    let reverse_outer = Path::new(vec![outer.reversed()]).flatten(0).unwrap();
    assert_eq!(
        winding_at(&reverse_outer, Point2::new(3.0, 3.0)).unwrap(),
        -1
    );
    let same = Path::new(vec![outer.clone(), inner.clone()])
        .flatten(0)
        .unwrap();
    let opposite = Path::new(vec![outer, inner.reversed()]).flatten(0).unwrap();
    assert_eq!(winding_at(&same, Point2::new(3.0, 3.0)).unwrap(), 2);
    assert_eq!(winding_at(&opposite, Point2::new(3.0, 3.0)).unwrap(), 0);
    let raster = Rasterizer::new(6, 6).unwrap();
    assert_eq!(*raster.render(&same, 1).unwrap().get(2, 2).unwrap(), 1.0);
    assert_eq!(
        *raster.render(&opposite, 1).unwrap().get(2, 2).unwrap(),
        0.0
    );
}

#[test]
fn supersampling_returns_area_fraction_at_original_dimensions() {
    let quarter = Path::new(vec![polygon(&[
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.0, 0.5),
    ])])
    .flatten(0)
    .unwrap();
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
