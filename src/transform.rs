//! Part 4 (Transformations, Chapter 4): rotate the star by 1° a hundred
//! times, once as vector geometry (rasterize at the end) and once as a bitmap
//! (resample after every step), and measure how far each route ends up from
//! the star rotated by 100° in a single step.
use crate::{
    experiments::{save_coverage, star_path, write_coverage},
    presentation::{chart_svg, write_points},
};
use multimedia::{Error, Image, Point2, Rasterizer};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

const STEPS: usize = 100;
/// Every image uses the same 8 x 8 supersampled fill, so the two routes
/// differ only in the order of rotation and rasterization.
const K: usize = 8;

/// RMS and maximum absolute per-pixel difference of two same-size images.
fn difference(a: &Image<f64>, b: &Image<f64>) -> (f64, f64) {
    let mut square = 0.0;
    let mut max = 0.0_f64;
    for (x, y) in a.pixels().iter().zip(b.pixels()) {
        let d = x - y;
        square += d * d;
        max = max.max(d.abs());
    }
    ((square / a.pixels().len() as f64).sqrt(), max)
}

/// Pixels that are visibly gray in the 8-bit output (neither 0 nor 255):
/// how wide the edge ramp looks. Tiny tails below half a level are ignored.
fn partial(image: &Image<f64>) -> usize {
    let half_level = 0.5 / 255.0;
    image
        .pixels()
        .iter()
        .filter(|v| **v >= half_level && **v < 1.0 - half_level)
        .count()
}

/// Total coverage, i.e. the shape's area in pixels.
fn area(image: &Image<f64>) -> f64 {
    image.pixels().iter().sum()
}

fn save(output: &Path, name: &str, image: &Image<f64>) -> Result<(), Error> {
    save_coverage(image, &output.join(format!("transform_{name}.bmp")))?;
    write_coverage(
        &output.join(format!("transform_{name}_coverage.csv")),
        image,
    )
}

pub fn run(output: &Path, size: usize) -> Result<(), Error> {
    let center = Point2::new(size as f64 / 2.0, size as f64 / 2.0);
    let step = 1.0_f64.to_radians();
    let raster = Rasterizer::new(size, size)?;
    // Ground truth for n steps: the original vertices rotated by n degrees
    // in one go, then rasterized.
    let direct = |n: usize| star_path(size, (n as f64).to_radians()).flatten(0);

    // Route 1, vector: rotate the vertices 1° at a time, rasterize once.
    let mut shape = star_path(size, 0.0);
    for _ in 0..STEPS {
        shape = shape.rotated_about(center, step);
    }
    let vector_flat = shape.flatten(0)?;
    let direct_flat = direct(STEPS)?;
    let vector = raster.render(&vector_flat, K)?;
    let reference = raster.render(&direct_flat, K)?;
    // Accumulated floating-point error in the vertex positions.
    let drift = vector_flat.subpaths[0]
        .vertices
        .iter()
        .zip(&direct_flat.subpaths[0].vertices)
        .map(|(a, b)| a.distance(*b))
        .fold(0.0_f64, f64::max);
    write_points(
        &output.join("transform_vector_vertices.csv"),
        &vector_flat.subpaths[0].vertices,
    )?;
    write_points(
        &output.join("transform_direct_vertices.csv"),
        &direct_flat.subpaths[0].vertices,
    )?;

    // Route 2, bitmap: rasterize once, then rotate and resample the bitmap
    // 1° at a time. After every step, compare with the direct rendering at
    // the same angle to see how the error grows.
    let base = raster.render(&direct(0)?, K)?;
    let mut bitmap = base.clone();
    let mut curve = BufWriter::new(File::create(output.join("transform_curve.csv"))?);
    writeln!(
        curve,
        "steps,rms,max_abs,partial_pixels,reference_partial_pixels,area_ratio"
    )?;
    let mut series = Vec::new();
    for n in 1..=STEPS {
        bitmap = bitmap.rotated_bilinear(center, step)?;
        if n == 1 {
            // Kept so the audit can recompute one resampling step itself.
            write_coverage(&output.join("transform_step1_coverage.csv"), &bitmap)?;
        }
        let target = raster.render(&direct(n)?, K)?;
        let (rms, max) = difference(&bitmap, &target);
        writeln!(
            curve,
            "{n},{rms:.12},{max:.12},{},{},{:.12}",
            partial(&bitmap),
            partial(&target),
            area(&bitmap) / area(&target)
        )?;
        series.push((n as f64, rms));
    }
    curve.flush()?;

    // Control: one single 100° resampling, to separate "resampled once" from
    // "resampled a hundred times".
    let once = base.rotated_bilinear(center, (STEPS as f64).to_radians())?;

    for (name, image) in [
        ("base", &base),
        ("reference", &reference),
        ("vector", &vector),
        ("bitmap", &bitmap),
        ("once", &once),
    ] {
        save(output, name, image)?;
    }
    // Where the bitmap route went wrong: |bitmap - reference|, dark = large.
    let mut error = Image::new(size, size, 0.0)?;
    for y in 0..size {
        for x in 0..size {
            *error.get_mut(x, y).unwrap() =
                (bitmap.get(x, y).unwrap() - reference.get(x, y).unwrap()).abs();
        }
    }
    save_coverage(&error, &output.join("transform_error.bmp"))?;

    chart_svg(
        &output.join("transform_curve.svg"),
        "Bitmap route: RMS error after n resampled 1-degree steps",
        "Steps n",
        "RMS coverage error",
        &[("bitmap", "#e66b18", series)],
    )?;
    let (vector_rms, vector_max) = difference(&vector, &reference);
    let (bitmap_rms, bitmap_max) = difference(&bitmap, &reference);
    let (once_rms, once_max) = difference(&once, &reference);
    std::fs::write(
        output.join("transform_summary.json"),
        format!(
            "{{\"canvas\":{size},\"k\":{K},\"steps\":{STEPS},\"step_degrees\":1,\"center\":[{},{}],\"resampling\":\"bilinear, inverse mapping, outside = 0\",\"vertex_max_drift_px\":{drift:.6e},\"vector_rms\":{vector_rms:.12},\"vector_max_abs\":{vector_max:.12},\"bitmap_rms\":{bitmap_rms:.12},\"bitmap_max_abs\":{bitmap_max:.12},\"once_rms\":{once_rms:.12},\"once_max_abs\":{once_max:.12},\"partial_reference\":{},\"partial_vector\":{},\"partial_bitmap\":{},\"partial_once\":{},\"area_reference\":{:.12},\"area_bitmap\":{:.12}}}\n",
            center.x(),
            center.y(),
            partial(&reference),
            partial(&vector),
            partial(&bitmap),
            partial(&once),
            area(&reference),
            area(&bitmap)
        ),
    )?;
    Ok(())
}
