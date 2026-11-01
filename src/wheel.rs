//! Part 3-III: a wheel of thin wedges whose detail becomes finer than one
//! pixel toward the center, rendered at k = 1 and k = 8.
use crate::experiments::{save_coverage, write_coverage};
use multimedia::{Error, Image, Path as ShapePath, Point2, Rasterizer, Subpath};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

/// Mean coverage and RMS deviation from 0.5 over the central
/// (2*radius) x (2*radius) pixels. A uniform 50% gray scores RMS 0; a
/// pattern of pure black and white scores RMS 0.5.
fn center_statistics(image: &Image<f64>, radius: usize) -> (f64, f64) {
    let cx = image.width() / 2;
    let cy = image.height() / 2;
    let mut sum = 0.0;
    let mut square = 0.0;
    let mut count = 0;
    for y in cy - radius..cy + radius {
        for x in cx - radius..cx + radius {
            let v = *image.get(x, y).unwrap();
            sum += v;
            square += (v - 0.5) * (v - 0.5);
            count += 1;
        }
    }
    (sum / count as f64, (square / count as f64).sqrt())
}
pub fn run(output: &Path, size: usize, wedges: usize) -> Result<(), Error> {
    let center = Point2::new(size as f64 / 2.0, size as f64 / 2.0);
    let radius = 0.43 * size as f64;
    // Small phase offset so no wedge edge lies exactly on a pixel axis.
    let phase = 0.013;
    // Wedge and gap each take half of one period 2*pi/wedges.
    let width = std::f64::consts::PI / wedges as f64;
    let mut subpaths = Vec::new();
    subpaths
        .try_reserve_exact(wedges)
        .map_err(|_| Error::Allocation)?;
    let mut points = BufWriter::new(File::create(output.join("wheel_triangles.csv"))?);
    writeln!(points, "wedge,vertex,x,y")?;
    // Each wedge is a triangle sub-path: center, then two points on the
    // circle `width` radians apart. Wedge i starts one full period after i-1.
    for i in 0..wedges {
        let angle = phase + i as f64 * 2.0 * width;
        let p = Point2::new(
            center.x() + radius * angle.cos(),
            center.y() + radius * angle.sin(),
        );
        let q = Point2::new(
            center.x() + radius * (angle + width).cos(),
            center.y() + radius * (angle + width).sin(),
        );
        let mut sub = Subpath::new(center);
        sub.line_to(p).line_to(q).close();
        subpaths.push(sub);
        for (j, point) in [center, p, q].iter().enumerate() {
            writeln!(points, "{i},{j},{:.12},{:.12}", point.x(), point.y())?;
        }
    }
    points.flush()?;
    let path = ShapePath::new(subpaths).flatten(0)?;
    let raster = Rasterizer::new(size, size)?;
    let mut stats = Vec::new();
    // Same geometry at one sample per pixel and at 8 x 8 supersampling.
    for k in [1, 8] {
        let image = raster.render(&path, k)?;
        save_coverage(&image, &output.join(format!("wheel_k{k}.bmp")))?;
        write_coverage(&output.join(format!("wheel_k{k}_coverage.csv")), &image)?;
        stats.push(center_statistics(&image, 8));
    }
    std::fs::write(output.join("wheel_summary.json"),format!("{{\"canvas\":{size},\"wedges\":{wedges},\"subpaths\":{wedges},\"radius\":{radius:.12},\"phase_radians\":{phase},\"wedge_angle\":{width:.12},\"gap_angle\":{width:.12},\"center_roi_half_width\":8,\"center_mean_k1\":{:.12},\"center_mean_k8\":{:.12},\"center_rms_from_half_k1\":{:.12},\"center_rms_from_half_k8\":{:.12}}}\n",stats[0].0,stats[1].0,stats[0].1,stats[1].1))?;
    std::fs::write(output.join("wheel_answers.md"),format!("# Where the grid loses\n\nThis wheel uses {wedges} triangular subpaths, equal wedge and gap angles, radius {radius:.3}, and phase {phase} radians. k=1 produces false coarse patterns (aliasing / moire), beyond mere edge staircases. The central 16 x 16 region has RMS deviation from half coverage {:.6} at k=1 and {:.6} at k=8; these are measured proxies for how uniform the central gray becomes, not a reconstruction-error ground truth. Inspect both images and the center detail.\n\nTangential pitch is 2*pi*r/{wedges}, so spatial frequency grows proportionally to 1/r toward the center and has no finite bound there. No fixed final pixel grid can satisfy the Sampling Theorem for all center detail. Increasing k estimates area coverage more accurately but does not increase the final output sampling rate; it cannot recover those wedges.\n\nBlock averaging is a box low-pass filter applied before decimation back to the final grid. It suppresses high-frequency content, reducing false patterns and producing gray where alternating wedges cannot be resolved. It does not restore information above the final Nyquist limit, and a finite box filter is not an ideal anti-alias filter.\n",stats[0].1,stats[1].1))?;
    Ok(())
}
