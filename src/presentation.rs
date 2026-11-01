//! Presentation of already-computed points and measurements (Part 1 exception).
use multimedia::{Error, Point2};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

/// Writes points as CSV (index, x, y) so every figure has its raw data.
pub fn write_points(path: &Path, points: &[Point2]) -> Result<(), Error> {
    let mut file = BufWriter::new(File::create(path)?);
    writeln!(file, "index,x,y")?;
    for (i, p) in points.iter().enumerate() {
        writeln!(file, "{i},{:.12},{:.12}", p.x(), p.y())?;
    }
    file.flush()?;
    Ok(())
}

/// SVG of computed curve points and control polygons. SVG only connects the
/// points we pass in; no SVG curve primitive is used (Part 1 exception).
pub fn geometry_svg(
    path: &Path,
    size: usize,
    title: &str,
    curves: &[(&[Point2], &str)],
    controls: &[&[Point2]],
) -> Result<(), Error> {
    let mut file = BufWriter::new(File::create(path)?);
    writeln!(file, "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {size} {}\"><rect width=\"100%\" height=\"100%\" fill=\"white\"/><text x=\"10\" y=\"20\" font-family=\"sans-serif\" font-size=\"12\">{title}</text><g transform=\"translate(0,28)\">",size+28)?;
    for points in controls {
        write!(
            file,
            "<polyline fill=\"none\" stroke=\"#a0a7b2\" stroke-width=\"1\" points=\""
        )?;
        for p in *points {
            write!(file, "{:.4},{:.4} ", p.x(), p.y())?;
        }
        writeln!(file, "\"/>")?;
        for p in *points {
            writeln!(
                file,
                "<circle cx=\"{:.4}\" cy=\"{:.4}\" r=\"2.5\" fill=\"#687080\"/>",
                p.x(),
                p.y()
            )?;
        }
    }
    for (points, color) in curves {
        write!(
            file,
            "<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"1.8\" points=\""
        )?;
        for p in *points {
            write!(file, "{:.4},{:.4} ", p.x(), p.y())?;
        }
        writeln!(file, "\"/>")?;
    }
    writeln!(file, "</g></svg>")?;
    file.flush()?;
    Ok(())
}

type ChartSeries<'a> = (&'a str, &'a str, Vec<(f64, f64)>);

/// Line chart of measurements (allowed: plotting what we measured).
pub fn chart_svg(
    path: &Path,
    title: &str,
    x_label: &str,
    y_label: &str,
    series: &[ChartSeries<'_>],
) -> Result<(), Error> {
    let all: Vec<_> = series
        .iter()
        .flat_map(|(_, _, points)| points.iter().copied())
        .collect();
    let xmin = all.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let xmax = all
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max)
        .max(xmin + 1.0);
    let ymin = 0.0_f64.min(all.iter().map(|p| p.1).fold(f64::INFINITY, f64::min));
    let ymax = all
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max)
        .max(ymin + 0.01);
    let px = |v: f64| 65.0 + (v - xmin) / (xmax - xmin) * 560.0;
    let py = |v: f64| 320.0 - (v - ymin) / (ymax - ymin) * 255.0;
    let mut file = BufWriter::new(File::create(path)?);
    writeln!(file,"<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 680 390\"><rect width=\"100%\" height=\"100%\" fill=\"white\"/><g font-family=\"sans-serif\" font-size=\"12\"><text x=\"65\" y=\"24\">{title}</text><path d=\"M65 65V320H625\" fill=\"none\" stroke=\"#222\"/>")?;
    for i in 0..=5 {
        let x = xmin + (xmax - xmin) * i as f64 / 5.0;
        let y = ymin + (ymax - ymin) * i as f64 / 5.0;
        writeln!(file,"<text x=\"{:.2}\" y=\"340\" text-anchor=\"middle\">{:.2}</text><text x=\"57\" y=\"{:.2}\" text-anchor=\"end\">{:.3}</text>",px(x),x,py(y)+4.0,y)?;
    }
    for (i, (name, color, points)) in series.iter().enumerate() {
        write!(
            file,
            "<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"1.6\" points=\""
        )?;
        for &(x, y) in points {
            write!(file, "{:.3},{:.3} ", px(x), py(y))?;
        }
        writeln!(
            file,
            "\"/><text x=\"{}\" y=\"47\" fill=\"{color}\">{name}</text>",
            65 + i * 200
        )?;
    }
    writeln!(file,"<text x=\"350\" y=\"380\" text-anchor=\"middle\">{x_label}</text><text transform=\"translate(15,200) rotate(-90)\" text-anchor=\"middle\">{y_label}</text></g></svg>")?;
    file.flush()?;
    Ok(())
}
