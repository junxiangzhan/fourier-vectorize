//! The two jobs: fitting a picture and writing its coefficients, and drawing
//! coefficients.
use crate::{
    color::{delta_e, lab, over_white, ssim},
    decode::decode,
    drawing::{Drawing, SvgDetail, ALPHA, PLATES},
    fit::fit_all,
    outline::{alpha_outlines, smooth_outlines},
    packed,
    preprocess::{bilateral, shrink, shrink_factor, shrink_transparent},
    prune::{prune_fits, Prune},
    Color, Complex, Error, FitKind, Image,
};
use std::path::{Path, PathBuf};

/// Without a given factor, a picture is shrunk by the smallest factor that
/// brings its longer side to at most this many pixels.
const LONGEST: usize = 2000;
/// A pixel differs visibly when ΔE exceeds this.
const VISIBLE_DELTA_E: f64 = 10.0;

/// How far `fit` and `pack` let rounding move each curve unless told otherwise,
/// as an RMS in pixels.
pub const DEFAULT_MOVEMENT: f64 = 0.2;

/// How to fit a picture.
#[derive(Clone, Copy)]
pub struct FitOptions {
    /// Shrink by box-averaging this many pixels per side; `None` picks the
    /// smallest factor that brings the longer side to at most 2000 px.
    pub factor: Option<usize>,
    /// Levels for L*, a*, b* and, if the picture has any transparency, alpha.
    pub levels: [usize; 4],
    pub kind: FitKind,
    pub prune: Option<Prune>,
    /// Write a packed file that lets rounding move each curve this far, as an
    /// RMS in pixels; `None` writes a CSV instead, which rounds far less.
    pub movement: Option<f64>,
}
impl Default for FitOptions {
    fn default() -> Self {
        Self { factor: None, levels: [32, 16, 16, 16], kind: FitKind::Refined, prune: None, movement: Some(DEFAULT_MOVEMENT) }
    }
}

/// Rounds to the 8 bits a BMP stores, so every measure can be recomputed
/// from the saved files.
fn eight_bit(image: &Image<Color>) -> Result<Image<Color>, Error> {
    let r = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() / 255.0;
    image.map(|c| Color::rgb(r(c.r), r(c.g), r(c.b)))
}

fn measure(image: &Image<Color>, reference: &Image<Color>) -> Result<String, Error> {
    let a = eight_bit(image)?;
    let d: Vec<f64> = a.pixels().iter().zip(reference.pixels()).map(|(x, y)| delta_e(*x, *y)).collect();
    Ok(format!(
        "{{\"mean_delta_e\":{:.6},\"visible\":{},\"ssim\":{:.6}}}",
        d.iter().sum::<f64>() / d.len() as f64,
        d.iter().filter(|v| **v > VISIBLE_DELTA_E).count(),
        ssim(&a, reference)
    ))
}

fn json_list(values: &[f64], digits: usize) -> String {
    format!("[{}]", values.iter().map(|v| format!("{v:.digits$}")).collect::<Vec<_>>().join(","))
}

/// The files one fit wrote.
pub struct Fitted {
    /// What `render` draws.
    pub coefficients: PathBuf,
    pub summary: PathBuf,
}

/// Fits one BMP, PNG or JPEG and writes its coefficients, as <name>_<fit>_m<movement>.fvc
/// or, without a movement, <name>_<fit>_coefficients.csv (every c_k of every outline), and
/// <name>_<fit>.json (size, levels, counts, and mean ΔE and lightness SSIM of the
/// rendering against the input). `render` draws the coefficients as SVG or BMP at any
/// scale. A pruned fit has the pruning settings in its name too, so it never
/// replaces another.
pub fn fit_image(output: &Path, path: &Path, options: &FitOptions) -> Result<Fitted, Error> {
    let started = std::time::Instant::now();
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or(Error::InvalidParameter("image path has no usable file name"))?;
    let (source, alpha) = decode(&std::fs::read(path)?)?;
    let factor = shrink_factor(source.width(), source.height(), LONGEST, options.factor);
    let (input, alpha) = match alpha {
        Some(alpha) => shrink_transparent(&source, &alpha, factor)?,
        None => (shrink(&source, factor)?, None),
    };
    let (w, h) = (input.width(), input.height());
    let smooth = bilateral(&input)?;
    let labs: Vec<[f64; 3]> = smooth.pixels().iter().map(|c| lab(*c)).collect();
    let fit_name = options.kind.name();
    let stem = match options.prune {
        Some(p) => format!("{name}_{fit_name}_keep{}_{}_r{}_s{}", p.keep, p.by.name(), p.rounds, p.steps),
        None => format!("{name}_{fit_name}"),
    };
    // Every outline of every mask goes through one queue; `masks` remembers
    // plate, level, value and the range of each mask's outlines.
    let mut outlines: Vec<Vec<Complex>> = Vec::new();
    let mut masks = Vec::new();
    let (mut dropped, mut base) = (0, [0.0; 4]);
    let mut levels_json = Vec::new();
    let plates = if alpha.is_some() { PLATES.len() } else { ALPHA };
    for (channel, plate) in PLATES.iter().enumerate().take(plates) {
        let (found_levels, thresholds, found, lost) = match &alpha {
            Some(alpha) if channel == ALPHA => alpha_outlines(alpha.pixels(), w, h, options.levels[channel]),
            _ => {
                let values: Vec<f64> = labs.iter().map(|v| v[channel]).collect();
                smooth_outlines(&values, w, h, options.levels[channel])
            }
        };
        dropped += lost;
        levels_json.push(format!(
            "\"{plate}\":{{\"levels\":{},\"thresholds\":{}}}",
            json_list(&found_levels, 9),
            json_list(&thresholds, 9)
        ));
        for ((j, &value), loops) in found_levels.iter().enumerate().skip(1).zip(found) {
            if !loops.is_empty() {
                masks.push((channel, j, value, outlines.len()..outlines.len() + loops.len()));
                outlines.extend(loops);
            }
        }
        base[channel] = found_levels[0];
    }
    let mut fits = fit_all(&outlines, options.kind)?;
    if let Some(p) = options.prune {
        prune_fits(&outlines, &mut fits, p)?;
    }
    let drawing = Drawing {
        w,
        h,
        base: [base[0], base[1], base[2]],
        alpha: alpha.is_some().then_some(base[ALPHA]),
        masks: masks
            .into_iter()
            .map(|(channel, level, value, range)| {
                let curves = fits[range].iter().map(|f| (f.n, f.m, f.coefficients.clone())).collect();
                (channel, level, value, curves)
            })
            .collect(),
    };
    let (mut points, mut coefficients, mut total, mut weighted) = (0usize, 0usize, 0usize, 0.0);
    for f in &fits {
        points += f.n;
        total += 2 * f.m + 1;
        let nonzero = f.coefficients.iter().filter(|c| c.re != 0.0 || c.im != 0.0).count();
        coefficients += if options.prune.is_some() { nonzero } else { 2 * f.m + 1 };
        weighted += f.n as f64 * f.loss.sqrt();
    }
    let prune_json = options.prune.map_or(String::new(), |p| {
        let share = coefficients as f64 / total as f64;
        println!("{stem}: kept {coefficients} of {total} coefficients ({:.1}%)", share * 100.0);
        format!(
            "\"prune\":{{\"keep\":{},\"by\":\"{}\",\"rounds\":{},\"steps\":{},\"total_coefficients\":{total},\"kept_share\":{share:.6}}},",
            p.keep,
            p.by.name(),
            p.rounds,
            p.steps
        )
    });
    // Measured on the rendering that `render` gives at scale 1, so a packed
    // file is measured after its rounding, and transparency over white.
    let reference = match &alpha {
        Some(alpha) => {
            let mut flat = input.clone();
            for (i, a) in alpha.pixels().iter().enumerate() {
                let c = flat.get_mut(i % w, i / w).unwrap();
                *c = over_white(*c, *a);
            }
            eight_bit(&flat)?
        }
        None => eight_bit(&input)?,
    };
    let (file_name, measures, movement_json) = match options.movement {
        Some(movement) => {
            let bytes = packed::pack(&drawing, movement)?;
            let file_name = format!("{stem}_m{movement}.fvc");
            std::fs::write(output.join(&file_name), &bytes)?;
            (file_name, measure(&packed::unpack(&bytes)?.bitmap(1)?, &reference)?, format!("\"movement\":{movement},"))
        }
        None => {
            let file_name = format!("{stem}_coefficients.csv");
            drawing.write_coefficients(&output.join(&file_name), options.prune.is_some())?;
            (file_name, measure(&drawing.bitmap(1)?, &reference)?, String::new())
        }
    };
    let summary = output.join(format!("{stem}.json"));
    std::fs::write(
        &summary,
        format!(
            "{{\"image\":\"{name}\",\"source\":[{},{}],\"factor\":{factor},\"size\":[{w},{h}],\"fit\":\"{fit_name}\",{prune_json}\"levels\":{{{}}},\"outlines\":{},\"outline_points\":{points},\"coefficients\":{coefficients},\"dropped_small\":{dropped},\"edge_rms\":{:.6},\"measures\":{measures},{movement_json}\"coefficients_file\":\"{file_name}\",\"seconds\":{:.3}}}\n",
            source.width(),
            source.height(),
            levels_json.join(","),
            fits.len(),
            weighted / points.max(1) as f64,
            started.elapsed().as_secs_f64()
        ),
    )?;
    Ok(Fitted { coefficients: output.join(file_name), summary })
}

/// Packs a coefficients CSV written by `fit_image` into <stem>_m<movement>.fvc,
/// quantizing so that each curve moves by an RMS of `movement` pixels, which
/// loses accuracy. Returns the file written.
pub fn pack(output: &Path, path: &Path, movement: f64) -> Result<PathBuf, Error> {
    let drawing = Drawing::read(path)?;
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    let stem = name.trim_end_matches("_coefficients.csv");
    let file = output.join(format!("{stem}_m{movement}.fvc"));
    std::fs::write(&file, packed::pack(&drawing, movement)?)?;
    Ok(file)
}

/// Draws a file written by `fit_image` or `pack`, `scale` times larger, as the browser SVG (when `svg` gives its detail) and/or a BMP, named after
/// the file's stem (with _x<scale> when enlarged; an SVG also has _lines, _t<tolerance>,
/// _d<decimals> and _nofilter when they differ from the defaults). The BMP is filled
/// from the series itself. Returns the files written.
pub fn render(output: &Path, path: &Path, scale: usize, svg: Option<SvgDetail>, bitmap: bool) -> Result<Vec<PathBuf>, Error> {
    let drawing = Drawing::load(path)?;
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    let stem = name.trim_end_matches("_coefficients.csv").trim_end_matches(".fvc");
    let stem = if scale == 1 { stem.to_owned() } else { format!("{stem}_x{scale}") };
    let mut written = Vec::new();
    if let Some(detail) = svg {
        let mut name = stem.clone();
        if detail.lines {
            name.push_str("_lines");
        }
        if detail.tolerance != SvgDetail::default().tolerance {
            name.push_str(&format!("_t{}", detail.tolerance));
        }
        if detail.decimals != SvgDetail::default().decimals {
            name.push_str(&format!("_d{}", detail.decimals));
        }
        if detail.no_filter {
            name.push_str("_nofilter");
        }
        let file = output.join(format!("{name}.svg"));
        drawing.write_svg(&file, scale, detail)?;
        written.push(file);
    }
    if bitmap {
        let file = output.join(format!("{stem}.bmp"));
        drawing.bitmap(scale)?.save_bmp(&file)?;
        written.push(file);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bmp, prune::PruneBy};

    /// Two colour blocks and a disc on a light ground, saved as blocks.bmp.
    fn blocks(dir: &Path) -> (Image<Color>, PathBuf) {
        let (w, h) = (48, 40);
        let mut picture = Image::new(w, h, Color::rgb(0.9, 0.85, 0.7)).unwrap();
        for y in 0..h {
            for x in 0..w {
                let c = picture.get_mut(x, y).unwrap();
                if (6..20).contains(&x) && (5..30).contains(&y) {
                    *c = Color::rgb(0.8, 0.2, 0.1);
                } else if ((x as f64 - 33.0).powi(2) + (y as f64 - 20.0).powi(2)).sqrt() < 9.0 {
                    *c = Color::rgb(0.1, 0.2, 0.7);
                }
            }
        }
        let path = dir.join("blocks.bmp");
        picture.save_bmp(&path).unwrap();
        (picture, path)
    }

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join(format!(
            "{name}-{}",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn mean_delta_e(text: &str) -> f64 {
        text.split(r#""mean_delta_e":"#).nth(1).unwrap().split(',').next().unwrap().parse().unwrap()
    }

    fn dft_options(prune: Option<Prune>) -> FitOptions {
        FitOptions { factor: Some(1), levels: [4, 3, 3, 4], kind: FitKind::Dft, prune, movement: None }
    }

    #[test]
    fn render_redraws_a_fit_from_its_files() {
        let dir = scratch_dir("render");
        let (picture, path) = blocks(&dir);
        let (w, h) = (picture.width(), picture.height());
        let summary = fit_image(&dir, &path, &dft_options(None)).unwrap().summary;
        let recorded = mean_delta_e(&std::fs::read_to_string(&summary).unwrap());
        let csv = dir.join("blocks_dft_coefficients.csv");
        let written = render(&dir, &csv, 1, Some(SvgDetail::default()), true).unwrap();
        assert_eq!(written.len(), 2);
        assert!(std::fs::read_to_string(&written[0]).unwrap().contains(r##"filter="url(#lab)""##));
        // Drawn from the file (coefficients rounded to 9 decimals), the
        // picture measures as it did when it was fitted.
        let redrawn = bmp::read(&std::fs::read(&written[1]).unwrap()).unwrap();
        let again = mean_delta_e(&measure(&redrawn, &eight_bit(&picture).unwrap()).unwrap());
        assert!((again - recorded).abs() < 1e-3, "{again} vs {recorded}");
        assert!(recorded < 10.0, "{recorded}");
        // Enlarged, the same series fills a canvas twice the size.
        let large = render(&dir, &csv, 2, None, true).unwrap();
        let enlarged = bmp::read(&std::fs::read(&large[0]).unwrap()).unwrap();
        assert_eq!((enlarged.width(), enlarged.height()), (2 * w, 2 * h));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn pruned_fits_are_sparse_and_redraw_as_measured() {
        let dir = scratch_dir("prune");
        let (picture, path) = blocks(&dir);
        let reference = eight_bit(&picture).unwrap();
        let rows = |file: &Path| std::fs::read_to_string(file).unwrap().lines().count() - 1;
        fit_image(&dir, &path, &dft_options(None)).unwrap();
        let dense = rows(&dir.join("blocks_dft_coefficients.csv"));
        for (by, tag) in [(PruneBy::Largest, "largest"), (PruneBy::LowPass, "lowpass")] {
            let prune = Prune { keep: 0.5, by, rounds: 2, steps: 20 };
            let summary = fit_image(&dir, &path, &dft_options(Some(prune))).unwrap().summary;
            let stem = format!("blocks_dft_keep0.5_{tag}_r2_s20");
            assert!(summary.ends_with(format!("{stem}.json")), "{summary:?}");
            let csv = dir.join(format!("{stem}_coefficients.csv"));
            let kept = rows(&csv);
            assert!(kept < dense, "{kept} of {dense}");
            let text = std::fs::read_to_string(&summary).unwrap();
            assert!(text.contains(&format!("\"coefficients\":{kept},")), "{text}");
            assert!(text.contains(&format!("\"total_coefficients\":{dense},")), "{text}");
            // No curve collapsed (its centre and its turning term remain), and
            // missing rows read as zero.
            let drawing = Drawing::read(&csv).unwrap();
            for (_, m, c) in drawing.masks.iter().flat_map(|m| &m.3) {
                assert!(c[*m].abs() > 0.0 && c[*m - 1].abs().max(c[*m + 1].abs()) > 0.0);
                if by == PruneBy::LowPass {
                    let top = c.iter().rposition(|v| v.re != 0.0 || v.im != 0.0).unwrap() - *m;
                    assert!(top <= ((2 * *m + 1) as f64 * 0.5 / 2.0).round() as usize + 1, "{top} of {m}");
                }
            }
            // Drawn from the sparse file, it measures as it did when fitted.
            let written = render(&dir, &csv, 1, None, true).unwrap();
            let redrawn = bmp::read(&std::fs::read(&written[0]).unwrap()).unwrap();
            let again = mean_delta_e(&measure(&redrawn, &reference).unwrap());
            assert!((again - mean_delta_e(&text)).abs() < 1e-3, "{again} vs {text}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn packed_files_stand_alone_and_draw_about_the_same() {
        let dir = scratch_dir("packed");
        let (picture, path) = blocks(&dir);
        let reference = eight_bit(&picture).unwrap();
        fit_image(&dir, &path, &dft_options(None)).unwrap();
        let csv = dir.join("blocks_dft_coefficients.csv");
        let fine = pack(&dir, &csv, 0.01).unwrap();
        assert!(fine.ends_with("blocks_dft_m0.01.fvc"), "{fine:?}");
        assert!(std::fs::metadata(&fine).unwrap().len() < std::fs::metadata(&csv).unwrap().len() / 4);
        let delta = |file: &Path| {
            let written = render(&dir, file, 1, None, true).unwrap();
            let image = bmp::read(&std::fs::read(&written[0]).unwrap()).unwrap();
            mean_delta_e(&measure(&image, &reference).unwrap())
        };
        let exact = delta(&csv);
        // It needs nothing else: not the summary, not the CSV.
        std::fs::remove_file(dir.join("blocks_dft.json")).unwrap();
        std::fs::remove_file(&csv).unwrap();
        // This small picture's long straight edges lie close to where the fill
        // samples, so a tiny movement can flip many of them at once.
        let packed = delta(&fine);
        assert!((packed - exact).abs() < 0.1, "{packed} vs {exact}");
        let svg = render(&dir, &fine, 1, Some(SvgDetail::default()), false).unwrap();
        assert!(svg[0].ends_with("blocks_dft_m0.01.svg"), "{svg:?}");
        // Packing takes a CSV, not its own output.
        assert!(pack(&dir, &fine, 0.2).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_fit_packs_by_default_and_records_how_the_packed_file_draws() {
        let dir = scratch_dir("fit-packed");
        let (picture, path) = blocks(&dir);
        let options = FitOptions { movement: Some(DEFAULT_MOVEMENT), ..dft_options(None) };
        let fitted = fit_image(&dir, &path, &options).unwrap();
        assert!(fitted.coefficients.ends_with("blocks_dft_m0.2.fvc"), "{:?}", fitted.coefficients);
        assert!(!dir.join("blocks_dft_coefficients.csv").exists());
        let text = std::fs::read_to_string(&fitted.summary).unwrap();
        assert!(text.contains(r#""movement":0.2,"coefficients_file":"blocks_dft_m0.2.fvc""#), "{text}");
        let written = render(&dir, &fitted.coefficients, 1, None, true).unwrap();
        let redrawn = bmp::read(&std::fs::read(&written[0]).unwrap()).unwrap();
        let again = mean_delta_e(&measure(&redrawn, &eight_bit(&picture).unwrap()).unwrap());
        assert!((again - mean_delta_e(&text)).abs() < 1e-3, "{again} vs {text}");
        // A pruned fit packs too, and stays sparse.
        let prune = Prune { keep: 0.5, by: PruneBy::Largest, rounds: 1, steps: 10 };
        let options = FitOptions { prune: Some(prune), ..options };
        let fitted = fit_image(&dir, &path, &options).unwrap();
        assert!(fitted.coefficients.ends_with("blocks_dft_keep0.5_largest_r1_s10_m0.2.fvc"), "{:?}", fitted.coefficients);
        assert!(!render(&dir, &fitted.coefficients, 1, None, true).unwrap().is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The blocks with a clear left third and a soft fade down the right third.
    /// The colour stored under fully clear pixels is `hidden`.
    fn transparent_blocks(dir: &Path, name: &str, hidden: [u8; 3]) -> PathBuf {
        let (w, h) = (48u32, 40u32);
        let picture = ::image::RgbaImage::from_fn(w, h, |x, y| {
            let alpha = match x {
                0..16 => 0,
                16..32 => 255,
                _ => 255 - (y * 255 / (h - 1)) as u8,
            };
            let disc = (x as f64 - 24.0).hypot(y as f64 - 20.0) < 7.0;
            let [r, g, b] = if alpha == 0 {
                hidden
            } else if disc {
                [30, 50, 180]
            } else {
                [230, 215, 180]
            };
            ::image::Rgba([r, g, b, alpha])
        });
        let path = dir.join(name);
        picture.save(&path).unwrap();
        path
    }

    #[test]
    fn transparency_survives_every_file_and_ignores_hidden_colours() {
        let dir = scratch_dir("alpha");
        let path = transparent_blocks(&dir, "clear.png", [0, 0, 0]);
        let options = FitOptions { levels: [4, 3, 3, 6], ..dft_options(None) };
        let summary = fit_image(&dir, &path, &options).unwrap().summary;
        let text = std::fs::read_to_string(&summary).unwrap();
        assert!(text.contains("\"alpha\":{\"levels\":[0.000000000,"), "{text}");
        let csv = dir.join("clear_dft_coefficients.csv");
        assert!(std::fs::read_to_string(&csv).unwrap().contains("\nalpha,"));
        let recorded = mean_delta_e(&text);
        assert!(recorded < 6.0, "{recorded}");
        // From the CSV: masked, and the BMP measures as recorded, over white.
        let written = render(&dir, &csv, 1, Some(SvgDetail::default()), true).unwrap();
        assert!(std::fs::read_to_string(&written[0]).unwrap().contains("mask=\"url(#alpha)\""));
        let redrawn = bmp::read(&std::fs::read(&written[1]).unwrap()).unwrap();
        assert_eq!(*redrawn.get(4, 20).unwrap(), Color::rgb(1.0, 1.0, 1.0));
        // Packed: the same plate, drawn about the same.
        let fitted = fit_image(&dir, &path, &FitOptions { movement: Some(0.01), ..options }).unwrap();
        assert!(fitted.coefficients.ends_with("clear_dft_m0.01.fvc"));
        let written = render(&dir, &fitted.coefficients, 1, Some(SvgDetail::default()), true).unwrap();
        assert!(std::fs::read_to_string(&written[0]).unwrap().contains("mask=\"url(#alpha)\""));
        let packed = mean_delta_e(&std::fs::read_to_string(&fitted.summary).unwrap());
        assert!((packed - recorded).abs() < 0.2, "{packed} vs {recorded}");
        // What is stored under clear pixels makes no difference at all.
        let other = transparent_blocks(&dir, "other.png", [255, 0, 200]);
        let again = fit_image(&dir, &other, &FitOptions { movement: Some(0.01), ..options }).unwrap();
        assert_eq!(std::fs::read(&again.coefficients).unwrap(), std::fs::read(&fitted.coefficients).unwrap());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_cut_off_coefficients_file_is_refused() {
        let dir = scratch_dir("cutoff");
        let (_, path) = blocks(&dir);
        fit_image(&dir, &path, &dft_options(None)).unwrap();
        let csv = dir.join("blocks_dft_coefficients.csv");
        let text = std::fs::read_to_string(&csv).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        std::fs::write(&csv, lines[..lines.len() - 1].join("\n")).unwrap();
        assert!(render(&dir, &csv, 1, None, true).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn svg_detail_trades_size_and_names_the_file() {
        let dir = scratch_dir("detail");
        let (_, path) = blocks(&dir);
        fit_image(&dir, &path, &dft_options(None)).unwrap();
        let csv = dir.join("blocks_dft_coefficients.csv");
        let svg = |tolerance: f64, decimals: u32| {
            render(&dir, &csv, 1, Some(SvgDetail { tolerance, decimals, ..SvgDetail::default() }), false).unwrap().remove(0)
        };
        let size = |file: &Path| std::fs::metadata(file).unwrap().len();
        let (base, coarse, rough, fine) = (svg(0.1, 1), svg(1.0, 1), svg(0.1, 0), svg(0.1, 2));
        assert!(base.ends_with("blocks_dft.svg"), "{base:?}");
        assert!(coarse.ends_with("blocks_dft_t1.svg"), "{coarse:?}");
        assert!(rough.ends_with("blocks_dft_d0.svg"), "{rough:?}");
        assert!(fine.ends_with("blocks_dft_d2.svg"), "{fine:?}");
        assert!(size(&coarse) < size(&base));
        assert!(size(&rough) < size(&base));
        assert!(size(&fine) > size(&base));
        let lines = render(&dir, &csv, 1, Some(SvgDetail { lines: true, ..SvgDetail::default() }), false).unwrap().remove(0);
        assert!(lines.ends_with("blocks_dft_lines.svg"), "{lines:?}");
        let flat = render(&dir, &csv, 1, Some(SvgDetail { no_filter: true, ..SvgDetail::default() }), false).unwrap().remove(0);
        assert!(flat.ends_with("blocks_dft_nofilter.svg"), "{flat:?}");
        assert!(!std::fs::read_to_string(&flat).unwrap().contains("filter="));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
