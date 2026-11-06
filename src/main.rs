//! Command-line entry point.
use fourier_vectorize::{fit_image, pack, render, Error, FitKind, FitOptions, Prune, PruneBy, SvgDetail, DEFAULT_MOVEMENT};
use std::{path::PathBuf, str::FromStr};

const HELP: &str = "\
fourier-vectorize fit IMAGE... [--out DIR] [--factor N] [--levels 32,16,16[,16]] [--palette N] [--method dft|refined|descent]
                      [--movement PX | --csv] [--reduce PX | --keep SHARE [--prune largest|lowpass] [--rounds N] [--prune-steps N]]
    Fit a PNG, JPEG or uncompressed 24-bit BMP with Fourier series, whatever the file's extension. EXIF
    orientation is applied. A picture with transparency keeps it in its SVG and is drawn over white in
    its BMP and measures; the fourth number of --levels is for its fades.
    Writes <name>_<method>_m<PX>.fvc, a small binary file for `render`, and a JSON summary with mean dE
    and SSIM (default method: refined). --movement is how far rounding the coefficients may move each
    curve, as an RMS in pixels (default 0.2), which costs accuracy; --csv writes
    <name>_<method>_coefficients.csv instead, which is far larger and can be packed later.
    --palette N paints the picture in a table of up to N colours (2..1024), each one shape over the ones
    before, instead of the L*, a* and b* plates, so the SVG needs no filter; --levels then sets only the
    fades. It goes in the file name, and only in a packed file.
    --factor N shrinks by box-averaging N x N blocks; without it, the smallest factor that brings the
    longer side to at most 2000 px is used.
    --reduce gives each curve only the harmonics it needs to stay this close, as an RMS in pixels
    (default 0.3; 0 keeps them all), which costs accuracy. Other values go in the file name.
    --keep drops coefficients until SHARE of them (0..1, exclusive) remain, then descends again, which
    costs accuracy. --prune picks them by size (largest, default) or lowest harmonics (lowpass);
    --rounds cuts in N equal ratios (default 1, all at once); --prune-steps is the total descent
    (default 400): half over the rounds, half on the final set. The file name records all of these.
fourier-vectorize pack COEFFICIENTS.csv... [--out DIR] [--movement PX]
    Pack a CSV written by `fit --csv` into <name>_m<PX>.fvc, a small binary file that needs nothing
    else to be drawn. --movement is how far rounding the coefficients may move each curve, as an RMS
    in pixels (default 0.2), which costs accuracy.
fourier-vectorize render FILE... [--out DIR] [--scale N] [--svg] [--bmp] [--lines] [--tolerance PX] [--decimals N] [--no-filter]
    Draw coefficients written by `fit` or `pack` as a browser SVG and/or a BMP, N times larger (default 1, both
    files). SVG only: outlines are cubic curves and lines, or with --lines lines only, whose corners
    show when enlarged; --tolerance is how far simplifying may move an outline, in output pixels
    (default 0.1; 0 drops only points exactly in line); --decimals is the coordinates' decimal places,
    0..4 (default 1); --no-filter paints one colour per combination of levels through clip paths instead
    of using an SVG filter, for software without filters, and is far slower to draw. Other settings go in
    the SVG's file name.

The default output directory is the current one.";

enum Job {
    Fit(FitOptions),
    Pack { movement: f64 },
    Render { scale: usize, svg: Option<SvgDetail>, bmp: bool },
}

struct Config {
    job: Job,
    output: PathBuf,
    files: Vec<PathBuf>,
}

/// The next argument, parsed.
fn value<T: FromStr>(args: &mut impl Iterator<Item = String>, message: &'static str) -> Result<T, Error> {
    args.next().and_then(|v| v.parse().ok()).ok_or(Error::InvalidParameter(message))
}

/// Parses the arguments; Ok(None) means the help was printed.
fn parse(mut args: impl Iterator<Item = String>) -> Result<Option<Config>, Error> {
    let command = args.next();
    let (is_fit, is_pack) = match command.as_deref() {
        Some("fit") => (true, false),
        Some("pack") => (false, true),
        Some("render") => (false, false),
        None | Some("--help") | Some("-h") => {
            println!("{HELP}");
            return Ok(None);
        }
        _ => return Err(Error::InvalidParameter("the command is fit, pack or render; use --help")),
    };
    let mut output = PathBuf::from(".");
    let mut files = Vec::new();
    let mut options = FitOptions::default();
    let mut prune = Prune { keep: 1.0, by: PruneBy::Largest, rounds: 1, steps: 400 };
    let mut prune_option = false;
    let (mut scale, mut svg, mut bmp) = (1, false, false);
    let mut detail = SvgDetail::default();
    let mut detail_option = false;
    let mut movement = None;
    let mut csv = false;
    let mut reduce_option = false;
    let is_render = !is_fit && !is_pack;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(None);
            }
            "--out" => output = PathBuf::from(args.next().ok_or(Error::InvalidParameter("--out needs a directory"))?),
            "--factor" if is_fit => {
                let factor: usize = value(&mut args, "--factor needs a positive integer")?;
                if factor == 0 {
                    return Err(Error::InvalidParameter("--factor needs a positive integer"));
                }
                options.factor = Some(factor);
            }
            "--levels" if is_fit => {
                let message = "--levels needs three or four integers in 2..64, e.g. 32,16,16";
                let list = args.next().ok_or(Error::InvalidParameter(message))?;
                let values: Vec<usize> = list.split(',').filter_map(|v| v.parse().ok()).collect();
                if !(3..=4).contains(&values.len()) || values.iter().any(|v| !(2..=64).contains(v)) {
                    return Err(Error::InvalidParameter(message));
                }
                options.levels[..values.len()].copy_from_slice(&values);
            }
            "--palette" if is_fit => {
                let message = "--palette needs a number of colours in 2..1024";
                let colours: usize = value(&mut args, message)?;
                if !(2..=1024).contains(&colours) {
                    return Err(Error::InvalidParameter(message));
                }
                options.palette = Some(colours);
            }
            "--method" if is_fit => {
                options.kind = match args.next().as_deref() {
                    Some("dft") => FitKind::Dft,
                    Some("refined") => FitKind::Refined,
                    Some("descent") => FitKind::Descent,
                    _ => return Err(Error::InvalidParameter("--method needs dft, refined or descent")),
                };
            }
            "--keep" if is_fit => {
                let message = "--keep needs a share above 0 and below 1";
                prune.keep = value(&mut args, message)?;
                if !(prune.keep > 0.0 && prune.keep < 1.0) {
                    return Err(Error::InvalidParameter(message));
                }
            }
            "--prune" if is_fit => {
                prune.by = match args.next().as_deref() {
                    Some("largest") => PruneBy::Largest,
                    Some("lowpass") => PruneBy::LowPass,
                    _ => return Err(Error::InvalidParameter("--prune needs largest or lowpass")),
                };
                prune_option = true;
            }
            "--rounds" if is_fit => {
                prune.rounds = value(&mut args, "--rounds needs an integer in 1..64")?;
                if !(1..=64).contains(&prune.rounds) {
                    return Err(Error::InvalidParameter("--rounds needs an integer in 1..64"));
                }
                prune_option = true;
            }
            "--prune-steps" if is_fit => {
                prune.steps = value(&mut args, "--prune-steps needs an integer")?;
                prune_option = true;
            }
            "--reduce" if is_fit => {
                let message = "--reduce needs a distance of at least 0 pixels";
                let budget: f64 = value(&mut args, message)?;
                if !(budget >= 0.0 && budget.is_finite()) {
                    return Err(Error::InvalidParameter(message));
                }
                options.reduce = (budget > 0.0).then_some(budget);
                reduce_option = true;
            }
            "--scale" if is_render => {
                scale = value(&mut args, "--scale needs an integer in 1..16")?;
                if !(1..=16).contains(&scale) {
                    return Err(Error::InvalidParameter("--scale needs an integer in 1..16"));
                }
            }
            "--svg" if is_render => svg = true,
            "--bmp" if is_render => bmp = true,
            "--lines" if is_render => {
                detail.lines = true;
                detail_option = true;
            }
            "--tolerance" if is_render => {
                let message = "--tolerance needs a distance of at least 0 pixels";
                detail.tolerance = value(&mut args, message)?;
                if !(detail.tolerance.is_finite() && detail.tolerance >= 0.0) {
                    return Err(Error::InvalidParameter(message));
                }
                detail_option = true;
            }
            "--decimals" if is_render => {
                detail.decimals = value(&mut args, "--decimals needs an integer in 0..4")?;
                if detail.decimals > 4 {
                    return Err(Error::InvalidParameter("--decimals needs an integer in 0..4"));
                }
                detail_option = true;
            }
            "--no-filter" if is_render => {
                detail.no_filter = true;
                detail_option = true;
            }
            "--movement" if is_fit || is_pack => {
                let message = "--movement needs a distance above 0 pixels";
                let distance: f64 = value(&mut args, message)?;
                if !(distance > 0.0 && distance.is_finite()) {
                    return Err(Error::InvalidParameter(message));
                }
                movement = Some(distance);
            }
            "--csv" if is_fit => csv = true,
            _ if arg.starts_with('-') => return Err(Error::InvalidParameter("unknown argument for this command; use --help")),
            _ => files.push(PathBuf::from(arg)),
        }
    }
    if files.is_empty() {
        return Err(Error::InvalidParameter("fit needs images; pack and render need coefficient files; use --help"));
    }
    let job = if is_fit {
        if prune_option && prune.keep >= 1.0 {
            return Err(Error::InvalidParameter("--prune, --rounds and --prune-steps need --keep"));
        }
        if csv && options.palette.is_some() {
            return Err(Error::InvalidParameter("--palette needs a packed file; --csv writes a CSV, which holds plates only"));
        }
        if csv && movement.is_some() {
            return Err(Error::InvalidParameter("--movement sets the rounding of a packed file; --csv writes a CSV instead"));
        }
        if reduce_option && prune.keep < 1.0 {
            return Err(Error::InvalidParameter("--reduce and --keep are two ways to drop coefficients; use one"));
        }
        options.prune = (prune.keep < 1.0).then_some(prune);
        options.movement = (!csv).then_some(movement.unwrap_or(DEFAULT_MOVEMENT));
        Job::Fit(options)
    } else if is_pack {
        Job::Pack { movement: movement.unwrap_or(DEFAULT_MOVEMENT) }
    } else {
        if detail_option && bmp && !svg {
            return Err(Error::InvalidParameter("--lines, --tolerance, --decimals and --no-filter only change the SVG; a BMP is filled from the series"));
        }
        if !svg && !bmp {
            (svg, bmp) = (true, true);
        }
        Job::Render { scale, svg: svg.then_some(detail), bmp }
    };
    Ok(Some(Config { job, output, files }))
}

fn run() -> Result<(), Error> {
    let Some(config) = parse(std::env::args().skip(1))? else {
        return Ok(());
    };
    std::fs::create_dir_all(&config.output)?;
    for file in &config.files {
        match &config.job {
            Job::Fit(options) => {
                let fitted = fit_image(&config.output, file, options)?;
                println!("Fitted: {}", fitted.coefficients.display());
                println!("Summary: {}", fitted.summary.display());
            }
            Job::Pack { movement } => {
                println!("Packed: {}", pack(&config.output, file, *movement)?.display());
            }
            Job::Render { scale, svg, bmp } => {
                for written in render(&config.output, file, *scale, *svg, *bmp)? {
                    println!("Rendered: {}", written.display());
                }
            }
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}
