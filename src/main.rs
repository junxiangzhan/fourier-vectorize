//! Command-line entry point.
use fourier_vectorize::{fit_image, render, Error, FitKind, FitOptions, Prune, PruneBy, SvgDetail};
use std::{path::PathBuf, str::FromStr};

const HELP: &str = "\
fourier-vectorize fit IMAGE... [--out DIR] [--factor N] [--levels 32,16,16] [--method dft|refined|descent]
                      [--keep SHARE [--prune largest|lowpass] [--rounds N] [--prune-steps N]]
    Fit a PNG, JPEG or uncompressed 24-bit BMP with Fourier series, whatever the file's extension.
    Writes <name>_<method>_coefficients.csv and a JSON summary with mean dE and SSIM (default method:
    refined). --factor N shrinks by box-averaging N x N blocks; without it, the smallest factor that
    brings the longer side to at most 2000 px is used.
    --keep drops coefficients until SHARE of them (0..1, exclusive) remain, then descends again, which
    costs accuracy. --prune picks them by size (largest, default) or lowest harmonics (lowpass);
    --rounds cuts in N equal ratios (default 1, all at once); --prune-steps is the total descent
    (default 400): half over the rounds, half on the final set. The file name records all of these.
fourier-vectorize render COEFFICIENTS.csv... [--out DIR] [--scale N] [--svg] [--bmp] [--tolerance PX] [--decimals N]
    Draw coefficients written by `fit` as a browser SVG and/or a BMP, N times larger (default 1, both
    files). SVG only: --tolerance is how far simplifying may move an outline, in output pixels
    (default 0.1; 0 drops only points exactly in line); --decimals is the coordinates' decimal places,
    0..4 (default 1). Other values go in the SVG's file name.

The default output directory is the current one.";

enum Job {
    Fit(FitOptions),
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
    let is_fit = match command.as_deref() {
        Some("fit") => true,
        Some("render") => false,
        None | Some("--help") | Some("-h") => {
            println!("{HELP}");
            return Ok(None);
        }
        _ => return Err(Error::InvalidParameter("the command is fit or render; use --help")),
    };
    let mut output = PathBuf::from(".");
    let mut files = Vec::new();
    let mut options = FitOptions::default();
    let mut prune = Prune { keep: 1.0, by: PruneBy::Largest, rounds: 1, steps: 400 };
    let mut prune_option = false;
    let (mut scale, mut svg, mut bmp) = (1, false, false);
    let mut detail = SvgDetail::default();
    let mut detail_option = false;
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
                let message = "--levels needs three integers in 2..64, e.g. 32,16,16";
                let list = args.next().ok_or(Error::InvalidParameter(message))?;
                let values: Vec<usize> = list.split(',').filter_map(|v| v.parse().ok()).collect();
                if values.len() != 3 || values.iter().any(|v| !(2..=64).contains(v)) {
                    return Err(Error::InvalidParameter(message));
                }
                options.levels = [values[0], values[1], values[2]];
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
            "--scale" if !is_fit => {
                scale = value(&mut args, "--scale needs an integer in 1..16")?;
                if !(1..=16).contains(&scale) {
                    return Err(Error::InvalidParameter("--scale needs an integer in 1..16"));
                }
            }
            "--svg" if !is_fit => svg = true,
            "--bmp" if !is_fit => bmp = true,
            "--tolerance" if !is_fit => {
                let message = "--tolerance needs a distance of at least 0 pixels";
                detail.tolerance = value(&mut args, message)?;
                if !(detail.tolerance.is_finite() && detail.tolerance >= 0.0) {
                    return Err(Error::InvalidParameter(message));
                }
                detail_option = true;
            }
            "--decimals" if !is_fit => {
                detail.decimals = value(&mut args, "--decimals needs an integer in 0..4")?;
                if detail.decimals > 4 {
                    return Err(Error::InvalidParameter("--decimals needs an integer in 0..4"));
                }
                detail_option = true;
            }
            _ if arg.starts_with('-') => return Err(Error::InvalidParameter("unknown argument for this command; use --help")),
            _ => files.push(PathBuf::from(arg)),
        }
    }
    if files.is_empty() {
        return Err(Error::InvalidParameter("fit needs images and render needs coefficient files; use --help"));
    }
    let job = if is_fit {
        if prune_option && prune.keep >= 1.0 {
            return Err(Error::InvalidParameter("--prune, --rounds and --prune-steps need --keep"));
        }
        options.prune = (prune.keep < 1.0).then_some(prune);
        Job::Fit(options)
    } else {
        if detail_option && bmp && !svg {
            return Err(Error::InvalidParameter("--tolerance and --decimals only change the SVG; a BMP is filled from the series"));
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
                println!("Fitted: {}", fit_image(&config.output, file, options)?.display());
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
