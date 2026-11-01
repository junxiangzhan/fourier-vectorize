//! Command-line entry point. Without a subcommand (or with `all`) it runs
//! every Part 1-4 experiment in order and writes all results into one
//! output directory; the other subcommands run one part, or one painting
//! task, on its own.
mod experiments;
mod hifi;
mod layers;
mod measurements;
mod paintings;
mod presentation;
mod recover;
mod transform;
mod vectorize;
mod wheel;
use multimedia::Error;
use std::path::{Path, PathBuf};

const HELP: &str = "\
multimedia [all] [OUTPUT] [--out DIR] [--size 128..2048] [--wedges 40..512] [--paintings FOLDER]
    Every Part 1-4 experiment (the painting showcase on every BMP in FOLDER).
multimedia part1|part2|part3|part4 [--out DIR] [--size N] [--wedges N] [--paintings FOLDER]
    One part only. run.json is marked partial, so the audit refuses it.
multimedia paint IMAGE.bmp... [--paintings FOLDER] [--out DIR] [--factor N] [--methods k24_stack,lab1266,...]
    The painting showcase (palette, CMY and Lab vectorizing) on chosen images.
    Methods: k8 k16 k24 k24_stack k24_stack_clean cmy888 lab888 lab1266 lab1266_clean (default all).
multimedia hifi --paintings FOLDER [--out DIR] [--factor N]
    The high-fidelity comparison (crack vs smooth outlines; DFT, refined, descent).
multimedia hifi IMAGE.bmp... [--out DIR] [--factor N] [--levels 32,16,16] [--fit dft|refined|descent]
    Fit an image with Fourier series and write its coefficients (export_<name>_<fit>_coefficients.csv)
    and a JSON summary with mean dE and SSIM (default fit: refined).
multimedia render COEFFICIENTS.csv... [--out DIR] [--scale N] [--svg] [--bmp]
    Draw coefficients written by `hifi IMAGE` as a browser SVG and/or a BMP, N times larger
    (default 1, both files).

--factor N shrinks by box-averaging N x N blocks. Without it, the smallest factor that
brings the longer side to at most 1000 px (paint) or 2000 px (hifi) is used.
JPEG files are decoded to BMP first with: python scripts/decode_paintings.py [FOLDER]
Default output: output/, size 384, wedges 96. No Python or Node runtime is needed.";

#[derive(Clone, Copy, PartialEq)]
enum Command {
    All,
    Part(u8),
    Paint,
    Hifi,
    Render,
}

/// Command-line options. Only the ones a command uses may be given.
struct Config {
    command: Command,
    output: PathBuf,
    size: usize,
    wedges: usize,
    /// Folder of decoded painting BMPs.
    paintings: Option<PathBuf>,
    /// Files named on the command line (images for paint and hifi,
    /// coefficient files for render).
    images: Vec<PathBuf>,
    factor: Option<usize>,
    methods: Option<Vec<String>>,
    levels: [usize; 3],
    fit: hifi::FitKind,
    scale: usize,
    svg: bool,
    bmp: bool,
}
impl Config {
    /// Parses the arguments; Ok(None) means --help was printed.
    fn parse() -> Result<Option<Self>, Error> {
        let mut args = std::env::args().skip(1).peekable();
        let named = match args.peek().map(String::as_str) {
            Some("all") => Some(Command::All),
            Some("part1") => Some(Command::Part(1)),
            Some("part2") => Some(Command::Part(2)),
            Some("part3") => Some(Command::Part(3)),
            Some("part4") => Some(Command::Part(4)),
            Some("paint") => Some(Command::Paint),
            Some("hifi") => Some(Command::Hifi),
            Some("render") => Some(Command::Render),
            _ => None,
        };
        // Without a subcommand: everything, as before subcommands existed.
        if named.is_some() {
            args.next();
        }
        let command = named.unwrap_or(Command::All);
        let mut config = Self {
            command,
            output: PathBuf::from("output"),
            size: 384,
            wedges: 96,
            paintings: None,
            images: Vec::new(),
            factor: None,
            methods: None,
            levels: [32, 16, 16],
            fit: hifi::FitKind::Refined,
            scale: 1,
            svg: false,
            bmp: false,
        };
        let (mut explicit_output, mut legacy_hifi) = (false, false);
        let (mut uses_canvas, mut uses_export, mut uses_render) = (false, false, false);
        let number = |value: Option<String>, message: &'static str| {
            value
                .ok_or(Error::InvalidParameter(message))?
                .parse::<usize>()
                .map_err(|_| Error::InvalidParameter(message))
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    println!("{HELP}");
                    return Ok(None);
                }
                "--out" => {
                    if explicit_output {
                        return Err(Error::InvalidParameter("output directory supplied twice"));
                    }
                    config.output = PathBuf::from(args.next().ok_or(Error::InvalidParameter("--out needs a directory"))?);
                    explicit_output = true;
                }
                "--size" => {
                    config.size = number(args.next(), "--size needs an integer")?;
                    uses_canvas = true;
                }
                "--wedges" => {
                    config.wedges = number(args.next(), "--wedges needs an integer")?;
                    uses_canvas = true;
                }
                "--paintings" => {
                    config.paintings = Some(PathBuf::from(args.next().ok_or(Error::InvalidParameter("--paintings needs a folder"))?));
                }
                // Earlier form of `hifi --paintings FOLDER`.
                "--hifi" => legacy_hifi = true,
                "--factor" => {
                    let factor = number(args.next(), "--factor needs a positive integer")?;
                    if factor == 0 {
                        return Err(Error::InvalidParameter("--factor needs a positive integer"));
                    }
                    config.factor = Some(factor);
                }
                "--methods" => {
                    let list = args.next().ok_or(Error::InvalidParameter("--methods needs a list"))?;
                    let methods: Vec<String> = list.split(',').map(str::to_owned).collect();
                    if methods.iter().any(|m| !paintings::METHOD_NAMES.contains(&m.as_str())) {
                        return Err(Error::InvalidParameter("unknown method in --methods; use --help"));
                    }
                    config.methods = Some(methods);
                }
                "--levels" => {
                    let list = args.next().ok_or(Error::InvalidParameter("--levels needs three integers"))?;
                    let values: Vec<usize> = list.split(',').filter_map(|v| v.parse().ok()).collect();
                    if values.len() != 3 || values.iter().any(|v| !(2..=64).contains(v)) {
                        return Err(Error::InvalidParameter("--levels needs three integers in 2..64, e.g. 32,16,16"));
                    }
                    config.levels = [values[0], values[1], values[2]];
                    uses_export = true;
                }
                "--fit" => {
                    config.fit = match args.next().as_deref() {
                        Some("dft") => hifi::FitKind::Dft,
                        Some("refined") => hifi::FitKind::Refined,
                        Some("descent") => hifi::FitKind::Descent,
                        _ => return Err(Error::InvalidParameter("--fit needs dft, refined or descent")),
                    };
                    uses_export = true;
                }
                "--scale" => {
                    config.scale = number(args.next(), "--scale needs an integer in 1..16")?;
                    if !(1..=16).contains(&config.scale) {
                        return Err(Error::InvalidParameter("--scale needs an integer in 1..16"));
                    }
                    uses_render = true;
                }
                "--svg" => {
                    config.svg = true;
                    uses_render = true;
                }
                "--bmp" => {
                    config.bmp = true;
                    uses_render = true;
                }
                "--export" => {
                    return Err(Error::InvalidParameter(
                        "hifi now writes coefficients only; draw them with `render COEFFICIENTS.csv --svg --bmp`",
                    ))
                }
                _ if arg.starts_with('-') => return Err(Error::InvalidParameter("unknown argument; use --help")),
                _ => match config.command {
                    Command::Paint | Command::Hifi | Command::Render => config.images.push(PathBuf::from(arg)),
                    _ => {
                        if explicit_output {
                            return Err(Error::InvalidParameter("unexpected positional argument"));
                        }
                        config.output = PathBuf::from(arg);
                        explicit_output = true;
                    }
                },
            }
        }
        if legacy_hifi {
            if config.command != Command::All {
                return Err(Error::InvalidParameter("--hifi only goes with the plain command; use `hifi`"));
            }
            config.command = Command::Hifi;
        }
        let painting_task = matches!(config.command, Command::Paint | Command::Hifi | Command::Render);
        if uses_render && config.command != Command::Render {
            return Err(Error::InvalidParameter("--scale, --svg and --bmp belong to render"));
        }
        if config.command == Command::Render {
            if config.images.is_empty() || config.paintings.is_some() || config.factor.is_some() || config.methods.is_some() || uses_export {
                return Err(Error::InvalidParameter("render takes coefficient files and --out, --scale, --svg, --bmp"));
            }
            if !config.svg && !config.bmp {
                (config.svg, config.bmp) = (true, true);
            }
        }
        if painting_task && uses_canvas {
            return Err(Error::InvalidParameter("--size and --wedges belong to the experiments, not paint or hifi"));
        }
        if !painting_task && (config.factor.is_some() || config.methods.is_some() || uses_export) {
            return Err(Error::InvalidParameter("--factor, --methods, --levels and --fit belong to paint or hifi"));
        }
        if config.command == Command::Paint && uses_export {
            return Err(Error::InvalidParameter("--levels and --fit belong to hifi"));
        }
        if config.command == Command::Hifi && config.methods.is_some() {
            return Err(Error::InvalidParameter("--methods belongs to paint"));
        }
        if config.command == Command::Hifi && !config.images.is_empty() && config.paintings.is_some() {
            return Err(Error::InvalidParameter("hifi takes images to export, or --paintings for the comparison, not both"));
        }
        if config.command == Command::Hifi && config.images.is_empty() && uses_export {
            return Err(Error::InvalidParameter("--levels and --fit need images to fit"));
        }
        if painting_task && config.images.is_empty() && config.paintings.is_none() {
            return Err(Error::InvalidParameter("paint and hifi need images or --paintings FOLDER"));
        }
        if !(128..=2048).contains(&config.size) {
            return Err(Error::InvalidParameter(
                "experiment size must be 128..2048; library images have no such limit",
            ));
        }
        if !(40..=512).contains(&config.wedges) {
            return Err(Error::InvalidParameter("wheel wedges must be 40..512"));
        }
        if config.output.as_os_str().is_empty() {
            return Err(Error::InvalidParameter("output directory must not be empty"));
        }
        Ok(Some(config))
    }
}

/// Every BMP in `folder`, in name order.
fn bitmaps_in(folder: &Path) -> Result<Vec<PathBuf>, Error> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(folder)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("bmp")))
        .collect();
    found.sort();
    if found.is_empty() {
        return Err(Error::InvalidParameter(
            "no .bmp files there; decode JPEGs with python scripts/decode_paintings.py",
        ));
    }
    Ok(found)
}

/// Quotes and escapes a string for the hand-written JSON in run.json.
fn json_string(value: &str) -> String {
    let mut result = String::from("\"");
    for c in value.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c < ' ' => result.push_str(&format!("\\u{:04x}", c as u32)),
            c => result.push(c),
        }
    }
    result.push('"');
    result
}

fn run() -> Result<(), Error> {
    let Some(config) = Config::parse()? else {
        return Ok(());
    };
    let output = &config.output;
    std::fs::create_dir_all(output)?;
    let size = config.size;
    let wedges = config.wedges;
    // Painting tasks leave run.json and every other result untouched.
    let images = || -> Result<Vec<PathBuf>, Error> {
        match &config.paintings {
            Some(folder) if config.images.is_empty() => bitmaps_in(folder),
            _ => Ok(config.images.clone()),
        }
    };
    match config.command {
        Command::Paint => {
            paintings::run(output, &images()?, config.factor, config.methods.as_deref())?;
            println!("Complete: {}", output.join("paintings_summary.json").display());
            return Ok(());
        }
        Command::Hifi if config.images.is_empty() => {
            hifi::run(output, &images()?, config.factor, env!("SOURCE_FINGERPRINT"))?;
            println!("Complete: {}", output.join("hifi_summary.json").display());
            return Ok(());
        }
        Command::Hifi => {
            for image in &config.images {
                let written = hifi::export(output, image, config.factor, config.levels, config.fit)?;
                println!("Exported: {}", written.display());
            }
            return Ok(());
        }
        Command::Render => {
            for file in &config.images {
                for written in hifi::render(output, file, config.scale, config.svg, config.bmp)? {
                    println!("Rendered: {}", written.display());
                }
            }
            return Ok(());
        }
        _ => {}
    }
    let part = |n: u8| config.command == Command::All || config.command == Command::Part(n);
    // Marked in-progress first, so a run that fails midway can never leave
    // a "complete" manifest next to stale results.
    std::fs::write(output.join("run.json"), "{\"status\":\"in-progress\"}\n")?;
    if part(1) {
        experiments::diagnostic(output)?;
        experiments::curves(output, size)?;
        println!("01-03: BMP and curve convergence written");
        experiments::joins(output, size)?;
        println!("04: joins written");
    }
    if part(2) {
        experiments::star(output, size)?;
        println!("05: star winding written");
        experiments::rings(output, size)?;
        println!("06: circle direction comparison written");
    }
    if part(3) {
        let images = experiments::antialias(output, size)?;
        println!("07: four supersampling levels written");
        measurements::run(output, size, &images)?;
        println!("08: row measurements and timing written");
        wheel::run(output, size, wedges)?;
        println!("09: wheel comparison written");
    }
    if part(4) {
        transform::run(output, size)?;
        println!("Part 4: vector vs bitmap rotation written");
        vectorize::run(output)?;
        println!("Part 4 appendix: Hilbert scan vs Fourier outline written");
        recover::run(output)?;
        println!("Part 4 appendix: shapes clustered and recovered as primitives");
        layers::run(output)?;
        println!("Part 4 appendix: holes and colour vectorized");
        if config.paintings.is_some() {
            paintings::run(output, &images()?, None, None)?;
            println!("Part 4 showcase: paintings vectorized");
        }
    }
    // Record the build environment and source fingerprint for the report.
    let rustc = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "not available on runtime PATH".into());
    let status = match config.command {
        Command::Part(n) => format!("partial\",\"part\":\"{n}"),
        _ => "complete".into(),
    };
    std::fs::write(output.join("run.json"),format!("{{\"status\":\"{status}\",\"package_version\":{},\"source_fingerprint\":{},\"fingerprint_algorithm\":\"FNV-1a-64 build inputs\",\"os\":{},\"architecture\":{},\"rustc\":{},\"canvas\":{size},\"wedges\":{wedges},\"timing_repeats\":3,\"program\":\"single Rust executable for Parts 1-4\"}}\n",json_string(env!("CARGO_PKG_VERSION")),json_string(env!("SOURCE_FINGERPRINT")),json_string(std::env::consts::OS),json_string(std::env::consts::ARCH),json_string(&rustc)))?;
    println!("Complete: {}", output.display());
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}
