use std::{
    fs,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn one_command_produces_traceable_experiments_and_rejects_invalid_arguments() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let output = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("cli-{stamp}"));
    let result = Command::new(env!("CARGO_BIN_EXE_multimedia"))
        .arg(&output)
        .args(["--size", "128", "--wedges", "96"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    for file in [
        "diagnostic.bmp",
        "curve_summary.json",
        "joins_metrics.csv",
        "star.bmp",
        "ring_opposite.bmp",
        "star_k8.bmp",
        "row_samples.csv",
        "row_selection.json",
        "wheel_k1.bmp",
        "wheel_k8.bmp",
        "transform_bitmap.bmp",
        "transform_summary.json",
        "vectorize_metrics.csv",
        "vectorize_epicycles_m8.bmp",
        "recover_summary.json",
        "recover_vector.svg",
        "holes_vector.svg",
        "color_channels.svg",
        "layers_summary.json",
        "run.json",
    ] {
        assert!(output.join(file).is_file(), "missing {file}");
    }
    let summary = fs::read_to_string(output.join("curve_summary.json")).unwrap();
    assert!(summary.contains("\"canvas\":128"));
    let row = fs::read_to_string(output.join("row_samples.csv")).unwrap();
    assert!(row.lines().count() > 5);
    assert!(!Command::new(env!("CARGO_BIN_EXE_multimedia"))
        .args(["--size", "0"])
        .status()
        .unwrap()
        .success());
    assert!(!Command::new(env!("CARGO_BIN_EXE_multimedia"))
        .args(["--unknown"])
        .status()
        .unwrap()
        .success());
}

#[test]
fn subcommands_run_one_part_and_check_their_arguments() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let output = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("cli-part2-{stamp}"));
    let result = Command::new(env!("CARGO_BIN_EXE_multimedia"))
        .args(["part2", "--size", "128", "--out"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.join("star.bmp").is_file());
    assert!(!output.join("diagnostic.bmp").exists(), "part2 ran part 1");
    let run = fs::read_to_string(output.join("run.json")).unwrap();
    assert!(run.contains("\"status\":\"partial\",\"part\":\"2\""), "{run}");
    for args in [
        &["paint"][..],
        &["hifi", "x.bmp", "--fit", "best"],
        &["part1", "--factor", "2"],
        &["paint", "x.bmp", "--size", "256"],
        &["hifi", "x.bmp", "--levels", "1,2,3"],
        &["hifi", "x.bmp", "--export", "svg"],
        &["render"],
        &["render", "x.csv", "--scale", "0"],
        &["render", "x.csv", "--factor", "2"],
        &["paint", "x.bmp", "--svg"],
    ] {
        assert!(
            !Command::new(env!("CARGO_BIN_EXE_multimedia"))
                .args(args)
                .status()
                .unwrap()
                .success(),
            "{args:?} should be rejected"
        );
    }
}
