use std::{fs, path::PathBuf};

fn update(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash = (*hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
}
fn main() {
    let mut files = vec![
        PathBuf::from("Cargo.toml"),
        PathBuf::from("Cargo.lock"),
        PathBuf::from("build.rs"),
    ];
    for entry in fs::read_dir("src").expect("read source directory") {
        let path = entry.expect("read source entry").path();
        if path.extension().is_some_and(|ext| ext == "rs") {
            files.push(path);
        }
    }
    files.sort();
    let mut hash = 0xcbf29ce484222325u64;
    for path in files {
        let name = path.to_string_lossy().replace('\\', "/");
        let bytes = fs::read(&path).expect("read build fingerprint input");
        update(&mut hash, name.as_bytes());
        update(&mut hash, b"\0");
        update(&mut hash, &bytes);
        update(&mut hash, b"\0");
        println!("cargo:rerun-if-changed={name}");
    }
    println!("cargo:rerun-if-changed=src");
    // Identity only, not a cryptographic integrity claim. The audit adds SHA-256.
    println!("cargo:rustc-env=SOURCE_FINGERPRINT={hash:016x}");
}
