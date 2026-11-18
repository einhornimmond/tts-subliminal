// Feature "cuda": fetch the sherpa-onnx CUDA runtime and put its DLLs next to the executable.
// The program itself is linked against the regular shared (CPU) libraries that sherpa-onnx-sys
// downloads; only the DLLs loaded at run time decide whether CUDA is available.
use std::{env, fs, path::PathBuf};

// must match the sherpa-onnx version pinned in Cargo.toml
const VERSION: &str = "v1.13.8";
const ARCHIVE: &str = "sherpa-onnx-v1.13.8-cuda-12.x-cudnn-9.x-onnxruntime1.28.2-win-x64-cuda";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    assert_eq!(
        env::var("CARGO_CFG_TARGET_OS").unwrap(),
        "windows",
        "the `cuda` feature supports Windows only"
    );

    // OUT_DIR is target/<profile>/build/<package>-<hash>/out
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let profile_dir = out_dir.ancestors().nth(3).unwrap();
    // kept in target/ so debug and release share one 500 MB download
    let cache = profile_dir.parent().unwrap().join(ARCHIVE);

    if !cache.is_dir() {
        // unpack beside the cache, so an aborted download never looks complete
        let part = cache.with_file_name(format!("{ARCHIVE}.part"));
        let _ = fs::remove_dir_all(&part);
        fs::create_dir_all(&part).unwrap();
        let url = format!(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/{VERSION}/{ARCHIVE}.tar.bz2"
        );
        let body = ureq::get(&url)
            .call()
            .unwrap_or_else(|e| panic!("downloading {url} failed: {e}"))
            .into_reader();
        let mut archive = tar::Archive::new(bzip2::read::BzDecoder::new(body));
        for entry in archive.entries().unwrap() {
            let mut entry = entry.unwrap();
            let path = entry.path().unwrap().into_owned();
            let is_runtime_dll = path.parent().is_some_and(|dir| dir.ends_with("lib"))
                && path.extension().is_some_and(|ext| ext == "dll");
            if is_runtime_dll {
                entry.unpack(part.join(path.file_name().unwrap())).unwrap();
            }
        }
        fs::rename(&part, &cache).unwrap();
    }

    // Overwrites the CPU DLLs that sherpa-onnx-sys copied here. Its build script runs first
    // because sherpa-onnx-sys is a direct dependency with a `links` key.
    for dll in fs::read_dir(&cache).unwrap() {
        let dll = dll.unwrap().path();
        fs::copy(&dll, profile_dir.join(dll.file_name().unwrap())).unwrap();
    }
}
