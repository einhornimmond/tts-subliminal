// Feature "cuda" (Windows only): find the installed CUDA major version, fetch the matching
// sherpa-onnx CUDA runtime and hand its DLLs to src/runtime.rs, which embeds them in the
// executable. The program itself is linked against the regular shared (CPU) import library
// that sherpa-onnx-sys downloads; which DLLs are loaded at run time decides about CUDA.
//
// Feature "cuda-source" additionally compiles ONNX Runtime from source against the installed
// CUDA and cuDNN and embeds that build instead of the downloaded one.
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

// must match the sherpa-onnx version pinned in Cargo.toml
const VERSION: &str = "v1.13.8";
// the ONNX Runtime version that sherpa-onnx release was built against
const ORT_VERSION: &str = "1.28.2";
// Commit of the tag v1.28.2 in microsoft/onnxruntime, as recorded by the Software Heritage
// archive. The hash pins the content, so any repository that serves the commit will do.
// The official one was not reachable when this was written; Intel's fork shares its history.
const ORT_COMMIT: &str = "33ca9628233dc8f002435e868d4c2e9f82766ca1";
const ORT_REPOS: [&str; 2] =
    ["https://github.com/microsoft/onnxruntime", "https://github.com/intel/onnxruntime"];
// prebuilt CUDA runtimes that exist for this sherpa-onnx version, newest first
const CUDA_MAJORS: [u32; 2] = [13, 12];
// must match the list in src/runtime.rs
const NEEDED: [&str; 4] = [
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
    "sherpa-onnx-c-api.dll",
    "onnxruntime_providers_cuda.dll",
];

/// Where the NVIDIA runtime DLLs can live: the toolkit folder first, then the PATH.
fn nvidia_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(root) = env::var_os("CUDA_PATH") {
        let bin = PathBuf::from(root).join("bin");
        dirs.push(bin.join("x64")); // layout of CUDA 13
        dirs.push(bin);
    }
    dirs.extend(env::split_paths(&env::var_os("PATH").unwrap_or_default()));
    dirs
}

/// Newest entry of `dir` that `keep` accepts, by name.
fn newest_entry(dir: &Path, keep: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    let mut entries: Vec<PathBuf> =
        fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).filter(|p| keep(p)).collect();
    entries.sort();
    entries.pop()
}

/// A cuDNN folder in the flat layout the ONNX Runtime build finds on its own:
/// include\cudnn.h and lib\x64\cudnn.lib.
fn cudnn_home(cuda_home: &Path, major: u32, target_dir: &Path) -> PathBuf {
    let has_header = |dir: &Path| dir.join("cudnn.h").is_file();
    let home = env::var_os("CUDNN_HOME")
        .map(PathBuf::from)
        // cuDNN copied into the toolkit folder
        .or_else(|| has_header(&cuda_home.join("include")).then(|| cuda_home.to_path_buf()))
        // default folder of the cuDNN 9 installer
        .or_else(|| {
            let root = PathBuf::from(env::var_os("ProgramFiles")?).join("NVIDIA").join("CUDNN");
            newest_entry(&root, |p| p.is_dir())
        })
        .expect("cuDNN not found: set CUDNN_HOME to the folder that contains bin, include and lib");
    if has_header(&home.join("include")) {
        return home;
    }
    // The installer sorts its files by CUDA version: include\12.x\cudnn.h, lib\12.x\x64\*.lib.
    // The build would only look there if told that exact version, so make a flat copy.
    let prefix = format!("{major}.");
    let versioned = newest_entry(&home.join("include"), |p| {
        has_header(p) && p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(&prefix))
    })
    .unwrap_or_else(|| panic!("no cudnn.h for CUDA {major} below {}", home.display()));
    let version = versioned.file_name().unwrap();
    let flat = target_dir.join("ort-cudnn");
    for (from, to) in [
        (versioned.clone(), flat.join("include")),
        (home.join("lib").join(version).join("x64"), flat.join("lib").join("x64")),
    ] {
        fs::create_dir_all(&to).unwrap();
        for file in fs::read_dir(&from).unwrap_or_else(|e| panic!("{}: {e}", from.display())) {
            let file = file.unwrap().path();
            if file.is_file() {
                fs::copy(&file, to.join(file.file_name().unwrap())).unwrap();
            }
        }
    }
    flat
}

/// Compute capabilities of the installed NVIDIA cards as CMake wants them: "8.6" -> "86",
/// several cards joined by ";". None if nvidia-smi is missing or too old to tell.
fn gpu_architectures() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|output| output.status.success())?;
    parse_architectures(&String::from_utf8_lossy(&output.stdout))
}

fn parse_architectures(smi_output: &str) -> Option<String> {
    let mut archs: Vec<String> = smi_output.lines().map(|line| line.trim().replace('.', "")).collect();
    archs.retain(|arch| !arch.is_empty());
    archs.sort();
    archs.dedup();
    let valid = !archs.is_empty() && archs.iter().all(|arch| arch.parse::<u32>().is_ok());
    valid.then(|| archs.join(";"))
}

/// Compiles ONNX Runtime against the installed CUDA and cuDNN (takes a long time) and
/// returns the folder with the resulting DLLs.
fn build_onnxruntime(target_dir: &Path, major: u32) -> PathBuf {
    let cuda_home = PathBuf::from(
        env::var_os("CUDA_PATH").expect("cuda-source needs CUDA_PATH (the CUDA Toolkit folder)"),
    );
    let cudnn_home = cudnn_home(&cuda_home, major, target_dir);
    println!("cargo:warning=building ONNX Runtime with cuDNN from {}", cudnn_home.display());

    // short names: the build produces deep paths and Windows limits their length
    let (source, build) = (target_dir.join("ort-src"), target_dir.join("ort-build"));
    let git = || {
        let mut git = Command::new("git");
        // fail instead of asking for a login; allow the long paths in this repository
        git.env("GIT_TERMINAL_PROMPT", "0").args(["-c", "core.longpaths=true", "-C"]).arg(&source);
        git
    };
    let run = |command: &mut Command, what: &str| {
        let status = command.status().unwrap_or_else(|e| panic!("cannot start {what}: {e}"));
        assert!(status.success(), "{what} failed ({status}); run cargo with -vv for its output");
    };
    if !source.join("build.bat").is_file() {
        let _ = fs::remove_dir_all(&source);
        fs::create_dir_all(&source).unwrap();
        run(git().args(["init", "-q"]), "git init");
        let fetched = ORT_REPOS.iter().any(|repo| {
            git()
                .args(["fetch", "--depth", "1", repo, ORT_COMMIT])
                .status()
                .is_ok_and(|status| status.success())
        });
        assert!(fetched, "could not fetch ONNX Runtime {ORT_COMMIT} from any of {ORT_REPOS:?}");
        run(git().args(["checkout", "-q", "FETCH_HEAD"]), "git checkout");
    }

    let mut build_bat = Command::new(source.join("build.bat"));
    build_bat
        .args(["--config", "Release", "--build_shared_lib", "--skip_tests"])
        .args(["--compile_no_warning_as_error", "--use_cuda"])
        // A plain --parallel starts one compiler per logical core, each CUDA one with 4 nvcc
        // threads on top; on 32 cores that ate more than 64 GB of RAM and crashed Windows.
        // shortcut: fixed job count, make it depend on the installed RAM if 8 is still too much
        .args(["--parallel", "8", "--nvcc_threads", "1"])
        .arg("--cuda_home")
        .arg(&cuda_home)
        .arg("--cudnn_home")
        .arg(&cudnn_home)
        .arg("--build_dir")
        .arg(&build)
        .current_dir(&source);
    // The toolkit folder is named after its version (v12.4). Passing it on makes Visual
    // Studio use this toolkit even when a newer one is installed as well.
    let version = cuda_home.file_name().unwrap_or_default().to_string_lossy();
    if let Some(version) = version.strip_prefix('v') {
        if version.split('.').all(|part| part.parse::<u32>().is_ok()) {
            build_bat.args(["--cuda_version", version]);
        }
    }
    // Compile the GPU kernels only for the installed cards instead of for every generation
    // (seven to nine of them). The resulting DLL then runs on these cards only.
    match gpu_architectures() {
        Some(archs) => {
            println!("cargo:warning=compiling the CUDA kernels for GPU architecture {archs} only");
            build_bat.arg("--cmake_extra_defines").arg(format!("CMAKE_CUDA_ARCHITECTURES={archs}"));
        }
        None => println!(
            "cargo:warning=nvidia-smi did not report the GPU architecture; compiling for all"
        ),
    }
    run(&mut build_bat, "the ONNX Runtime build");
    build.join("Release").join("Release")
}

fn main() {
    // self-check of the nvidia-smi parsing; build scripts have no test runner
    assert_eq!(parse_architectures("8.6\r\n").as_deref(), Some("86"));
    assert_eq!(parse_architectures("12.0\n8.6\n8.6\n").as_deref(), Some("120;86"));
    assert_eq!(parse_architectures("[N/A]\n"), None);
    assert_eq!(parse_architectures(""), None);

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDNN_HOME");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    assert_eq!(
        env::var("CARGO_CFG_TARGET_OS").unwrap(),
        "windows",
        "the `cuda` feature supports Windows only"
    );

    let dirs = nvidia_dirs();
    // the first folder holding a CUDA runtime wins, as it would when Windows loads the DLL
    let (major, found_in) = dirs
        .iter()
        .find_map(|dir| {
            let major = CUDA_MAJORS
                .into_iter()
                .find(|major| dir.join(format!("cudart64_{major}.dll")).is_file())?;
            Some((major, dir))
        })
        .expect(
            "no CUDA runtime found: neither cudart64_13.dll nor cudart64_12.dll is in \
             %CUDA_PATH%\\bin or on the PATH. Install the CUDA Toolkit 12 or 13 and cuDNN 9.",
        );
    println!("cargo:warning=using CUDA {major} found in {}", found_in.display());
    if !dirs.iter().any(|dir| dir.join("cudnn64_9.dll").is_file()) {
        println!("cargo:warning=cudnn64_9.dll is not on the PATH; the GPU will not work without cuDNN 9");
    }

    let archive =
        format!("sherpa-onnx-{VERSION}-cuda-{major}.x-cudnn-9.x-onnxruntime{ORT_VERSION}-win-x64-cuda");
    // OUT_DIR is target/<profile>/build/<package>-<hash>/out; the download is kept in
    // target/ so debug and release builds share it
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let target_dir = out_dir.ancestors().nth(4).unwrap();
    let cache = target_dir.join(&archive);

    if !cache.is_dir() {
        // unpack beside the cache, so an aborted download never looks complete
        let part = cache.with_file_name(format!("{archive}.part"));
        let _ = fs::remove_dir_all(&part);
        fs::create_dir_all(&part).unwrap();
        let url = format!(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/{VERSION}/{archive}.tar.bz2"
        );
        let body = ureq::get(&url)
            .call()
            .unwrap_or_else(|e| panic!("downloading {url} failed: {e}"))
            .into_reader();
        let mut tar = tar::Archive::new(bzip2::read::BzDecoder::new(body));
        for entry in tar.entries().unwrap() {
            let mut entry = entry.unwrap();
            let path = entry.path().unwrap().into_owned();
            let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
            let in_lib = path.parent().is_some_and(|dir| dir.ends_with("lib"));
            if in_lib && NEEDED.contains(&name.as_str()) {
                entry.unpack(part.join(name)).unwrap();
            }
        }
        for name in NEEDED {
            assert!(part.join(name).is_file(), "{name} is missing in {url}");
        }
        fs::rename(&part, &cache).unwrap();
    }

    let mut runtime = cache.clone();
    if env::var_os("CARGO_FEATURE_CUDA_SOURCE").is_some() {
        // sherpa-onnx from the download, ONNX Runtime from our own build; delete this
        // folder to force a rebuild
        runtime = target_dir.join("cuda-runtime-source");
        if !NEEDED.iter().all(|name| runtime.join(name).is_file()) {
            let built = build_onnxruntime(target_dir, major);
            fs::create_dir_all(&runtime).unwrap();
            for name in NEEDED {
                let from = if name.starts_with("onnxruntime") { &built } else { &cache };
                fs::copy(from.join(name), runtime.join(name)).unwrap_or_else(|e| {
                    panic!("cannot copy {name} from {}: {e}", from.display())
                });
            }
        }
        println!("cargo:warning=embedding the self-built ONNX Runtime {ORT_VERSION}");
    }

    println!("cargo:rustc-env=CUDA_RUNTIME_DIR={}", runtime.display());
    println!("cargo:rustc-env=CUDA_MAJOR={major}");
    // Load sherpa-onnx on first use instead of at program start, so src/runtime.rs
    // gets the chance to unpack the embedded DLLs first.
    println!("cargo:rustc-link-arg=/DELAYLOAD:sherpa-onnx-c-api.dll");
    println!("cargo:rustc-link-arg=delayimp.lib");
}
