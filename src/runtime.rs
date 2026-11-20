//! CUDA build on Windows: the sherpa-onnx and ONNX Runtime DLLs travel inside the executable.
//! They are unpacked into a folder beside it and loaded from there by full path, before
//! the first sherpa-onnx call (the DLL is delay-loaded, see build.rs). Only the NVIDIA
//! libraries (CUDA, cuDNN) have to be installed on the machine.
use std::{
    env, error::Error, ffi::c_void, ffi::OsStr, fs, io, os::windows::ffi::OsStrExt, path::PathBuf,
    ptr,
};

macro_rules! embedded {
    ($name:literal) => {
        ($name, include_bytes!(concat!(env!("CUDA_RUNTIME_DIR"), "/", $name)).as_slice())
    };
}

// Load order: a DLL comes after the ones it imports. Must match the list in build.rs.
const DLLS: [(&str, &[u8]); 4] = [
    embedded!("onnxruntime.dll"),
    embedded!("onnxruntime_providers_shared.dll"),
    embedded!("sherpa-onnx-c-api.dll"),
    embedded!("onnxruntime_providers_cuda.dll"),
];
const GPU_ONLY: &str = "onnxruntime_providers_cuda.dll";

const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x8; // resolve imports from the DLL's own folder

extern "system" {
    fn LoadLibraryExW(name: *const u16, reserved: *mut c_void, flags: u32) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
}

fn load_library(name: &OsStr, flags: u32) -> *mut c_void {
    let wide: Vec<u16> = name.encode_wide().chain([0]).collect();
    // SAFETY: `wide` is a NUL-terminated UTF-16 string that outlives the call.
    unsafe { LoadLibraryExW(wide.as_ptr(), ptr::null_mut(), flags) }
}

/// Why the CUDA provider did not load: NVIDIA DLLs missing, or present in the wrong version.
fn nvidia_problem(cause: io::Error) -> String {
    let major = env!("CUDA_MAJOR");
    // the provider imports these directly; cuDNN is only loaded later, on first use
    let missing: Vec<String> = ["cudart", "cublas", "cublasLt"]
        .iter()
        .map(|lib| format!("{lib}64_{major}.dll"))
        .filter(|dll| load_library(dll.as_ref(), 0).is_null())
        .collect();
    if missing.is_empty() {
        format!(
            "cannot load the CUDA provider: {cause}\n\
             The CUDA {major} libraries were found, but in a version that does not fit this \
             build. Run `where cudart64_{major}.dll` to see which toolkit is used; the first \
             hit must be the toolkit this program was built for. Or set PROVIDER=cpu."
        )
    } else {
        format!(
            "cannot load the CUDA provider: {cause}\n\
             Not found: {}. Add the folder of the CUDA {major} toolkit that holds them \
             (bin or bin\\x64) to the PATH and open a new terminal. Or set PROVIDER=cpu.",
            missing.join(", ")
        )
    }
}

pub fn load(use_gpu: bool) -> Result<(), Box<dyn Error>> {
    // The toolkit's DLL folder is not always on the PATH (CUDA 13 moved it to bin\x64).
    if let Some(root) = env::var_os("CUDA_PATH") {
        let bin = PathBuf::from(root).join("bin");
        let path = env::var_os("PATH").unwrap_or_default();
        let dirs = [bin.join("x64"), bin].into_iter().chain(env::split_paths(&path));
        env::set_var("PATH", env::join_paths(dirs)?);
    }
    let dir = env::current_exe()?.with_file_name("tts_subliminal_runtime");
    fs::create_dir_all(&dir)?;
    for (name, bytes) in DLLS {
        let path = dir.join(name);
        // an interrupted or outdated copy has a different size and is written again
        if fs::metadata(&path).map_or(true, |meta| meta.len() != bytes.len() as u64) {
            println!("Unpacking {name} ...");
            fs::write(&path, bytes)?;
        }
        if name == GPU_ONLY && !use_gpu {
            continue;
        }
        let module = load_library(path.as_os_str(), LOAD_WITH_ALTERED_SEARCH_PATH);
        if module.is_null() {
            let cause = io::Error::last_os_error();
            return Err(if name == GPU_ONLY {
                nvidia_problem(cause)
            } else {
                format!("cannot load {}: {cause}", path.display())
            }
            .into());
        }
        if name == GPU_ONLY {
            // This was only the check that gives the readable error above;
            // ONNX Runtime loads the provider itself.
            // SAFETY: `module` is the valid handle returned just above.
            unsafe { FreeLibrary(module) };
        }
    }
    Ok(())
}
