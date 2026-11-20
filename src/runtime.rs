//! CUDA build on Windows: the sherpa-onnx and ONNX Runtime DLLs travel inside the executable.
//! They are unpacked into a folder beside it and loaded from there by full path, before
//! the first sherpa-onnx call (the DLL is delay-loaded, see build.rs). Only the NVIDIA
//! libraries (CUDA, cuDNN) have to be installed on the machine.
use std::{env, error::Error, ffi::c_void, fs, io, os::windows::ffi::OsStrExt, ptr};

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

pub fn load(use_gpu: bool) -> Result<(), Box<dyn Error>> {
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
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
        // SAFETY: `wide` is a NUL-terminated UTF-16 path that outlives the call.
        let module =
            unsafe { LoadLibraryExW(wide.as_ptr(), ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH) };
        if module.is_null() {
            let cause = io::Error::last_os_error();
            return Err(if name == GPU_ONLY {
                format!(
                    "cannot load the CUDA provider: {cause}\n\
                     The installed NVIDIA libraries do not fit this build. It needs a recent \
                     CUDA {major}.x toolkit and cuDNN 9, both on the PATH. Update them, \
                     or set PROVIDER=cpu.",
                    major = env!("CUDA_MAJOR")
                )
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
