use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsKokoroModelConfig,
    OfflineTtsModelConfig, OfflineTtsVitsModelConfig,
};
use std::{
    env,
    error::Error,
    f64::consts::TAU,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    str::FromStr,
    sync::atomic::{AtomicUsize, Ordering},
    thread,
};

#[cfg(all(windows, feature = "cuda"))]
mod runtime;

type Res<T> = Result<T, Box<dyn Error>>;

const BASE: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/";
const MODELS: &[(&str, &str)] = &[
    ("vits-piper-en_US-lessac-medium", "Piper Lessac, US female, 67 MB, fast"),
    ("vits-piper-en_US-amy-low", "Piper Amy, US female, 67 MB, fast"),
    ("vits-piper-en_GB-alba-medium", "Piper Alba, British female, 67 MB, fast"),
    ("kokoro-en-v0_19", "Kokoro, 11 US/UK voices, 320 MB, most natural"),
];
const CARRIER_HZ: f64 = 180.0;

fn var<T: FromStr>(key: &str, default: T) -> T {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// First subfolder of `dir` that looks like a sherpa-onnx TTS model.
fn find_model(dir: &Path) -> Option<PathBuf> {
    let mut found: Vec<_> = fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("tokens.txt").is_file())
        .collect();
    found.sort();
    found.into_iter().next()
}

fn download_model(dir: &Path) -> Res<PathBuf> {
    println!("No TTS model found in {}. Which one should I download?", dir.display());
    for (i, (name, desc)) in MODELS.iter().enumerate() {
        println!("  {}) {desc}  [{name}]", i + 1);
    }
    print!("> ");
    io::stdout().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    let (name, _) = line
        .trim()
        .parse::<usize>()
        .ok()
        .and_then(|i| MODELS.get(i.wrapping_sub(1)))
        .ok_or("invalid choice")?;

    // unpack beside the target, so an aborted download never looks like a model
    let tmp = dir.join(".download");
    let _ = fs::remove_dir_all(&tmp);
    println!("Downloading {name}, this can take a few minutes ...");
    let body = ureq::get(&format!("{BASE}{name}.tar.bz2")).call()?.into_reader();
    tar::Archive::new(bzip2::read::BzDecoder::new(body)).unpack(&tmp)?;
    fs::rename(tmp.join(name), dir.join(name))?;
    let _ = fs::remove_dir_all(&tmp);
    Ok(dir.join(name))
}

fn load_tts(model_dir: &Path, provider: &str) -> Res<OfflineTts> {
    let file = |f: &str| Some(model_dir.join(f).to_string_lossy().into_owned());
    let mut model = OfflineTtsModelConfig {
        provider: Some(provider.into()),
        num_threads: 1, // parallelism comes from one worker per core in main
        ..Default::default()
    };
    if model_dir.join("voices.bin").is_file() {
        model.kokoro = OfflineTtsKokoroModelConfig {
            model: file("model.onnx"),
            voices: file("voices.bin"),
            tokens: file("tokens.txt"),
            data_dir: file("espeak-ng-data"),
            ..Default::default()
        };
    } else {
        let onnx = fs::read_dir(model_dir)?
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|x| x == "onnx"))
            .ok_or("no .onnx file in the model folder")?;
        model.vits = OfflineTtsVitsModelConfig {
            model: Some(onnx.to_string_lossy().into_owned()),
            tokens: file("tokens.txt"),
            data_dir: file("espeak-ng-data"),
            ..Default::default()
        };
    }
    OfflineTts::create(&OfflineTtsConfig { model, ..Default::default() })
        .ok_or_else(|| format!("could not load the model in {}", model_dir.display()).into())
}

/// Beat schedule of a mode: (minute, beat Hz) key points, linear in between.
/// The last minute is the track length.
type Schedule = &'static [(f64, f64)];

const MODES: &[(&str, Schedule)] = &[
    // An idealized 6 h night: four 90-minute cycles. Deep sleep (N3, delta) is long in the
    // first cycles and gone by the last; REM (theta) grows from 5 to 35 minutes.
    (
        "sleep",
        &[
            (0.0, 8.0),   // relaxed, eyes closed (alpha)
            (15.0, 5.0),  // drifting off (theta)
            (30.0, 2.0),  // cycle 1: deep sleep ...
            (70.0, 2.0),  // ... for 40 min
            (80.0, 4.5),  // light sleep
            (85.0, 6.0),  // first, short REM
            (90.0, 6.0),
            (100.0, 4.5), // cycle 2: light sleep
            (115.0, 2.0), // deep sleep, 30 min
            (145.0, 2.0),
            (155.0, 4.5),
            (160.0, 6.0), // REM, 20 min
            (180.0, 6.0),
            (190.0, 4.5), // cycle 3: light sleep
            (205.0, 3.0), // last, shallow deep sleep, 15 min
            (220.0, 3.0),
            (230.0, 4.5),
            (240.0, 6.0), // REM, 30 min
            (270.0, 6.0),
            (280.0, 4.5), // cycle 4: light sleep only
            (310.0, 4.5),
            (315.0, 6.0), // REM, 35 min
            (350.0, 6.0),
            (360.0, 10.0), // rise toward waking (alpha)
        ],
    ),
    ("work", &[(0.0, 15.0), (30.0, 15.0)]), // low beta: focused attention
    ("meditate", &[(0.0, 10.0), (5.0, 6.0), (30.0, 6.0)]), // alpha settling into theta
    ("active", &[(0.0, 20.0), (30.0, 20.0)]), // beta: alert and energetic
];

fn beat_hz(schedule: Schedule, minute: f64) -> f64 {
    let i = schedule.partition_point(|p| p.0 <= minute).clamp(1, schedule.len() - 1);
    let ((m0, hz0), (m1, hz1)) = (schedule[i - 1], schedule[i]);
    hz0 + (hz1 - hz0) * ((minute - m0) / (m1 - m0)).clamp(0.0, 1.0)
}

/// One loop of the voice: all affirmations in order with a short pause, stacked `layers`
/// times at evenly spaced offsets. Dividing by sqrt(layers) keeps the overall level the same.
fn voice_bed(clips: &[Vec<f32>], sr: usize, layers: usize) -> Vec<f32> {
    let mut sequence = Vec::new();
    for clip in clips {
        sequence.extend_from_slice(clip);
        sequence.resize(sequence.len() + sr / 2, 0.0);
    }
    let n = sequence.len();
    (0..n)
        .map(|i| {
            let stacked: f32 = (0..layers).map(|k| sequence[(i + k * n / layers) % n]).sum();
            stacked / (layers as f32).sqrt()
        })
        .collect()
}

/// Mono track, sample by sample: the looping quiet voice bed under a monaural beat
/// (two tones summed in one channel) that follows the schedule.
fn track(
    bed: &[f32],
    sr: usize,
    schedule: Schedule,
    voice_db: f32,
) -> impl Iterator<Item = f32> + '_ {
    let n = (schedule[schedule.len() - 1].0 * 60.0 * sr as f64) as usize;
    let fade = 5 * sr;
    let gain = 10f32.powf(voice_db / 20.0);
    let step = TAU / sr as f64;
    let (mut a, mut b) = (0.0f64, 0.0f64);
    (0..n).map(move |i| {
        let envelope = (i.min(n - i) as f32 / fade as f32).min(1.0);
        let minute = i as f64 / (60 * sr) as f64;
        a = (a + step * CARRIER_HZ) % TAU;
        b = (b + step * (CARRIER_HZ + beat_hz(schedule, minute))) % TAU;
        let voice = bed.get(i % bed.len().max(1)).map_or(0.0, |s| s * gain);
        (voice + 0.25 * (a.sin() + b.sin()) as f32) * envelope
    })
}

fn main() -> Res<()> {
    dotenvy::dotenv().ok();
    let args: Vec<String> = env::args().collect();
    let input = args.get(1).ok_or("usage: tts_subliminal <affirmations.txt> [out.wav]")?;
    let output = args.get(2).map_or("subliminal.wav", String::as_str);

    let text = fs::read_to_string(input)?;
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    if lines.is_empty() {
        return Err(format!("{input} contains no affirmations").into());
    }

    let mode = var("MODE", "sleep".to_string()).trim().to_lowercase();
    let schedule = MODES
        .iter()
        .find(|(name, _)| *name == mode)
        .ok_or_else(|| format!("unknown MODE `{mode}`, use sleep, work, meditate or active"))?
        .1;
    let speed: f32 = var("SPEED", 2.0);
    let layers: usize = var("LAYERS", 3).max(1);
    println!(
        "Mode: {mode} ({} min), voice speed {speed}x, layers: {layers}",
        schedule[schedule.len() - 1].0
    );

    let model_path = PathBuf::from(var("MODEL_PATH", "models".to_string()));
    fs::create_dir_all(&model_path)?;
    let model_dir = match find_model(&model_path) {
        Some(dir) => dir,
        None => download_model(&model_path)?,
    };
    println!("Model: {}", model_dir.display());
    // a CUDA build uses the GPU unless .env says otherwise
    let default_provider = if cfg!(feature = "cuda") { "cuda" } else { "cpu" };
    let provider = var("PROVIDER", default_provider.to_string()).trim().to_lowercase();
    // Otherwise sherpa-onnx silently falls back to the CPU with a misleading hint.
    if provider == "cuda" && !cfg!(feature = "cuda") {
        return Err("PROVIDER=cuda, but this is a CPU build. \
                    Rebuild with `cargo build --release --features cuda`."
            .into());
    }
    #[cfg(all(windows, feature = "cuda"))]
    runtime::load(provider == "cuda")?;
    let tts = load_tts(&model_dir, &provider)?;

    let config = GenerationConfig { sid: var("SPEAKER", 0), speed, ..Default::default() };
    // One shared model; each worker pulls the next line. CPU: one worker per core.
    // GPU: the model sits on the card once and a few workers feed it at the same time.
    // shortcut: the GPU default of 4 is a guess, measure with WORKERS and adjust it
    let default_workers =
        if provider == "cpu" { thread::available_parallelism().map_or(1, |n| n.get()) } else { 4 };
    let workers = var("WORKERS", default_workers).clamp(1, lines.len().max(1));
    println!("Provider: {provider}, worker threads: {workers}");
    let (next, done) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let finished: Option<Vec<Vec<(usize, Vec<f32>)>>> = thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(line) = lines.get(i) else { break };
                        let audio = tts.generate_with_config(
                            line,
                            &config,
                            None::<fn(&[f32], f32) -> bool>,
                        )?;
                        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                        println!("[{n}/{}] {line}", lines.len());
                        mine.push((i, audio.samples().to_vec()));
                    }
                    Some(mine)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("worker panicked")).collect()
    });
    let mut clips: Vec<_> = finished.ok_or("speech generation failed")?.into_iter().flatten().collect();
    clips.sort_by_key(|(position, _)| *position); // back into file order
    let clips: Vec<Vec<f32>> = clips.into_iter().map(|(_, clip)| clip).collect();

    let sr = tts.sample_rate() as u32;
    let bed = voice_bed(&clips, sr as usize, layers);
    println!("Voice loop: {:.1} s", bed.len() as f32 / sr as f32);
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut wav = hound::WavWriter::create(output, spec)?;
    for s in track(&bed, sr as usize, schedule, var("VOICE_DB", -30.0)) {
        wav.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    wav.finalize()?;
    println!("Wrote {output}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule(mode: &str) -> Schedule {
        MODES.iter().find(|m| m.0 == mode).unwrap().1
    }

    #[test]
    fn schedules_are_ordered_and_have_the_promised_length() {
        for (name, points) in MODES {
            assert!(points.windows(2).all(|w| w[0].0 < w[1].0), "{name}: minutes must increase");
            assert!(points.iter().all(|p| (0.5..=40.0).contains(&p.1)), "{name}: beat range");
            let minutes = if *name == "sleep" { 360.0 } else { 30.0 };
            assert_eq!(points[points.len() - 1].0, minutes, "{name}: length");
        }
    }

    #[test]
    fn sleep_beat_follows_the_cycles() {
        let hz = |minute| beat_hz(schedule("sleep"), minute);
        assert_eq!(hz(0.0), 8.0);
        assert_eq!(hz(50.0), 2.0); // deep sleep in cycle 1
        assert_eq!(hz(22.5), 3.5); // halfway between two key points
        assert_eq!(hz(170.0), 6.0); // REM in cycle 2
        assert_eq!(hz(300.0), 4.5); // no deep sleep in cycle 4
        assert_eq!(hz(360.0), 10.0);
        assert_eq!(hz(999.0), 10.0); // past the end: stays at the last value
    }

    #[test]
    fn voice_bed_stacks_layers_at_constant_level() {
        let sr = 1000;
        let clips = [vec![1.0; sr], vec![0.5; sr]];
        let single = voice_bed(&clips, sr, 1);
        assert_eq!(single.len(), 3 * sr); // two clips, each followed by a 0.5 s pause
        assert_eq!((single[0], single[sr + 10], single[sr + 600]), (0.0 + 1.0, 0.0, 0.5));

        let double = voice_bed(&clips, sr, 2);
        assert_eq!(double.len(), single.len());
        // second layer is shifted by half the loop: clip 1 overlaps the second clip
        assert_eq!(double[100], (1.0 + 0.5) / 2f32.sqrt());
        let power = |bed: &[f32]| bed.iter().map(|s| s * s).sum::<f32>();
        assert!(power(&double) >= power(&single)); // never quieter than one layer
    }

    #[test]
    fn track_has_the_mode_length_fades_in_and_never_clips() {
        let sr = 100;
        let bed = voice_bed(&[vec![1.0; sr]], sr, 3);
        let samples: Vec<f32> = track(&bed, sr, schedule("work"), -30.0).collect();
        assert_eq!(samples.len(), 30 * 60 * sr);
        assert_eq!(samples[0], 0.0);
        assert!(samples.iter().all(|s| s.abs() <= 0.6));
        // the voice is in there: differs from the beat alone
        let beat_only: Vec<f32> = track(&[], sr, schedule("work"), -30.0).collect();
        assert_ne!(samples[10 * sr + 5], beat_only[10 * sr + 5]);
    }
}
