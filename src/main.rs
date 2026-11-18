use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsKokoroModelConfig,
    OfflineTtsModelConfig, OfflineTtsVitsModelConfig,
};
use std::{
    env,
    error::Error,
    f32::consts::TAU,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    str::FromStr,
};

type Res<T> = Result<T, Box<dyn Error>>;

const BASE: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/";
const MODELS: &[(&str, &str)] = &[
    ("vits-piper-en_US-lessac-medium", "Piper Lessac, US female, 67 MB, fast"),
    ("vits-piper-en_US-amy-low", "Piper Amy, US female, 67 MB, fast"),
    ("vits-piper-en_GB-alba-medium", "Piper Alba, British female, 67 MB, fast"),
    ("kokoro-en-v0_19", "Kokoro, 11 US/UK voices, 320 MB, most natural"),
];
const CARRIER_HZ: f32 = 180.0;

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
    let mut model = OfflineTtsModelConfig { provider: Some(provider.into()), ..Default::default() };
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

/// Beat frequency over the track (0..1): theta 7 Hz sinks to delta 2 Hz (deep sleep)
/// by the middle, then rises to theta 5 Hz (dreaming).
fn beat_hz(progress: f32) -> f32 {
    if progress < 0.5 {
        7.0 - 10.0 * progress
    } else {
        2.0 + 6.0 * (progress - 0.5)
    }
}

/// Mono track: looping quiet voice under a monaural beat (two tones summed in one channel).
fn mix(clips: &[Vec<f32>], sr: usize, minutes: f32, voice_db: f32) -> Vec<f32> {
    let n = (minutes * 60.0 * sr as f32) as usize;
    let fade = 5 * sr;
    let gain = 10f32.powf(voice_db / 20.0);
    let mut out = vec![0.0f32; n];

    let mut pos = fade;
    'fill: while !clips.is_empty() {
        for clip in clips {
            if pos + clip.len() + fade > n {
                break 'fill;
            }
            for (o, s) in out[pos..].iter_mut().zip(clip) {
                *o = s * gain;
            }
            pos += clip.len() + 2 * sr; // 2 s pause between affirmations
        }
    }

    let (mut a, mut b) = (0.0f32, 0.0f32);
    for (i, o) in out.iter_mut().enumerate() {
        let envelope = (i.min(n - i) as f32 / fade as f32).min(1.0);
        a = (a + TAU * CARRIER_HZ / sr as f32) % TAU;
        b = (b + TAU * (CARRIER_HZ + beat_hz(i as f32 / n as f32)) / sr as f32) % TAU;
        *o = (*o + 0.25 * (a.sin() + b.sin())) * envelope;
    }
    out
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

    let model_path = PathBuf::from(var("MODEL_PATH", "models".to_string()));
    fs::create_dir_all(&model_path)?;
    let model_dir = match find_model(&model_path) {
        Some(dir) => dir,
        None => download_model(&model_path)?,
    };
    println!("Model: {}", model_dir.display());
    let tts = load_tts(&model_dir, &var("PROVIDER", "cpu".to_string()))?;

    let config = GenerationConfig { sid: var("SPEAKER", 0), ..Default::default() };
    let mut clips = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        println!("[{}/{}] {line}", i + 1, lines.len());
        let audio = tts
            .generate_with_config(line, &config, None::<fn(&[f32], f32) -> bool>)
            .ok_or("speech generation failed")?;
        clips.push(audio.samples().to_vec());
    }

    let sr = tts.sample_rate() as u32;
    let track = mix(&clips, sr as usize, var("MINUTES", 30.0), var("VOICE_DB", -30.0));
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut wav = hound::WavWriter::create(output, spec)?;
    for s in track {
        wav.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    wav.finalize()?;
    println!("Wrote {output}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beat_stays_in_sleep_range() {
        assert_eq!(beat_hz(0.0), 7.0);
        assert_eq!(beat_hz(0.5), 2.0);
        assert_eq!(beat_hz(1.0), 5.0);
    }

    #[test]
    fn mix_loops_voice_and_never_clips() {
        let sr = 1000;
        let track = mix(&[vec![1.0; sr]], sr, 1.0, -30.0);
        assert_eq!(track.len(), 60 * sr);
        assert!(track.iter().all(|s| s.abs() <= 0.6));
        assert_eq!(track[0], 0.0); // fades in from silence
        // voice repeats: a second copy sits 3 s (1 s clip + 2 s pause) after the first
        let voiced = |at: usize| mix(&[vec![1.0; sr]], sr, 1.0, -30.0)[at] != mix(&[], sr, 1.0, -30.0)[at];
        assert!(voiced(5 * sr + 10) && voiced(8 * sr + 10) && !voiced(6 * sr + 500));
    }
}
