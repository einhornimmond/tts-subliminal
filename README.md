# tts_subliminal

Turns a list of English affirmations into a subliminal audio track: a locally
generated TTS voice, sped up and layered, plays very quietly under monaural
beats. Four modes set the beats for sleeping, working, meditating or active
tasks.

Everything runs offline on your own machine. Only the speech model is
downloaded once on first start.

> **Note:** The code in this repository was written entirely by an AI
> (Claude, Anthropic). It has been tested on Linux only; review it before you
> rely on it.

## Build

Requires Rust 1.88 or newer.

```
cargo build --release
```

The result is a single executable at `target/release/tts_subliminal`
(Windows: `tts_subliminal.exe`). The build fetches the matching sherpa-onnx
library automatically and links it statically.

## Usage

```
tts_subliminal <affirmations.txt> [output.wav]
```

- `affirmations.txt`: one affirmation per line, in English. Empty lines are
  ignored. See `affirmations.txt` for an example.
- `output.wav`: optional, defaults to `subliminal.wav`.

If the program finds no model in `MODEL_PATH`, it asks which one to download:

| No. | Model | Size | Property |
|---|---|---|---|
| 1 | Piper Lessac (US, female) | 67 MB | fast |
| 2 | Piper Amy (US, female) | 67 MB | fast |
| 3 | Piper Alba (British, female) | 67 MB | fast |
| 4 | Kokoro (11 US/UK voices) | 320 MB | most natural |

If the folder holds several models, the alphabetically first one is used. To
switch models, delete the model folder or change `MODEL_PATH`.

## Settings

Settings live in a `.env` file in the working directory. Copy `.env.dist` to
`.env` as a starting point. Without a `.env`, the defaults apply.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `models` | Folder that holds the TTS model |
| `PROVIDER` | `cpu` (`cuda` in a CUDA build) | `cpu` or `cuda` (see below) |
| `VOICE_DB` | `-30` | Voice level in dB; lower = more hidden |
| `MODE` | `sleep` | `sleep`, `work`, `meditate` or `active` (see below) |
| `SPEED` | `2` | Speaking rate of the voice; 1 = normal |
| `LAYERS` | `3` | Copies of the voice stacked on top of each other; 1 = none |
| `SPEAKER` | `0` | Speaker id for multi-voice models (Kokoro: 0–10) |

Invalid values are silently replaced by the default, except for `MODE`:
an unknown mode stops the program.

## Modes

The mode sets the track length and how the beat frequency moves.

| Mode | Length | Beat frequency |
|---|---|---|
| `sleep` | 6 h | follows four sleep cycles, see below |
| `work` | 30 min | 15 Hz (low beta, focused attention) |
| `meditate` | 30 min | 10 Hz falling to 6 Hz in the first 5 min (alpha into theta) |
| `active` | 30 min | 20 Hz (beta, alert; for chores and other active tasks) |

The sleep track follows the textbook pattern of a healthy night: four cycles
of 90 minutes, with deep sleep concentrated in the first cycles and REM
phases that grow toward the morning.

| Cycle | Time | Deep sleep (2–3 Hz, delta) | REM (6 Hz, theta) |
|---|---|---|---|
| 1 | 0:00–1:30 | 40 min, after 30 min of falling asleep from 8 Hz | 5 min |
| 2 | 1:30–3:00 | 30 min | 20 min |
| 3 | 3:00–4:30 | 15 min, shallower (3 Hz) | 30 min |
| 4 | 4:30–6:00 | none | 35 min, then a rise to 10 Hz for waking |

Between these phases the beat passes through light sleep at 4.5 Hz. The exact
key points are the `MODES` table in `src/main.rs`.

Real nights differ from person to person and from night to night, and the
track cannot know when you actually fall asleep. That beats steer sleep
stages, focus or mood is not established.

## The audio

- **Format:** mono WAV, 16 bit, at the model's sample rate (Piper 22.05 kHz,
  Kokoro 24 kHz). A sleep track is about 1 GB.
- **Voice:** the affirmations are spoken in order at `SPEED` times the normal
  rate, with a 0.5 s pause in between, and loop for the whole track. The
  program prints how long one loop takes.
- **Layers:** the loop is stacked `LAYERS` times, each copy shifted by an
  equal share of the loop, so several affirmations sound at once. The overall
  voice level stays the same whatever the number of layers.
- **Monaural beats:** two tones around 180 Hz are summed in one channel, so
  they work over speakers too, not only over headphones.
- **Fades:** start and end fade in and out over 5 s each.

Whether the voice really sits below the hearing threshold depends on your
headphones and playback volume, so tune `VOICE_DB` by ear.

### How fast, how many layers

There is no agreed limit among people who make subliminals, and no research
that sets one. Hobbyist guides most often name about twice the normal speed as
the ceiling and advise a few layers that stay distinguishable, on the grounds
that more turns the voice into noise. The defaults (`SPEED=2`, `LAYERS=3`) sit
at that point. Higher values are possible; nothing shows that they work
better, or that subliminal audio changes behavior at all.

## CUDA

Windows only, and so far tested only up to the point of producing the files:

```
cargo build --release --features cuda
```

The build downloads the CUDA runtime of sherpa-onnx once (about 570 MB, so the
first build sits at the build-script step for a few minutes) and puts seven
DLLs next to `target\release\tts_subliminal.exe`. The program then uses the
GPU by default; `PROVIDER=cpu` in `.env` switches back. To move the program
somewhere else, copy the `.exe` together with those DLLs.

Requirements on the machine: an NVIDIA GPU with a current driver, plus the
CUDA 12 and cuDNN 9 runtime libraries installed and on the `PATH`.

A plain `cargo build --release` afterwards goes back to the static CPU binary.
The DLLs left in `target\release` do no harm; that build does not load them.

## Notes

- **Version pin:** `sherpa-onnx` is pinned to 1.13.8. From 2.0 on, the built-in
  phonemizer that the Piper models need is gone.
- **Speed:** in CPU mode the affirmations are synthesized in parallel, one
  worker per logical core, so progress lines appear out of order. The track
  keeps the order of the input file.
- **Tests:** `cargo test` checks the beat progression and the mixing.

## License

Copyright (C) 2026 einhornimmond

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. It comes WITHOUT ANY WARRANTY. See [LICENSE](LICENSE) for the full
text.

## Third-party software

The executable links the following components statically. Licenses are as
declared by the upstream projects; check them yourself before you redistribute
a binary.

Native libraries, bundled through sherpa-onnx 1.13.8:

| Component | License |
|---|---|
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | Apache-2.0 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT |
| [espeak-ng](https://github.com/espeak-ng/espeak-ng) (incl. ucd-tools) | GPL-3.0-or-later |
| [piper-phonemize](https://github.com/rhasspy/piper-phonemize) | MIT |
| [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank), [kaldi-decoder](https://github.com/k2-fsa/kaldi-decoder), [kaldifst](https://github.com/k2-fsa/kaldifst) / OpenFst | Apache-2.0 |
| [ssentencepiece](https://github.com/k2-fsa/ssentencepiece) | Apache-2.0 |
| [KISS FFT](https://github.com/mborgerding/kissfft) | BSD-3-Clause |

Because espeak-ng is GPL-licensed, any distributed binary falls under the
GPL-3.0 as a whole.

Rust crates (direct dependencies):

| Crate | License |
|---|---|
| [sherpa-onnx](https://crates.io/crates/sherpa-onnx) | Apache-2.0 |
| [hound](https://crates.io/crates/hound) | Apache-2.0 |
| [ureq](https://crates.io/crates/ureq) | MIT OR Apache-2.0 |
| [tar](https://crates.io/crates/tar) | MIT OR Apache-2.0 |
| [bzip2](https://crates.io/crates/bzip2) | MIT OR Apache-2.0 |
| [dotenvy](https://crates.io/crates/dotenvy) | MIT |

Their transitive dependencies use MIT, Apache-2.0, BSD-3-Clause, ISC, Zlib,
Unicode-3.0 and CDLA-Permissive-2.0 licenses, all compatible with the GPL-3.0.
Run `cargo metadata` or [cargo-license](https://crates.io/crates/cargo-license)
for the full list.

## Speech models

The models are not part of this repository; the program downloads them from
the [sherpa-onnx TTS model releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models).
Each model has its own license, which also depends on the dataset it was
trained on and may restrict commercial use. Read the `MODEL_CARD` or `LICENSE`
file in the downloaded model folder before you publish audio made with it.
