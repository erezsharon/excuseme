# NeuralFM 🧠🎧

A personal focus-music desktop app — a self-hosted [Brain.fm](https://brain.fm)
alternative.

NeuralFM plays ambient / lo-fi music from **local files** or a **YouTube URL**
and applies **real-time amplitude modulation (AM)** at brainwave frequencies to
nudge your brain toward focus, relaxation, meditation, or sleep.

> ⚠️ **Not a medical device.** The "neural" effect is amplitude modulation of
> the audio at EEG-band frequencies. Results are subjective. Don't use Sleep
> mode while driving, and stop if you feel any discomfort.

---

## How it works

The core lives in [`audio_engine.py`](audio_engine.py). The headline function:

```python
apply_neural_am(audio, sample_rate, mode, intensity=1.0, chunk_size=4096)
```

- **Sinusoidal AM** at the mode's brainwave frequency:

  | Mode      | Band  | Frequency | Modulation depth |
  |-----------|-------|-----------|------------------|
  | Focus     | Beta  | 16 Hz     | 0.30             |
  | Relax     | Alpha | 10 Hz     | 0.25             |
  | Meditate  | Theta | 6 Hz      | 0.30             |
  | Sleep     | Delta | 2 Hz      | 0.40             |

- Modulation is applied **only to the mid band (300 Hz – 3 kHz)** via a
  4th-order Butterworth band-pass filter, so bass and treble stay clean and the
  effect stays subtle.
- Audio is processed in **4096-sample chunks** for low latency. During live
  playback a stateful `NeuralAM` instance keeps the band-pass filter state and
  the oscillator phase continuous across chunks (no clicks, no phase jumps).
- The **neural intensity slider** scales the modulation depth from 0 to its
  baseline value.

---

## Features

- **Modes:** Focus / Relax / Meditate / Sleep (one active at a time).
- **Neural intensity slider** (0–100%) and **volume slider**.
- **Timer:** Infinite (count-up), Timer (countdown with custom minutes),
  Pomodoro (25 min work / 5 min break cycles).
- **Audio sources:**
  - Paste a YouTube URL → downloaded via `yt-dlp` → played with modulation.
  - Curated CC-friendly presets (long, calm, vocal-free streams).
  - Local **MP3 / WAV** (and FLAC/OGG/M4A) via file picker, plus optional
    drag-and-drop.
- **Visualizer:** a pulsing circle that beats at the current modulation
  frequency, tinted per mode, so you can *see* the effect.
- **Dark theme** throughout.

---

## Install

Requires **Python 3.9+** and **ffmpeg** (used by `yt-dlp` and `pydub`).

```bash
# 1. Install ffmpeg
#    macOS:    brew install ffmpeg
#    Ubuntu:   sudo apt install ffmpeg
#    Windows:  https://ffmpeg.org/download.html  (add to PATH)

# 2. Create a virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. (Optional) drag-and-drop support
pip install tkinterdnd2
```

`sounddevice` needs the PortAudio library. It ships with the wheel on most
platforms; on Linux you may need `sudo apt install libportaudio2`.

---

## Run

```bash
python main.py
```

1. Pick a **mode**.
2. Load a track: paste a YouTube URL and hit **Load URL**, choose a **preset**,
   or **Browse…** for a local file (downloads/loads happen in the background).
3. Adjust **neural intensity** and **volume**.
4. Choose a **timer** mode and press **Play**.

---

## Project structure

```
.
├── main.py          # entry point + customtkinter UI + visualizer
├── audio_engine.py  # apply_neural_am(), NeuralAM modulator, AudioEngine playback
├── downloader.py    # yt-dlp wrapper + curated default sources
├── timer_module.py  # infinite / countdown / pomodoro logic
├── requirements.txt
└── README.md
```

---

## Default sources & licensing

The presets in [`downloader.py`](downloader.py) point to long, low-salience
ambient / lo-fi YouTube streams that suit background listening. They are offered
for convenience only — **stream availability and licensing change over time, so
verify the licence of anything you rely on.** You can paste any YouTube URL or
load your own files.
