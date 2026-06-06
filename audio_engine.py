"""
audio_engine.py
===============

Core of NeuralFM.

Implements scientifically-flavoured amplitude modulation (AM) of audio at
brainwave frequencies, applied *only* to the mid-frequency band (300 Hz - 3 kHz)
so that bass and treble stay untouched and the modulation stays subtle.

Two things live here:

* ``apply_neural_am(audio, sample_rate, mode, ...)`` -- a stateless convenience
  function that processes a whole numpy buffer in 4096-sample chunks. Handy for
  offline rendering / tests.

* ``NeuralAM`` -- a *stateful*, phase-continuous modulator used during live
  playback (one ``process_chunk`` call per audio callback).

* ``AudioEngine`` -- a ``sounddevice`` output stream that loads a local file or
  a downloaded YouTube track, applies ``NeuralAM`` in real time, and exposes
  play / pause / stop, volume, mode and intensity controls plus the current
  modulation phase for the UI visualiser.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

try:
    from scipy.signal import butter, sosfilt, sosfilt_zi
except Exception as exc:  # pragma: no cover - scipy is a hard dependency
    raise ImportError(
        "scipy is required for the NeuralFM audio engine. "
        "Install it with `pip install scipy`."
    ) from exc


# ---------------------------------------------------------------------------
# Mode table
# ---------------------------------------------------------------------------
# Frequencies and modulation depths come straight from the product spec. The
# depth is the *baseline* depth at 100% neural-effect intensity; the UI slider
# scales it down towards 0.
@dataclass(frozen=True)
class Mode:
    key: str
    label: str
    freq: float          # modulation frequency in Hz
    depth: float         # baseline modulation depth at 100% intensity
    band: str            # brainwave band name (cosmetic)


MODES: dict[str, Mode] = {
    "focus":    Mode("focus",    "Focus",    16.0, 0.30, "Beta"),
    "relax":    Mode("relax",    "Relax",    10.0, 0.25, "Alpha"),
    "meditate": Mode("meditate", "Meditate",  6.0, 0.30, "Theta"),
    "sleep":    Mode("sleep",    "Sleep",     2.0, 0.40, "Delta"),
}

# Mid band that gets modulated. Bass below LOW_HZ and treble above HIGH_HZ are
# left completely alone.
LOW_HZ = 300.0
HIGH_HZ = 3000.0

CHUNK_SIZE = 4096


# ---------------------------------------------------------------------------
# Stateful, phase-continuous modulator
# ---------------------------------------------------------------------------
class NeuralAM:
    """Applies brainwave AM to the mid band, one chunk at a time.

    The class keeps two pieces of state between chunks so that real-time,
    block-by-block processing produces no clicks or phase jumps:

    * the band-pass filter delay state (``_zi``), and
    * the modulation oscillator phase (``_phase``).
    """

    def __init__(
        self,
        sample_rate: int,
        mode: str = "focus",
        intensity: float = 1.0,
        channels: int = 2,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.intensity = float(np.clip(intensity, 0.0, 1.0))
        self._lock = threading.Lock()
        self._phase = 0.0
        self._mode = MODES.get(mode, MODES["focus"])
        self._design_filter()
        self._reset_filter_state()

    # -- configuration -----------------------------------------------------
    def _design_filter(self) -> None:
        nyq = self.sample_rate / 2.0
        low = LOW_HZ / nyq
        high = min(HIGH_HZ / nyq, 0.999)
        # 4th-order Butterworth band-pass, second-order-sections for stability.
        self._sos = butter(4, [low, high], btype="band", output="sos")
        # Per-channel steady-state initial conditions, scaled per chunk.
        self._zi_proto = sosfilt_zi(self._sos)

    def _reset_filter_state(self) -> None:
        # One filter-state array per channel.
        self._zi = [None] * self.channels

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self._mode = MODES.get(mode, self._mode)

    def set_intensity(self, intensity: float) -> None:
        with self._lock:
            self.intensity = float(np.clip(intensity, 0.0, 1.0))

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def phase(self) -> float:
        """Current oscillator phase in radians (for the UI visualiser)."""
        return self._phase

    @property
    def freq(self) -> float:
        return self._mode.freq

    # -- processing --------------------------------------------------------
    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Process one block of audio.

        ``chunk`` may be 1-D (mono) or 2-D ``(frames, channels)``. The returned
        array has the same shape and dtype ``float32``.
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk[:, np.newaxis]

        frames, channels = chunk.shape
        if channels != self.channels:
            self.channels = channels
            self._reset_filter_state()

        with self._lock:
            mode = self._mode
            intensity = self.intensity
            phase = self._phase

        depth = mode.depth * intensity
        freq = mode.freq

        # Modulation envelope: m(t) = depth * sin(2*pi*f*t + phase).
        # Output = original + midband * m(t). At m == 0 the signal is untouched,
        # so the perceived loudness baseline is preserved.
        t = np.arange(frames, dtype=np.float64) / self.sample_rate
        env = (depth * np.sin(2.0 * np.pi * freq * t + phase)).astype(np.float32)

        out = np.empty_like(chunk)
        for ch in range(channels):
            x = chunk[:, ch]
            if self._zi[ch] is None:
                # Initialise filter state from the first sample to avoid a
                # transient at the very start of playback.
                self._zi[ch] = self._zi_proto * x[0]
            mid, self._zi[ch] = sosfilt(self._sos, x, zi=self._zi[ch])
            out[:, ch] = x + mid.astype(np.float32) * env

        # Advance and wrap the oscillator phase for the next chunk.
        with self._lock:
            self._phase = float(
                (phase + 2.0 * np.pi * freq * frames / self.sample_rate)
                % (2.0 * np.pi)
            )

        return out


# ---------------------------------------------------------------------------
# Stateless convenience wrapper
# ---------------------------------------------------------------------------
def apply_neural_am(
    audio: np.ndarray,
    sample_rate: int,
    mode: str = "focus",
    intensity: float = 1.0,
    chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    """Apply brainwave AM to a whole buffer, processed in real-time-sized chunks.

    This is the headline function from the spec. It is *stateless* from the
    caller's point of view -- it spins up a fresh :class:`NeuralAM` and streams
    ``audio`` through it in ``chunk_size`` (default 4096) blocks, exactly as the
    live engine would, which keeps the offline and real-time results identical.

    Parameters
    ----------
    audio:
        ``(frames,)`` mono or ``(frames, channels)`` float array in [-1, 1].
    sample_rate:
        Sample rate in Hz.
    mode:
        One of ``focus``, ``relax``, ``meditate``, ``sleep``.
    intensity:
        Neural-effect intensity in [0, 1]; scales the modulation depth.
    chunk_size:
        Block size in frames (4096 by default for low latency).
    """
    audio = np.asarray(audio, dtype=np.float32)
    squeeze_back = audio.ndim == 1
    if squeeze_back:
        audio = audio[:, np.newaxis]

    channels = audio.shape[1]
    engine = NeuralAM(sample_rate, mode=mode, intensity=intensity, channels=channels)

    out = np.empty_like(audio)
    for start in range(0, len(audio), chunk_size):
        end = start + chunk_size
        out[start:end] = engine.process_chunk(audio[start:end])

    return out[:, 0] if squeeze_back else out


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_audio(path: str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load an audio file into a float32 ``(frames, channels)`` array.

    Tries :mod:`pydub` first (great with ffmpeg for mp3/m4a/webm), then falls
    back to :mod:`soundfile` and finally :mod:`librosa`.
    """
    # 1) pydub (needs ffmpeg) -- most robust for compressed formats.
    try:
        from pydub import AudioSegment

        seg = AudioSegment.from_file(path)
        samples = np.array(seg.get_array_of_samples())
        if seg.channels > 1:
            samples = samples.reshape((-1, seg.channels))
        else:
            samples = samples.reshape((-1, 1))
        max_val = float(1 << (8 * seg.sample_width - 1))
        data = samples.astype(np.float32) / max_val
        sr = seg.frame_rate
        if target_sr and sr != target_sr:
            data, sr = _resample(data, sr, target_sr)
        return data, sr
    except Exception:
        pass

    # 2) soundfile.
    try:
        import soundfile as sf

        data, sr = sf.read(path, always_2d=True, dtype="float32")
        if target_sr and sr != target_sr:
            data, sr = _resample(data, sr, target_sr)
        return data, sr
    except Exception:
        pass

    # 3) librosa (also pulls in audioread / soundfile under the hood).
    import librosa

    data, sr = librosa.load(path, sr=target_sr, mono=False)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    else:
        data = data.T  # librosa returns (channels, frames)
    return data, sr


def _resample(data: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    """Lightweight linear resampler (avoids a hard scipy/librosa requirement)."""
    if sr == target_sr:
        return data, sr
    n_out = int(round(data.shape[0] * target_sr / sr))
    old_idx = np.linspace(0.0, 1.0, num=data.shape[0], endpoint=False)
    new_idx = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    out = np.empty((n_out, data.shape[1]), dtype=np.float32)
    for ch in range(data.shape[1]):
        out[:, ch] = np.interp(new_idx, old_idx, data[:, ch]).astype(np.float32)
    return out, target_sr


# ---------------------------------------------------------------------------
# Real-time playback engine
# ---------------------------------------------------------------------------
class AudioEngine:
    """Streams an in-memory track through :class:`NeuralAM` via sounddevice.

    State transitions: stopped -> playing <-> paused. ``on_finish`` (if set) is
    called from the audio thread when playback reaches the end of the track.
    """

    def __init__(self, blocksize: int = CHUNK_SIZE) -> None:
        self.blocksize = blocksize
        self._audio: np.ndarray | None = None
        self._sr: int = 44100
        self._pos = 0
        self._volume = 0.8
        self._lock = threading.Lock()
        self._stream = None
        self._sd = None
        self._playing = False
        self._paused = False
        self._loop = False
        self._modulator: NeuralAM | None = None
        self._mode = "focus"
        self._intensity = 1.0
        self.on_finish = None  # optional callback()

    # -- track management --------------------------------------------------
    def load(self, path: str) -> None:
        """Load a file from disk and make it the current track (stops playback)."""
        data, sr = load_audio(path)
        self.set_buffer(data, sr)

    def set_buffer(self, data: np.ndarray, sr: int) -> None:
        self.stop()
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        with self._lock:
            self._audio = data
            self._sr = int(sr)
            self._pos = 0
            self._modulator = NeuralAM(
                self._sr,
                mode=self._mode,
                intensity=self._intensity,
                channels=data.shape[1],
            )

    @property
    def loaded(self) -> bool:
        return self._audio is not None

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def duration(self) -> float:
        if self._audio is None:
            return 0.0
        return self._audio.shape[0] / self._sr

    @property
    def position(self) -> float:
        return self._pos / self._sr if self._sr else 0.0

    # -- live controls -----------------------------------------------------
    def set_mode(self, mode: str) -> None:
        self._mode = mode
        if self._modulator is not None:
            self._modulator.set_mode(mode)

    def set_intensity(self, intensity_pct: float) -> None:
        """``intensity_pct`` is 0-100 from the UI slider."""
        self._intensity = float(np.clip(intensity_pct / 100.0, 0.0, 1.0))
        if self._modulator is not None:
            self._modulator.set_intensity(self._intensity)

    def set_volume(self, volume_pct: float) -> None:
        """``volume_pct`` is 0-100 from the UI slider."""
        with self._lock:
            self._volume = float(np.clip(volume_pct / 100.0, 0.0, 1.0))

    def set_loop(self, loop: bool) -> None:
        self._loop = bool(loop)

    @property
    def is_playing(self) -> bool:
        return self._playing and not self._paused

    @property
    def mod_phase(self) -> float:
        return self._modulator.phase if self._modulator else 0.0

    @property
    def mod_freq(self) -> float:
        return self._modulator.freq if self._modulator else MODES[self._mode].freq

    # -- transport ---------------------------------------------------------
    def play(self) -> None:
        if self._audio is None:
            return
        if self._paused and self._stream is not None:
            self._paused = False
            return
        if self._playing:
            return
        self._start_stream()

    def pause(self) -> None:
        if self._playing:
            self._paused = True

    def toggle(self) -> None:
        if not self._playing:
            self.play()
        elif self._paused:
            self.play()
        else:
            self.pause()

    def stop(self) -> None:
        self._playing = False
        self._paused = False
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        with self._lock:
            self._pos = 0
        if self._modulator is not None:
            self._modulator._reset_filter_state()

    def _start_stream(self) -> None:
        import sounddevice as sd

        # Keep a reference so the audio callback can raise CallbackStop without
        # re-importing inside the realtime thread.
        self._sd = sd
        channels = self._audio.shape[1]
        self._playing = True
        self._paused = False
        self._stream = sd.OutputStream(
            samplerate=self._sr,
            channels=channels,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    # -- audio thread ------------------------------------------------------
    def _callback(self, outdata, frames, time_info, status):  # noqa: ARG002
        if self._paused or self._audio is None:
            outdata.fill(0.0)
            return

        with self._lock:
            pos = self._pos
            audio = self._audio
            volume = self._volume

        end = pos + frames
        block = audio[pos:end]

        finished = False
        if len(block) < frames:
            if self._loop and len(audio):
                # Wrap around for seamless looping.
                pad = audio[: frames - len(block)]
                block = np.concatenate([block, pad], axis=0)
                end = frames - len(audio[pos:])
            else:
                pad = np.zeros((frames - len(block), audio.shape[1]), dtype=np.float32)
                block = np.concatenate([block, pad], axis=0)
                finished = True

        processed = self._modulator.process_chunk(block)
        np.multiply(processed, volume, out=processed)
        np.clip(processed, -1.0, 1.0, out=processed)
        outdata[:] = processed

        with self._lock:
            self._pos = 0 if (self._loop and end >= len(audio)) else end

        if finished:
            self._playing = False
            cb = self.on_finish
            if cb is not None:
                try:
                    cb()
                except Exception:
                    pass
            raise self._sd.CallbackStop()
