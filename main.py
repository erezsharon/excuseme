"""
main.py -- NeuralFM
===================

Entry point and UI for NeuralFM, a personal focus-music app that applies
real-time amplitude modulation at brainwave frequencies to ambient / lo-fi
music.

Run with::

    python main.py

The UI is built with customtkinter (dark theme). Audio processing lives in
:mod:`audio_engine`, downloads in :mod:`downloader`, session timing in
:mod:`timer_module`.
"""

from __future__ import annotations

import math
import os
import threading
import time

import customtkinter as ctk

from audio_engine import MODES, AudioEngine
from downloader import DEFAULT_SOURCES, DownloadError, download_audio
from timer_module import Phase, SessionTimer, TimerMode

# Optional drag-and-drop support. Falls back gracefully to the file picker.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore

    _DND_AVAILABLE = True
except Exception:
    _DND_AVAILABLE = False


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

BG = "#0d0f14"
CARD = "#161a22"
ACCENT = "#5b8cff"
ACCENT_DIM = "#2a3550"
TEXT = "#e6e9ef"
MUTED = "#8a93a6"

MODE_ORDER = ["focus", "relax", "meditate", "sleep"]


class NeuralFMApp:
    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        self.engine = AudioEngine()
        self.timer = SessionTimer()
        self.current_mode = "focus"
        self.current_title = "No track loaded"
        self._viz_phase = 0.0
        self._last_viz_t = time.time()

        self.engine.set_mode(self.current_mode)
        self.engine.on_finish = self._on_track_finished
        self.timer.on_phase_change = self._on_phase_change

        root.title("NeuralFM")
        root.geometry("760x760")
        root.minsize(680, 720)
        root.configure(fg_color=BG)

        self._build_ui()
        self._select_mode("focus")
        self._set_timer_mode(TimerMode.INFINITE)

        # Animation / clock loops.
        self._animate_visualizer()
        self._tick_timer()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        header = ctk.CTkLabel(
            self.root,
            text="NeuralFM",
            font=ctk.CTkFont(size=30, weight="bold"),
            text_color=TEXT,
        )
        header.pack(pady=(18, 0))
        ctk.CTkLabel(
            self.root,
            text="Focus music with real-time brainwave modulation",
            font=ctk.CTkFont(size=13),
            text_color=MUTED,
        ).pack(pady=(0, 10))

        # ---- Mode selector ---------------------------------------------
        mode_card = self._card()
        ctk.CTkLabel(
            mode_card, text="MODE", font=ctk.CTkFont(size=12, weight="bold"),
            text_color=MUTED,
        ).pack(anchor="w", padx=14, pady=(10, 4))

        btn_row = ctk.CTkFrame(mode_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(0, 12))
        self.mode_buttons: dict[str, ctk.CTkButton] = {}
        for key in MODE_ORDER:
            m = MODES[key]
            b = ctk.CTkButton(
                btn_row,
                text=f"{m.label}\n{m.band} · {m.freq:g}Hz",
                command=lambda k=key: self._select_mode(k),
                height=54,
                fg_color=ACCENT_DIM,
                hover_color=ACCENT,
                text_color=TEXT,
                font=ctk.CTkFont(size=13, weight="bold"),
            )
            b.pack(side="left", expand=True, fill="x", padx=4)
            self.mode_buttons[key] = b

        # ---- Visualizer -------------------------------------------------
        viz_card = self._card()
        self.canvas = ctk.CTkCanvas(
            viz_card, height=180, bg=CARD, highlightthickness=0
        )
        self.canvas.pack(fill="x", padx=14, pady=14)

        # ---- Sliders ----------------------------------------------------
        slider_card = self._card()
        self.intensity_value = ctk.CTkLabel(
            slider_card, text="Neural intensity: 100%", text_color=TEXT,
            font=ctk.CTkFont(size=13),
        )
        self.intensity_value.pack(anchor="w", padx=14, pady=(12, 0))
        self.intensity_slider = ctk.CTkSlider(
            slider_card, from_=0, to=100, command=self._on_intensity,
            progress_color=ACCENT, button_color=ACCENT,
        )
        self.intensity_slider.set(100)
        self.intensity_slider.pack(fill="x", padx=14, pady=(2, 10))

        self.volume_value = ctk.CTkLabel(
            slider_card, text="Volume: 80%", text_color=TEXT,
            font=ctk.CTkFont(size=13),
        )
        self.volume_value.pack(anchor="w", padx=14, pady=(2, 0))
        self.volume_slider = ctk.CTkSlider(
            slider_card, from_=0, to=100, command=self._on_volume,
            progress_color=ACCENT, button_color=ACCENT,
        )
        self.volume_slider.set(80)
        self.volume_slider.pack(fill="x", padx=14, pady=(2, 14))
        self.engine.set_volume(80)

        # ---- Timer ------------------------------------------------------
        timer_card = self._card()
        trow = ctk.CTkFrame(timer_card, fg_color="transparent")
        trow.pack(fill="x", padx=14, pady=(12, 6))
        ctk.CTkLabel(
            trow, text="TIMER", font=ctk.CTkFont(size=12, weight="bold"),
            text_color=MUTED,
        ).pack(side="left")
        self.timer_display = ctk.CTkLabel(
            trow, text="00:00", font=ctk.CTkFont(size=26, weight="bold"),
            text_color=TEXT,
        )
        self.timer_display.pack(side="right")
        self.timer_phase = ctk.CTkLabel(
            trow, text="", text_color=ACCENT, font=ctk.CTkFont(size=13),
        )
        self.timer_phase.pack(side="right", padx=10)

        tmode_row = ctk.CTkFrame(timer_card, fg_color="transparent")
        tmode_row.pack(fill="x", padx=10, pady=(0, 6))
        self.timer_mode_buttons: dict[TimerMode, ctk.CTkButton] = {}
        for tm in TimerMode:
            b = ctk.CTkButton(
                tmode_row, text=tm.value, height=32,
                command=lambda m=tm: self._set_timer_mode(m),
                fg_color=ACCENT_DIM, hover_color=ACCENT, text_color=TEXT,
            )
            b.pack(side="left", expand=True, fill="x", padx=4)
            self.timer_mode_buttons[tm] = b

        tctl_row = ctk.CTkFrame(timer_card, fg_color="transparent")
        tctl_row.pack(fill="x", padx=10, pady=(0, 12))
        ctk.CTkLabel(tctl_row, text="Minutes:", text_color=MUTED).pack(
            side="left", padx=(4, 4)
        )
        self.minutes_entry = ctk.CTkEntry(tctl_row, width=60)
        self.minutes_entry.insert(0, "25")
        self.minutes_entry.pack(side="left", padx=(0, 8))
        self.minutes_entry.bind("<Return>", lambda e: self._apply_minutes())
        ctk.CTkButton(
            tctl_row, text="Set", width=50, command=self._apply_minutes,
            fg_color=ACCENT_DIM, hover_color=ACCENT,
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            tctl_row, text="Reset", width=60, command=self._reset_timer,
            fg_color=ACCENT_DIM, hover_color=ACCENT,
        ).pack(side="right", padx=4)

        # ---- Audio source ----------------------------------------------
        src_card = self._card()
        ctk.CTkLabel(
            src_card, text="AUDIO SOURCE",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=MUTED,
        ).pack(anchor="w", padx=14, pady=(10, 4))

        url_row = ctk.CTkFrame(src_card, fg_color="transparent")
        url_row.pack(fill="x", padx=10, pady=(0, 6))
        self.url_entry = ctk.CTkEntry(
            url_row, placeholder_text="Paste a YouTube URL…"
        )
        self.url_entry.pack(side="left", expand=True, fill="x", padx=4)
        ctk.CTkButton(
            url_row, text="Load URL", width=90, command=self._load_url,
            fg_color=ACCENT, hover_color=ACCENT_DIM,
        ).pack(side="left", padx=4)

        preset_row = ctk.CTkFrame(src_card, fg_color="transparent")
        preset_row.pack(fill="x", padx=10, pady=(0, 6))
        ctk.CTkLabel(preset_row, text="Presets:", text_color=MUTED).pack(
            side="left", padx=(4, 6)
        )
        self.preset_menu = ctk.CTkOptionMenu(
            preset_row,
            values=[p.name for p in DEFAULT_SOURCES],
            command=self._load_preset,
            fg_color=ACCENT_DIM, button_color=ACCENT_DIM,
            button_hover_color=ACCENT,
        )
        self.preset_menu.set("Choose a curated stream…")
        self.preset_menu.pack(side="left", expand=True, fill="x", padx=4)

        file_row = ctk.CTkFrame(src_card, fg_color="transparent")
        file_row.pack(fill="x", padx=10, pady=(0, 12))
        dnd_hint = (
            "Drag & drop an MP3/WAV here, or"
            if _DND_AVAILABLE
            else "Pick a local MP3/WAV file:"
        )
        self.file_drop = ctk.CTkLabel(
            file_row, text=dnd_hint, text_color=MUTED,
            fg_color=CARD, corner_radius=6, height=34,
        )
        self.file_drop.pack(side="left", expand=True, fill="x", padx=4)
        ctk.CTkButton(
            file_row, text="Browse…", width=90, command=self._pick_file,
            fg_color=ACCENT_DIM, hover_color=ACCENT,
        ).pack(side="left", padx=4)

        if _DND_AVAILABLE:
            self.file_drop.drop_target_register(DND_FILES)
            self.file_drop.dnd_bind("<<Drop>>", self._on_drop)

        # ---- Transport + status ----------------------------------------
        transport = ctk.CTkFrame(self.root, fg_color="transparent")
        transport.pack(fill="x", padx=20, pady=(4, 4))
        self.play_btn = ctk.CTkButton(
            transport, text="▶  Play", command=self._toggle_play,
            height=46, font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=ACCENT, hover_color=ACCENT_DIM,
        )
        self.play_btn.pack(side="left", expand=True, fill="x", padx=4)
        ctk.CTkButton(
            transport, text="■  Stop", command=self._stop, height=46,
            fg_color=ACCENT_DIM, hover_color=ACCENT,
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(side="left", expand=True, fill="x", padx=4)

        self.status = ctk.CTkLabel(
            self.root, text="No track loaded.", text_color=MUTED,
            font=ctk.CTkFont(size=12),
        )
        self.status.pack(pady=(4, 12))

    def _card(self) -> ctk.CTkFrame:
        c = ctk.CTkFrame(self.root, fg_color=CARD, corner_radius=12)
        c.pack(fill="x", padx=20, pady=6)
        return c

    # -------------------------------------------------------------- modes
    def _select_mode(self, key: str) -> None:
        self.current_mode = key
        self.engine.set_mode(key)
        for k, b in self.mode_buttons.items():
            b.configure(fg_color=ACCENT if k == key else ACCENT_DIM)
        m = MODES[key]
        self._set_status(f"Mode: {m.label} ({m.band}, {m.freq:g} Hz)")

    def _on_intensity(self, value: float) -> None:
        pct = int(round(value))
        self.intensity_value.configure(text=f"Neural intensity: {pct}%")
        self.engine.set_intensity(pct)

    def _on_volume(self, value: float) -> None:
        pct = int(round(value))
        self.volume_value.configure(text=f"Volume: {pct}%")
        self.engine.set_volume(pct)

    # -------------------------------------------------------------- timer
    def _set_timer_mode(self, mode: TimerMode) -> None:
        minutes = self._read_minutes()
        self.timer.set_mode(mode, timer_minutes=minutes)
        for tm, b in self.timer_mode_buttons.items():
            b.configure(fg_color=ACCENT if tm == mode else ACCENT_DIM)
        self._refresh_timer_label(self.timer.state())

    def _apply_minutes(self) -> None:
        self.timer.set_timer_minutes(self._read_minutes())
        self._refresh_timer_label(self.timer.state())

    def _read_minutes(self) -> int:
        try:
            return max(1, int(float(self.minutes_entry.get())))
        except (ValueError, TypeError):
            return 25

    def _reset_timer(self) -> None:
        self.timer.reset()
        self._refresh_timer_label(self.timer.state())

    def _tick_timer(self) -> None:
        state = self.timer.tick(1.0)
        self._refresh_timer_label(state)
        self.root.after(1000, self._tick_timer)

    def _refresh_timer_label(self, state) -> None:
        self.timer_display.configure(text=state.display)
        if state.mode is TimerMode.POMODORO:
            self.timer_phase.configure(
                text=f"{state.phase.value} · cycle {state.cycle}"
            )
        elif state.phase is Phase.DONE:
            self.timer_phase.configure(text="Done")
        else:
            self.timer_phase.configure(text="")

    def _on_phase_change(self, phase: Phase) -> None:
        # Called from the timer when pomodoro flips or a countdown ends.
        if phase is Phase.DONE:
            self.root.after(0, self._stop)

    # ------------------------------------------------------------- source
    def _load_url(self) -> None:
        url = self.url_entry.get().strip()
        if not url:
            self._set_status("Paste a YouTube URL first.")
            return
        self._download_async(url)

    def _load_preset(self, name: str) -> None:
        for p in DEFAULT_SOURCES:
            if p.name == name:
                self.url_entry.delete(0, "end")
                self.url_entry.insert(0, p.url)
                self._download_async(p.url)
                return

    def _download_async(self, url: str) -> None:
        self._set_status("Downloading audio… this can take a moment.")

        def worker() -> None:
            try:
                path, title = download_audio(url)
                self.engine.load(path)
                self.current_title = title
                self.root.after(
                    0, lambda: self._on_loaded(f"Loaded: {title}")
                )
            except DownloadError as exc:
                self.root.after(0, lambda: self._set_status(str(exc)))
            except Exception as exc:  # pragma: no cover - runtime safety
                self.root.after(
                    0, lambda e=exc: self._set_status(f"Error: {e}")
                )

        threading.Thread(target=worker, daemon=True).start()

    def _pick_file(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Choose an audio file",
            filetypes=[
                ("Audio", "*.mp3 *.wav *.flac *.ogg *.m4a"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_file(path)

    def _on_drop(self, event) -> None:
        path = event.data.strip().strip("{}")
        if path:
            self._load_file(path)

    def _load_file(self, path: str) -> None:
        self._set_status("Loading file…")

        def worker() -> None:
            try:
                self.engine.load(path)
                self.current_title = os.path.basename(path)
                self.root.after(
                    0, lambda: self._on_loaded(f"Loaded: {self.current_title}")
                )
            except Exception as exc:
                self.root.after(
                    0, lambda e=exc: self._set_status(f"Could not load: {e}")
                )

        threading.Thread(target=worker, daemon=True).start()

    def _on_loaded(self, message: str) -> None:
        self._set_status(message)
        self.engine.set_loop(True)
        self.engine.play()
        self.timer.start()
        self._update_play_button()

    # ---------------------------------------------------------- transport
    def _toggle_play(self) -> None:
        if not self.engine.loaded:
            self._set_status("Load a track first (URL, preset or file).")
            return
        self.engine.toggle()
        if self.engine.is_playing:
            self.timer.start()
        else:
            self.timer.pause()
        self._update_play_button()

    def _stop(self) -> None:
        self.engine.stop()
        self.timer.pause()
        self.timer.reset()
        self._update_play_button()
        self._set_status("Stopped.")

    def _on_track_finished(self) -> None:
        self.root.after(0, self._update_play_button)

    def _update_play_button(self) -> None:
        if self.engine.is_playing:
            self.play_btn.configure(text="❚❚  Pause")
        else:
            self.play_btn.configure(text="▶  Play")

    # --------------------------------------------------------- visualizer
    def _animate_visualizer(self) -> None:
        now = time.time()
        dt = now - self._last_viz_t
        self._last_viz_t = now

        freq = self.engine.mod_freq
        playing = self.engine.is_playing
        if playing:
            self._viz_phase = (self._viz_phase + 2 * math.pi * freq * dt) % (
                2 * math.pi
            )

        self._draw_pulse(playing)
        self.root.after(33, self._animate_visualizer)  # ~30 fps

    def _draw_pulse(self, playing: bool) -> None:
        c = self.canvas
        c.delete("all")
        w = c.winfo_width() or 700
        h = c.winfo_height() or 180
        cx, cy = w / 2, h / 2

        # Pulse 0..1 from the modulation oscillator. Idle = gentle breathing.
        if playing:
            pulse = 0.5 + 0.5 * math.sin(self._viz_phase)
        else:
            pulse = 0.5 + 0.5 * math.sin(self._last_viz_t * 1.2)

        base = min(w, h) * 0.18
        amp = min(w, h) * 0.16
        intensity = self.intensity_slider.get() / 100.0
        radius = base + amp * pulse * (0.4 + 0.6 * intensity)

        accent = self._mode_color()
        # Outer halo rings.
        for i, scale in enumerate((1.7, 1.35, 1.0)):
            r = radius * scale
            shade = self._blend(CARD, accent, (0.12 + 0.18 * pulse) / (i + 1))
            c.create_oval(
                cx - r, cy - r, cx + r, cy + r, outline=shade, width=2
            )
        # Core disc.
        core = self._blend(accent, "#ffffff", 0.15 * pulse)
        c.create_oval(
            cx - radius, cy - radius, cx + radius, cy + radius,
            fill=core, outline="",
        )
        label = f"{self.engine.mod_freq:g} Hz" if playing else "paused"
        c.create_text(
            cx, cy, text=label, fill="#0d0f14",
            font=("TkDefaultFont", 16, "bold"),
        )

    def _mode_color(self) -> str:
        colors = {
            "focus": "#5b8cff",
            "relax": "#3ddc97",
            "meditate": "#b76bff",
            "sleep": "#6b7bff",
        }
        return colors.get(self.current_mode, ACCENT)

    @staticmethod
    def _blend(c1: str, c2: str, t: float) -> str:
        t = max(0.0, min(1.0, t))
        a = tuple(int(c1[i : i + 2], 16) for i in (1, 3, 5))
        b = tuple(int(c2[i : i + 2], 16) for i in (1, 3, 5))
        m = tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))
        return f"#{m[0]:02x}{m[1]:02x}{m[2]:02x}"

    # -------------------------------------------------------------- misc
    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)


def main() -> None:
    root = ctk.CTk()
    if _DND_AVAILABLE:
        # Bolt the tkinterdnd2 machinery onto the customtkinter root so we keep
        # the proper dark theming while still accepting file drops.
        try:
            root.TkdndVersion = TkinterDnD._require(root)
        except Exception:
            pass
    app = NeuralFMApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
