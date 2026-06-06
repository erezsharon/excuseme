"""
downloader.py
=============

Thin wrapper around ``yt-dlp`` that downloads the best audio track from a
YouTube URL, transcodes it to WAV with ffmpeg, and returns the local path so
:mod:`audio_engine` can load it.

Also holds the curated list of default ambient / lo-fi sources.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Curated, low-salience default sources
# ---------------------------------------------------------------------------
# These are long, calm, vocal-free streams that fit the "low salience" brief
# (good background music that doesn't grab attention). They are offered as
# convenience presets only -- always confirm the licence of any track before
# relying on it, as YouTube stream availability and licensing can change.
@dataclass(frozen=True)
class Preset:
    name: str
    url: str
    note: str


DEFAULT_SOURCES: list[Preset] = [
    Preset(
        "Lofi Girl - beats to relax/study",
        "https://www.youtube.com/watch?v=jfKfPfyJRdk",
        "Iconic 24/7 lo-fi hip-hop stream, no vocals, very calm.",
    ),
    Preset(
        "Ambient Sleeping Pill - ambient radio",
        "https://www.youtube.com/watch?v=S_MOd40zlYU",
        "Long-form spacious ambient, no beats, good for sleep/meditate.",
    ),
    Preset(
        "Chillhop - lofi hip hop radio",
        "https://www.youtube.com/watch?v=5yx6BWlEVcY",
        "Mellow instrumental lo-fi, great for focus sessions.",
    ),
    Preset(
        "Space Ambient - deep relaxation",
        "https://www.youtube.com/watch?v=tNkZsRW7h2c",
        "Slow evolving drones, ideal for the delta/theta modes.",
    ),
]


class DownloadError(RuntimeError):
    pass


def download_audio(
    url: str,
    out_dir: str | None = None,
    progress_hook=None,
    preferred_codec: str = "wav",
) -> tuple[str, str]:
    """Download the audio of ``url`` and return ``(local_path, title)``.

    Parameters
    ----------
    url:
        Any URL yt-dlp understands (single video; playlists are flattened to
        the first entry).
    out_dir:
        Destination directory. A temp dir is created when omitted.
    progress_hook:
        Optional yt-dlp progress hook ``callable(dict)``.
    preferred_codec:
        Codec for the extracted audio (``wav`` keeps loading dependency-free).
    """
    try:
        import yt_dlp
    except Exception as exc:  # pragma: no cover
        raise DownloadError(
            "yt-dlp is not installed. Run `pip install yt-dlp`."
        ) from exc

    out_dir = out_dir or tempfile.mkdtemp(prefix="neuralfm_")
    os.makedirs(out_dir, exist_ok=True)

    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": preferred_codec,
            }
        ],
    }
    if progress_hook is not None:
        opts["progress_hooks"] = [progress_hook]

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if "entries" in info and info["entries"]:
                info = info["entries"][0]
            title = info.get("title", "audio")
            vid = info.get("id", "audio")
    except Exception as exc:
        raise DownloadError(f"Download failed: {exc}") from exc

    # The post-processor rewrites the extension; find the resulting file.
    expected = os.path.join(out_dir, f"{vid}.{preferred_codec}")
    if os.path.exists(expected):
        return expected, title

    for fname in os.listdir(out_dir):
        if fname.startswith(vid):
            return os.path.join(out_dir, fname), title

    raise DownloadError("Download finished but no output file was found.")
