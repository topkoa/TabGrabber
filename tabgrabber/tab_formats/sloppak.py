"""Sloppak writer — packages a TabGrabber run as a slopsmith `.sloppak` song.

A `.sloppak` is an open song package consumed by slopsmith
(https://github.com/byrongamatos/slopsmith). See PR #7 for the format spec.
This writer emits the zip form by default, or a directory form if `as_dir=True`.

Layout:
    <name>.sloppak/
      manifest.yaml
      stems/<id>.ogg          # one per demucs stem
      arrangements/<name>.json  # slopsmith highway wire format
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import yaml

from tabgrabber.midi_to_tab import InstrumentConfig, TabNote

logger = logging.getLogger("tabgrabber")


# Slopsmith stores tunings as per-string offsets from E-standard.
# Guitar reference: E2 A2 D3 G3 B3 E4  -> [40,45,50,55,59,64]
# Bass   reference: E1 A1 D2 G2        -> [28,33,38,43]
_GUITAR_STD = [40, 45, 50, 55, 59, 64]
_BASS_STD = [28, 33, 38, 43]


def _tuning_offsets(config: InstrumentConfig) -> list[int]:
    n = config.num_strings
    ref = _BASS_STD if n == 4 else _GUITAR_STD
    return [config.tuning[i] - ref[i] for i in range(n)]


def _arrangement_name(instrument: str) -> str:
    return {"guitar": "Lead", "bass": "Bass"}.get(instrument, instrument.title())


def _tab_notes_to_wire(
    tab_notes: list[TabNote],
    config: InstrumentConfig,
    instrument: str,
) -> dict:
    """Serialize to the slopsmith highway wire JSON shape.

    Mirrors `arrangement_to_wire` in slopsmith's lib/song.py. TabGrabber does
    not detect effects (bends, hammer-ons, palm mute, ...), so those fields
    stay at their zero/false defaults — slopsmith's loader tolerates missing
    keys.
    """
    notes = [
        {
            "t": round(n.time, 3),
            "s": n.string,
            "f": n.fret,
            "sus": round(n.duration, 3),
            "sl": -1,
            "slu": -1,
            "bn": 0,
            "ho": False,
            "po": False,
            "hm": False,
            "hp": False,
            "pm": False,
            "mt": False,
            "tr": False,
            "ac": False,
            "tp": False,
        }
        for n in tab_notes
    ]
    return {
        "name": _arrangement_name(instrument),
        "tuning": _tuning_offsets(config),
        "capo": 0,
        "notes": notes,
        "chords": [],
        "anchors": [],
        "handshapes": [],
        "templates": [],
    }


def _encode_ogg(src: Path, dst: Path) -> bool:
    """Transcode any audio file to OGG Vorbis via ffmpeg. Returns True on success."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-c:a", "libvorbis", "-q:a", "5", str(dst)],
            capture_output=True,
        )
    except FileNotFoundError:
        logger.error("ffmpeg not found on PATH — cannot encode sloppak stems")
        return False
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size < 100:
        logger.warning(f"ffmpeg failed for {src}: {r.stderr.decode(errors='replace')[:200]}")
        return False
    return True


def _parse_artist_title(input_stem: str) -> tuple[str, str]:
    """'Artist - Title' → ('Artist', 'Title'); otherwise ('', cleaned stem)."""
    if " - " in input_stem:
        left, right = input_stem.split(" - ", 1)
        return left.strip(), right.strip()
    return "", input_stem.replace("_", " ").replace("-", " ").strip()


def write_sloppak(
    out_path: Path,
    input_stem: str,
    duration: float,
    tab_data: dict[str, tuple[list[TabNote], InstrumentConfig]],
    stem_paths: dict[str, Path],
    as_dir: bool = False,
    lyrics: list[dict] | None = None,
) -> Path:
    """Assemble a sloppak package at `out_path`.

    Args:
        out_path: Target path. For zip form, ends in `.sloppak`. For dir form,
            a directory of that name is created.
        input_stem: Original input filename stem (used for Artist/Title parsing).
        duration: Song duration in seconds.
        tab_data: {instrument: (tab_notes, config)} for each transcribed instrument.
        stem_paths: {stem_id: path_to_wav} for every demucs stem to include.
        as_dir: True to emit the directory form; False to emit a zip file.

    Returns:
        Path to the created `.sloppak` (file or directory).
    """
    artist, title = _parse_artist_title(input_stem)

    # Build manifest structure.
    arrangements_manifest = []
    arrangement_files: dict[str, dict] = {}
    for instrument, (tab_notes, config) in tab_data.items():
        wire = _tab_notes_to_wire(tab_notes, config, instrument)
        rel = f"arrangements/{instrument}.json"
        arrangement_files[rel] = wire
        arrangements_manifest.append(
            {
                "name": wire["name"],
                "file": rel,
                "tuning": wire["tuning"],
                "capo": 0,
            }
        )

    stems_manifest = []
    for sid in stem_paths:
        stems_manifest.append(
            {
                "id": sid,
                "file": f"stems/{sid}.ogg",
                "default": True,
            }
        )

    manifest = {
        "title": title,
        "artist": artist,
        "album": "",
        "year": 0,
        "duration": round(float(duration), 3),
        "stems": stems_manifest,
        "arrangements": arrangements_manifest,
    }
    if lyrics:
        manifest["lyrics"] = "lyrics.json"

    # Stage all files in a temp directory, then either copy to out_path (dir
    # form) or zip them up (file form).
    with tempfile.TemporaryDirectory(prefix="tabgrabber_sloppak_") as td:
        staging = Path(td)
        (staging / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        for rel, wire in arrangement_files.items():
            p = staging / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(wire, separators=(",", ":")), encoding="utf-8")

        if lyrics:
            (staging / "lyrics.json").write_text(
                json.dumps(lyrics, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )

        for sid, src in stem_paths.items():
            dst = staging / "stems" / f"{sid}.ogg"
            if not _encode_ogg(Path(src), dst):
                logger.warning(f"Skipping stem '{sid}' — encode failed")

        out_path = Path(out_path)
        if as_dir:
            if out_path.exists():
                shutil.rmtree(out_path, ignore_errors=True)
            shutil.copytree(staging, out_path)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                out_path.unlink()
            with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(staging.rglob("*")):
                    if f.is_file():
                        zf.write(f, f.relative_to(staging).as_posix())

    logger.info(f"Wrote sloppak: {out_path}")
    return out_path
