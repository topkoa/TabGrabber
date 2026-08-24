"""Build a chord sheet: timed lyrics with the chords placed above them.

Chords come from :func:`tabgrabber.song_analysis.analyze_song`, which already
smooths its per-beat detection into bar-length chords. Lyrics come from Whisper
running over the isolated vocal stem, which is far more accurate than running
it over the full mix.

The two are merged onto one timeline so a player can highlight the word being
sung and show each chord above the word where it actually falls, the way a
printed chord sheet does.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tabgrabber.song_analysis import SongAnalysis

logger = logging.getLogger("tabgrabber")

# A silence longer than this between sung lines is an instrumental break worth
# printing chords for on its own row.
INSTRUMENTAL_GAP = 4.0
# A chord starting just before a line still belongs to that line.
CHORD_LEAD_IN = 0.35

DEFAULT_LYRICS_MODEL = "large-v3"


def transcribe_words(
    vocals_path: Path,
    model_size: str = DEFAULT_LYRICS_MODEL,
    device: str = "auto",
    language: str | None = None,
) -> list[dict]:
    """Transcribe a vocal stem into lines of word-level timings.

    Returns a list of ``{"t", "end", "words": [{"t", "d", "w"}]}`` entries,
    with every time in seconds. Requires the ``lyrics`` extra.
    """
    try:
        import torch
        import whisper
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Lyrics transcription needs the 'lyrics' extra: "
            "pip install 'tabgrabber[lyrics]'"
        ) from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading Whisper model '{model_size}' on {device} "
                f"(the first run downloads the weights)")
    model = whisper.load_model(model_size, device=device)

    logger.info("Transcribing vocal stem")
    try:
        result = model.transcribe(
            str(vocals_path),
            word_timestamps=True,
            language=language,
            fp16=(device == "cuda"),
            verbose=False,
        )
    finally:
        del model
        if device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    lines: list[dict] = []
    for segment in result.get("segments", []):
        words = []
        for word in segment.get("words") or []:
            text = str(word.get("word", "")).strip()
            if not text:
                continue
            start = float(word.get("start", segment.get("start", 0.0)))
            stop = float(word.get("end", start))
            words.append({
                "t": round(start, 3),
                "d": round(max(stop - start, 0.05), 3),
                "w": text,
            })
        if words:
            lines.append({
                "t": round(float(segment.get("start", words[0]["t"])), 3),
                "end": round(float(segment.get(
                    "end", words[-1]["t"] + words[-1]["d"])), 3),
                "words": words,
            })

    logger.info(f"Transcribed {len(lines)} lines "
                f"({sum(len(ln['words']) for ln in lines)} words), "
                f"language={result.get('language', '?')}")
    return lines


def _place_chord(chord_time: float, words: list[dict]) -> int:
    """Index of the word a chord should sit above (-1 means before the line)."""
    if not words:
        return -1
    for i, word in enumerate(words):
        if word["t"] <= chord_time < word["t"] + word["d"]:
            return i
    for i, word in enumerate(words):
        if word["t"] >= chord_time:
            if i > 0 or chord_time >= words[0]["t"] - CHORD_LEAD_IN:
                return i
            return -1
    return len(words) - 1


def build_chordsheet(
    analysis: SongAnalysis,
    vocals_path: Path | None = None,
    model_size: str = DEFAULT_LYRICS_MODEL,
    device: str = "auto",
    language: str | None = None,
) -> dict[str, Any]:
    """Merge an analysis and a transcribed vocal stem into one chord sheet.

    Passing ``vocals_path=None`` (or a stem that cannot be transcribed) yields
    a chords-only sheet rather than failing.
    """
    chords = [
        {"t": round(c.time, 3), "d": round(c.duration, 3), "name": c.chord}
        for c in analysis.chords
    ]
    duration = float(analysis.duration or 0.0)

    transcript: list[dict] = []
    if vocals_path and Path(vocals_path).exists():
        try:
            transcript = transcribe_words(
                Path(vocals_path), model_size, device, language)
        except Exception as exc:  # noqa: BLE001 - lyrics are best-effort
            logger.error(f"Lyrics transcription failed, "
                         f"falling back to chords only: {exc}")
    else:
        logger.info("No vocal stem given, building a chords-only sheet")

    def chords_between(low: float, high: float) -> list[dict]:
        return [c for c in chords if low <= c["t"] < high]

    def sounding_at(when: float) -> dict | None:
        """Chord still ringing at a time, having started on an earlier line."""
        held = [c for c in chords if c["t"] <= when < c["t"] + c["d"] + 0.5]
        return held[-1] if held else None

    def instrumental_row(row_chords: list[dict], end: float) -> dict:
        return {
            "t": round(row_chords[0]["t"], 3),
            "end": round(end, 3),
            "instrumental": True,
            "words": [],
            "chords": [{"t": c["t"], "name": c["name"], "at": -1}
                       for c in row_chords],
        }

    lines: list[dict] = []
    previous_end = 0.0

    for segment in transcript:
        gap_chords = chords_between(previous_end, segment["t"] - CHORD_LEAD_IN)
        if gap_chords and (segment["t"] - previous_end) >= INSTRUMENTAL_GAP:
            lines.append(instrumental_row(gap_chords, segment["t"]))

        placed = [
            {"t": c["t"], "name": c["name"],
             "at": _place_chord(c["t"], segment["words"])}
            for c in chords_between(segment["t"] - CHORD_LEAD_IN, segment["end"])
        ]

        # A chord held over from the previous line still has to be shown, or
        # the line looks like it has no chord at all.
        if not placed or placed[0]["at"] > 0:
            held = sounding_at(segment["t"])
            if held and (not placed or held["name"] != placed[0]["name"]):
                placed.insert(
                    0, {"t": segment["t"], "name": held["name"], "at": 0})

        lines.append({
            "t": segment["t"],
            "end": segment["end"],
            "instrumental": False,
            "words": segment["words"],
            "chords": placed,
        })
        previous_end = segment["end"]

    # Trailing instrumental, or the whole song when there are no lyrics at all.
    tail = chords_between(previous_end, duration + 1.0)
    if tail:
        lines.append(instrumental_row(
            tail, duration or tail[-1]["t"] + tail[-1]["d"]))

    return {
        "title": analysis.title,
        "key": analysis.key,
        "tempo": round(float(analysis.tempo or 0.0), 1),
        "time_signature": analysis.time_signature,
        "duration": round(duration, 3),
        "sections": [
            {"label": s.label, "start": round(s.start, 3), "end": round(s.end, 3)}
            for s in analysis.sections
        ],
        "chords": chords,
        "lines": lines,
        "has_lyrics": bool(transcript),
    }
