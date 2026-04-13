"""Karaoke-style lyrics extraction from a vocal stem via WhisperX.

WhisperX = Whisper transcription + wav2vec2 forced alignment for phoneme-level
word timing. Much tighter sync than plain Whisper's `word_timestamps=True` on
sung vocals, at the cost of a larger install and an extra alignment model.

Output format matches slopsmith's lyrics schema (`Song.lyrics` in lib/song.py):
    [{"t": start_seconds, "d": duration_seconds, "w": word}, ...]
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

from tabgrabber.utils import get_device

logger = logging.getLogger("tabgrabber")


def _pick_compute_type(device: str) -> str:
    """float16 on CUDA, int8 on CPU — WhisperX's recommended defaults."""
    return "float16" if device == "cuda" else "int8"


def extract_lyrics(
    vocal_path: Path,
    device: str = "auto",
    model_size: str = "large-v2",
    language: str | None = None,
    compute_type: str | None = None,
) -> list[dict]:
    """Transcribe a vocal stem and return word-level lyrics in slopsmith format.

    Args:
        vocal_path: Path to the isolated vocal stem (Demucs output).
        device: "auto", "cuda", or "cpu".
        model_size: Whisper size (tiny/base/small/medium/large-v2/large-v3).
        language: ISO language code to force, or None to autodetect.
        compute_type: CTranslate2 compute type; None for auto.

    Returns:
        List of `{"t", "d", "w"}` entries, one per word, sorted by start time.
    """
    import whisperx  # heavy import — deferred so non-lyrics runs stay light

    vocal_path = Path(vocal_path)
    resolved_device = get_device(device)
    ct = compute_type or _pick_compute_type(resolved_device)

    logger.info(
        f"Loading WhisperX model '{model_size}' on {resolved_device} "
        f"(compute_type={ct})"
    )
    model = whisperx.load_model(model_size, resolved_device, compute_type=ct)

    logger.info(f"Transcribing {vocal_path.name}...")
    audio = whisperx.load_audio(str(vocal_path))
    result = model.transcribe(audio, language=language)
    detected_lang = result.get("language", language or "en")
    logger.info(f"Detected language: {detected_lang}")

    logger.info("Loading alignment model for forced word-level alignment...")
    align_model, metadata = whisperx.load_align_model(
        language_code=detected_lang, device=resolved_device
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        resolved_device,
        return_char_alignments=False,
    )

    words: list[dict] = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            start = w.get("start")
            end = w.get("end")
            text = (w.get("word") or "").strip()
            if start is None or end is None or not text:
                continue
            duration = max(0.05, float(end) - float(start))
            words.append(
                {"t": round(float(start), 3), "d": round(duration, 3), "w": text}
            )

    words.sort(key=lambda e: e["t"])
    logger.info(f"Aligned {len(words)} word entries")

    # Free model handles — pipeline continues with song analysis after this.
    del model, align_model
    gc.collect()
    if resolved_device == "cuda":
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    return words
