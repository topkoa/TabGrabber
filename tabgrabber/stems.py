"""Demucs-based audio stem extraction.

Adapted from RocksmithGuitarMute's separate_stems() implementation.
"""

import contextlib
import io
import logging
import sys
from pathlib import Path

logger = logging.getLogger("tabgrabber")

DEFAULT_MODEL = "htdemucs_6s"


def extract_stems(
    audio_path: Path,
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    device: str = "cpu",
    stems: list[str] | None = None,
    shifts: int = 0,
    overlap: float = 0.25,
    segment: int | None = None,
) -> dict[str, Path]:
    """
    Run Demucs source separation on an audio file.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Directory for Demucs output.
        model: Demucs model name (default: htdemucs_6s for 6-stem separation).
        device: Processing device ('cpu', 'cuda', or 'auto').
        stems: Which stems to return. Default: all available.
        shifts: Number of random shifts for test-time augmentation (0=off, higher=better quality).
        overlap: Overlap between segments (0.0-0.99, higher=smoother boundaries).
        segment: Segment length in seconds (None=model default, larger=more context).

    Returns:
        Dict mapping stem name to WAV file path.
    """
    import demucs.separate

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_info = f"shifts={shifts}, overlap={overlap}"
    if segment:
        quality_info += f", segment={segment}"
    logger.info(f"Running Demucs separation ({model}, {quality_info}) on: {audio_path.name}")

    args = [
        "--name", model,
        "--device", device,
        "--out", str(output_dir),
        "--shifts", str(shifts),
        "--overlap", str(overlap),
    ]
    if segment is not None:
        args.extend(["--segment", str(segment)])
    args.append(str(audio_path))

    logger.debug(f"Demucs args: {args}")

    # Ensure stdout/stderr exist (can be None in frozen apps)
    if sys.stdout is None:
        sys.stdout = io.StringIO()
    if sys.stderr is None:
        sys.stderr = io.StringIO()

    # Capture output to prevent console noise
    captured_out = io.StringIO()
    captured_err = io.StringIO()

    with contextlib.redirect_stdout(captured_out), contextlib.redirect_stderr(captured_err):
        try:
            demucs.separate.main(args)
        except SystemExit as e:
            if e.code != 0:
                logger.error(f"Demucs failed (exit code {e.code})")
                logger.error(f"Demucs stderr: {captured_err.getvalue()}")
                raise RuntimeError(f"Demucs processing failed with exit code {e.code}")

    # Log captured output at debug level
    out_str = captured_out.getvalue()
    err_str = captured_err.getvalue()
    if out_str:
        logger.debug(f"Demucs stdout: {out_str}")
    if err_str:
        logger.debug(f"Demucs stderr: {err_str}")

    # Discover separated stem files: output_dir / model_name / audio_stem / *.wav
    stems_dir = output_dir / model / audio_path.stem

    if not stems_dir.exists():
        raise FileNotFoundError(f"Demucs output directory not found: {stems_dir}")

    found_stems: dict[str, Path] = {}
    for stem_file in stems_dir.glob("*.wav"):
        found_stems[stem_file.stem] = stem_file

    if not found_stems:
        raise FileNotFoundError(f"No stem WAV files found in: {stems_dir}")

    logger.info(f"Separated into {len(found_stems)} stems: {', '.join(sorted(found_stems.keys()))}")

    # Filter to requested stems if specified
    if stems is not None:
        filtered = {}
        for name in stems:
            if name in found_stems:
                filtered[name] = found_stems[name]
            else:
                logger.warning(f"Requested stem '{name}' not found in Demucs output")
        return filtered

    return found_stems
