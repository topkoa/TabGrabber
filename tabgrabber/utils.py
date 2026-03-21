"""Utility functions for device detection and logging setup."""

import logging
import sys


def get_device(device: str = "auto") -> str:
    """Determine the best device for ML processing."""
    if device == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for TabGrabber."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger = logging.getLogger("tabgrabber")
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


def validate_audio_file(path) -> None:
    """Raise ValueError if path is not a supported audio file."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    if p.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{p.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
        )
