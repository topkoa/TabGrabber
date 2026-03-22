"""Command-line interface for TabGrabber."""

import argparse
import sys
from pathlib import Path

from tabgrabber import __version__
from tabgrabber.pipeline import PipelineOptions, QUALITY_PRESETS, process
from tabgrabber.utils import setup_logging, validate_audio_file


def parse_tuning(tuning_str: str) -> list[int]:
    """
    Parse a tuning string into MIDI note numbers.

    Accepts either:
    - Comma-separated MIDI note numbers: "40,45,50,55,59,64"
    - Comma-separated note names: "E2,A2,D3,G3,B3,E4"
    """
    NOTE_MAP = {
        "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
    }

    parts = [p.strip() for p in tuning_str.split(",")]
    midi_notes = []

    for part in parts:
        # Try as integer first
        try:
            midi_notes.append(int(part))
            continue
        except ValueError:
            pass

        # Try as note name (e.g., "E2", "A#3", "Bb4")
        if len(part) >= 2:
            note_name = part[0].upper()
            rest = part[1:]

            semitone_offset = 0
            if rest and rest[0] == "#":
                semitone_offset = 1
                rest = rest[1:]
            elif rest and rest[0].lower() == "b":
                semitone_offset = -1
                rest = rest[1:]

            if note_name in NOTE_MAP:
                try:
                    octave = int(rest)
                    midi = (octave + 1) * 12 + NOTE_MAP[note_name] + semitone_offset
                    midi_notes.append(midi)
                    continue
                except ValueError:
                    pass

        raise ValueError(f"Cannot parse tuning value: '{part}'")

    return midi_notes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tabgrabber",
        description="Extract guitar and bass tabs from audio files using AI.",
    )
    parser.add_argument("input_file", type=Path, help="Path to audio file (WAV, MP3, FLAC, OGG)")
    parser.add_argument("output_dir", type=Path, help="Directory for output files")

    parser.add_argument("--model", default="htdemucs_6s",
                        help="Demucs model name (default: htdemucs_6s)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Processing device (default: auto)")
    parser.add_argument("--instruments", nargs="+", default=["guitar", "bass"],
                        choices=["guitar", "bass"],
                        help="Instruments to extract (default: guitar bass)")
    parser.add_argument("--format", dest="formats", nargs="+",
                        default=["ascii"],
                        choices=["ascii", "gp5", "musicxml", "all"],
                        help="Output tab format(s) (default: ascii)")
    parser.add_argument("--tuning", type=str, default=None,
                        help="Custom tuning as comma-separated notes, e.g. 'E2,A2,D3,G3,B3,E4' or '40,45,50,55,59,64'")
    parser.add_argument("--onset-threshold", type=float, default=0.5,
                        help="Onset detection sensitivity 0-1 (default: 0.5)")
    parser.add_argument("--frame-threshold", type=float, default=0.3,
                        help="Pitch confidence threshold 0-1 (default: 0.3)")
    parser.add_argument("--keep-intermediates", action="store_true", default=True,
                        help="Keep stem WAVs and MIDI files (default: true)")

    # Quality presets and advanced parameters
    quality_group = parser.add_argument_group("quality options")
    quality_group.add_argument("--quality", default="fast",
                               choices=["fast", "balanced", "high", "extreme"],
                               help="Quality preset (default: fast)")
    quality_group.add_argument("--demucs-shifts", type=int, default=None,
                               help="Demucs random shifts for augmentation (overrides preset)")
    quality_group.add_argument("--demucs-overlap", type=float, default=None,
                               help="Demucs segment overlap 0.0-0.99 (overrides preset)")
    quality_group.add_argument("--hop-length", type=int, default=None,
                               choices=[128, 256, 512, 1024],
                               help="Analysis hop length (overrides preset)")
    quality_group.add_argument("--n-fft", type=int, default=None,
                               choices=[1024, 2048, 4096, 8192],
                               help="FFT window size (overrides preset)")
    quality_group.add_argument("--onset-window", type=int, default=None,
                               help="Frames to analyze around each onset (overrides preset)")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    # Validate input
    try:
        validate_audio_file(args.input_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse tuning if provided
    tuning = None
    if args.tuning:
        try:
            tuning = parse_tuning(args.tuning)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Build quality overrides (only include explicitly set values)
    quality_overrides = {
        "model": args.model,
        "device": args.device,
        "instruments": args.instruments,
        "formats": args.formats,
        "tuning": tuning,
        "onset_threshold": args.onset_threshold,
        "frame_threshold": args.frame_threshold,
        "keep_intermediates": args.keep_intermediates,
    }
    if args.demucs_shifts is not None:
        quality_overrides["demucs_shifts"] = args.demucs_shifts
    if args.demucs_overlap is not None:
        quality_overrides["demucs_overlap"] = args.demucs_overlap
    if args.hop_length is not None:
        quality_overrides["hop_length"] = args.hop_length
    if args.n_fft is not None:
        quality_overrides["n_fft"] = args.n_fft
    if args.onset_window is not None:
        quality_overrides["onset_window"] = args.onset_window

    opts = PipelineOptions.from_preset(args.quality, **quality_overrides)

    try:
        result = process(args.input_file, args.output_dir, opts)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    # Print output paths
    for instrument, paths in result.tabs.items():
        for p in paths:
            print(p)
