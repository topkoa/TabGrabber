"""Pipeline orchestration: audio → stems → MIDI → tablature."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from tabgrabber.stems import extract_stems
from tabgrabber.audio_to_midi import convert_to_midi
from tabgrabber.midi_to_tab import midi_to_tab_notes
from tabgrabber.tab_formats.ascii_tab import write_ascii_tab
from tabgrabber.tab_formats.guitar_pro import write_guitar_pro
from tabgrabber.tab_formats.musicxml import write_musicxml
from tabgrabber.song_analysis import analyze_song, write_analysis_report
from tabgrabber.utils import get_device

logger = logging.getLogger("tabgrabber")


QUALITY_PRESETS = {
    "fast": {
        "demucs_shifts": 0,
        "demucs_overlap": 0.25,
        "demucs_segment": None,  # model default
        "hop_length": 512,
        "n_fft": 2048,
        "onset_window": 4,
    },
    "balanced": {
        "demucs_shifts": 1,
        "demucs_overlap": 0.25,
        "demucs_segment": None,
        "hop_length": 512,
        "n_fft": 2048,
        "onset_window": 6,
    },
    "high": {
        "demucs_shifts": 3,
        "demucs_overlap": 0.5,
        "demucs_segment": None,
        "hop_length": 256,
        "n_fft": 4096,
        "onset_window": 8,
    },
    "extreme": {
        "demucs_shifts": 5,
        "demucs_overlap": 0.75,
        "demucs_segment": 40,       # longer segments = more context for separation
        "hop_length": 128,
        "n_fft": 8192,
        "onset_window": 12,
    },
}


@dataclass
class PipelineOptions:
    """Configuration for the processing pipeline."""
    model: str = "htdemucs_6s"
    device: str = "auto"
    instruments: list[str] = field(default_factory=lambda: ["guitar", "bass"])
    formats: list[str] = field(default_factory=lambda: ["ascii"])
    tuning: list[int] | None = None
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    keep_intermediates: bool = True
    invert_strings: bool = False
    # Quality parameters — Demucs
    demucs_shifts: int = 0
    demucs_overlap: float = 0.25
    demucs_segment: int | None = None  # None = model default
    # Quality parameters — audio-to-MIDI
    hop_length: int = 512
    n_fft: int = 2048
    onset_window: int = 4

    @classmethod
    def from_preset(cls, preset: str = "fast", **overrides) -> "PipelineOptions":
        """Create options from a quality preset with optional overrides."""
        params = dict(QUALITY_PRESETS.get(preset, QUALITY_PRESETS["fast"]))
        params.update(overrides)
        return cls(**params)


@dataclass
class PipelineResult:
    """Paths to all generated files."""
    stems: dict[str, Path] = field(default_factory=dict)
    midi: dict[str, Path] = field(default_factory=dict)
    tabs: dict[str, list[Path]] = field(default_factory=dict)
    backing_track: Path | None = None
    analysis_report: Path | None = None


def process(
    input_path: Path,
    output_dir: Path,
    opts: PipelineOptions | None = None,
) -> PipelineResult:
    """
    Run the full pipeline: audio → stems → MIDI → tablature.

    Args:
        input_path: Path to the input audio file.
        output_dir: Root directory for all output files.
        opts: Pipeline configuration options.

    Returns:
        PipelineResult with paths to all generated files.
    """
    if opts is None:
        opts = PipelineOptions()

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    result = PipelineResult()

    device = get_device(opts.device)
    logger.info(f"Processing: {input_path.name} (device={device})")

    # Step 1: Extract stems
    logger.info("Step 1/4: Extracting stems with Demucs...")
    stems_dir = output_dir / "stems"
    # Extract ALL stems so we can build a backing track
    all_stem_paths = extract_stems(
        audio_path=input_path,
        output_dir=stems_dir,
        model=opts.model,
        device=device,
        stems=None,  # get all stems
        shifts=opts.demucs_shifts,
        overlap=opts.demucs_overlap,
        segment=opts.demucs_segment,
    )
    # Filter to just the requested instruments for MIDI conversion
    stem_paths = {k: v for k, v in all_stem_paths.items() if k in opts.instruments}
    result.stems = stem_paths

    # Build backing track (everything except the extracted instruments)
    backing_stems = {k: v for k, v in all_stem_paths.items() if k not in opts.instruments}
    if backing_stems:
        try:
            backing_path = _create_backing_track(backing_stems, output_dir / "backing_track.wav")
            result.backing_track = backing_path
            logger.info(f"Created backing track from: {', '.join(backing_stems.keys())}")
        except Exception as e:
            logger.warning(f"Could not create backing track: {e}")

    # Step 2: Convert stems to MIDI
    logger.info("Step 2/4: Converting stems to MIDI...")
    midi_dir = output_dir / "midi"
    for instrument in opts.instruments:
        if instrument not in stem_paths:
            logger.warning(f"Stem '{instrument}' not available, skipping MIDI conversion")
            continue

        try:
            midi_path = convert_to_midi(
                audio_path=stem_paths[instrument],
                output_dir=midi_dir,
                instrument=instrument,
                onset_threshold=opts.onset_threshold,
                frame_threshold=opts.frame_threshold,
                hop_length=opts.hop_length,
                n_fft=opts.n_fft,
                onset_window=opts.onset_window,
            )
            result.midi[instrument] = midi_path
        except Exception as e:
            logger.error(f"MIDI conversion failed for {instrument}: {e}")

    # Step 3: Generate tablature
    logger.info("Step 3/4: Generating tablature...")
    tabs_dir = output_dir / "tabs"
    title = input_path.stem.replace("_", " ").replace("-", " ").title()

    for instrument, midi_path in result.midi.items():
        try:
            tab_notes, tempo, config = midi_to_tab_notes(
                midi_path=midi_path,
                instrument=instrument,
                tuning=opts.tuning,
            )

            tab_paths: list[Path] = []

            for fmt in opts.formats:
                if fmt == "ascii":
                    out = tabs_dir / f"{instrument}_tab.txt"
                    write_ascii_tab(tab_notes, config, out, tempo, title,
                                            invert_strings=opts.invert_strings)
                    tab_paths.append(out)

                elif fmt == "gp5":
                    out = tabs_dir / f"{instrument}_tab.gp5"
                    write_guitar_pro(tab_notes, config, out, tempo, title)
                    tab_paths.append(out)

                elif fmt == "musicxml":
                    out = tabs_dir / f"{instrument}_tab.xml"
                    write_musicxml(tab_notes, config, out, tempo, title)
                    tab_paths.append(out)

                elif fmt == "all":
                    for sub_fmt in ["ascii", "gp5", "musicxml"]:
                        # Recurse with individual format
                        if sub_fmt not in opts.formats:
                            opts_copy = PipelineOptions(**{
                                **vars(opts),
                                "formats": [sub_fmt],
                            })
                            # Just call the writers directly
                            if sub_fmt == "ascii":
                                out = tabs_dir / f"{instrument}_tab.txt"
                                write_ascii_tab(tab_notes, config, out, tempo, title,
                                            invert_strings=opts.invert_strings)
                            elif sub_fmt == "gp5":
                                out = tabs_dir / f"{instrument}_tab.gp5"
                                write_guitar_pro(tab_notes, config, out, tempo, title)
                            elif sub_fmt == "musicxml":
                                out = tabs_dir / f"{instrument}_tab.xml"
                                write_musicxml(tab_notes, config, out, tempo, title)
                            tab_paths.append(out)

            result.tabs[instrument] = tab_paths

        except Exception as e:
            logger.error(f"Tab generation failed for {instrument}: {e}")

    # Step 4: Song analysis
    logger.info("Step 4: Analyzing song structure...")
    try:
        analysis = analyze_song(input_path)
        report_path = output_dir / "song_analysis.txt"
        write_analysis_report(analysis, report_path)
        result.analysis_report = report_path
    except Exception as e:
        logger.warning(f"Song analysis failed: {e}")

    logger.info("Processing complete!")
    _log_summary(result)
    return result


def _create_backing_track(stems: dict[str, Path], output_path: Path) -> Path:
    """Mix non-instrument stems into a single backing track WAV."""
    import soundfile as sf
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)

    mixed = None
    sample_rate = None

    for name, stem_path in stems.items():
        data, sr = sf.read(str(stem_path))
        if mixed is None:
            mixed = data.astype(np.float64)
            sample_rate = sr
        else:
            # Match lengths (pad shorter with zeros)
            if len(data) > len(mixed):
                pad = np.zeros((len(data) - len(mixed), mixed.shape[1] if mixed.ndim > 1 else 1))
                if mixed.ndim == 1:
                    pad = pad.flatten()
                mixed = np.concatenate([mixed, pad])
            elif len(data) < len(mixed):
                pad = np.zeros((len(mixed) - len(data), data.shape[1] if data.ndim > 1 else 1))
                if data.ndim == 1:
                    pad = pad.flatten()
                data = np.concatenate([data, pad])
            mixed += data.astype(np.float64)

    if mixed is None:
        raise ValueError("No stems to mix")

    # Normalize to prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak * 0.95

    sf.write(str(output_path), mixed.astype(np.float32), sample_rate)
    logger.info(f"Wrote backing track: {output_path}")
    return output_path


def _log_summary(result: PipelineResult) -> None:
    """Log a summary of generated files."""
    logger.info("--- Output Summary ---")
    for instrument, paths in result.tabs.items():
        logger.info(f"  {instrument}:")
        for p in paths:
            logger.info(f"    {p}")
    if result.midi:
        logger.info("  MIDI files:")
        for inst, p in result.midi.items():
            logger.info(f"    {p}")
    if result.backing_track:
        logger.info(f"  Backing track: {result.backing_track}")
    if result.analysis_report:
        logger.info(f"  Song analysis: {result.analysis_report}")
