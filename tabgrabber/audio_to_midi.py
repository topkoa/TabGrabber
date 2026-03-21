"""Convert audio stems to MIDI using librosa for onset/pitch detection."""

import logging
from pathlib import Path

import librosa
import numpy as np
import pretty_midi

logger = logging.getLogger("tabgrabber")

# Instrument-specific frequency ranges (Hz)
INSTRUMENT_FREQ_RANGES = {
    "guitar": {
        "minimum_frequency": 82.0,    # E2 (lowest open string)
        "maximum_frequency": 1175.0,  # D6 (fret 22, high E string)
    },
    "bass": {
        "minimum_frequency": 41.0,    # E1 (lowest open string)
        "maximum_frequency": 400.0,   # ~G4 (fret 24, G string)
    },
}

# MIDI program numbers
INSTRUMENT_PROGRAMS = {
    "guitar": 25,   # Steel acoustic guitar
    "bass": 33,     # Electric bass (finger)
}

# Max simultaneous notes per instrument
INSTRUMENT_MAX_POLYPHONY = {
    "guitar": 6,    # Up to 6 strings
    "bass": 1,      # Mostly monophonic — take strongest pitch only
}


def convert_to_midi(
    audio_path: Path,
    output_dir: Path,
    instrument: str = "guitar",
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 58.0,
) -> Path:
    """
    Convert an audio stem to MIDI using librosa onset/pitch detection.

    Args:
        audio_path: Path to the stem WAV file.
        output_dir: Directory to write MIDI output.
        instrument: Instrument type for frequency filtering ('guitar' or 'bass').
        onset_threshold: Sensitivity for note onsets (0-1, higher = fewer notes).
        frame_threshold: Sensitivity for pitch confidence (0-1, higher = fewer notes).
        minimum_note_length: Minimum note duration in milliseconds.

    Returns:
        Path to the generated MIDI file.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    freq_range = INSTRUMENT_FREQ_RANGES.get(instrument, {})
    min_freq = freq_range.get("minimum_frequency", 40.0)
    max_freq = freq_range.get("maximum_frequency", 2000.0)

    logger.info(f"Converting {audio_path.name} to MIDI (instrument={instrument})")
    logger.debug(
        f"Parameters: onset={onset_threshold}, frame={frame_threshold}, "
        f"min_note={minimum_note_length}ms, freq_range={min_freq}-{max_freq}Hz"
    )

    # Load audio
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)

    # Detect onsets
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Map onset_threshold (0-1) to librosa's delta parameter
    # Higher threshold = fewer onsets (larger delta needed)
    onset_delta = onset_threshold * 0.15
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, onset_envelope=onset_env,
        delta=onset_delta, wait=int(sr * minimum_note_length / 1000 / 512),
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    if len(onset_times) == 0:
        logger.warning(f"No onsets detected in {audio_path.name}")
        # Write an empty MIDI file
        midi = pretty_midi.PrettyMIDI()
        midi.instruments.append(pretty_midi.Instrument(
            program=INSTRUMENT_PROGRAMS.get(instrument, 25)))
        output_path = output_dir / f"{audio_path.stem}.mid"
        midi.write(str(output_path))
        return output_path

    logger.debug(f"Detected {len(onset_times)} onsets")

    # Pitch tracking using piptrack (polyphonic capable)
    pitches, magnitudes = librosa.piptrack(
        y=y, sr=sr, fmin=min_freq, fmax=max_freq,
        threshold=frame_threshold,
    )

    # Build note events from onsets + pitch tracking
    max_polyphony = INSTRUMENT_MAX_POLYPHONY.get(instrument, 6)
    notes = _extract_notes(
        onset_times=onset_times,
        pitches=pitches,
        magnitudes=magnitudes,
        sr=sr,
        min_freq=min_freq,
        max_freq=max_freq,
        min_note_length=minimum_note_length / 1000.0,
        total_duration=librosa.get_duration(y=y, sr=sr),
        max_polyphony=max_polyphony,
    )

    logger.debug(f"Extracted {len(notes)} note events")

    # Build MIDI
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(
        program=INSTRUMENT_PROGRAMS.get(instrument, 25),
        name=instrument,
    )

    for note_start, note_end, pitch, velocity in notes:
        midi_note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=note_start,
            end=note_end,
        )
        inst.notes.append(midi_note)

    midi.instruments.append(inst)

    output_path = output_dir / f"{audio_path.stem}.mid"
    midi.write(str(output_path))

    logger.info(f"Generated MIDI: {output_path.name} ({len(notes)} notes)")
    return output_path


def _extract_notes(
    onset_times: np.ndarray,
    pitches: np.ndarray,
    magnitudes: np.ndarray,
    sr: int,
    min_freq: float,
    max_freq: float,
    min_note_length: float,
    total_duration: float,
    max_polyphony: int = 6,
) -> list[tuple[float, float, int, int]]:
    """
    Extract note events by combining onset times with pitch data.

    Args:
        max_polyphony: Maximum simultaneous notes per onset (1 for bass, 6 for guitar).

    Returns list of (start_time, end_time, midi_pitch, velocity) tuples.
    """
    hop_length = 512
    notes = []

    for i, onset_time in enumerate(onset_times):
        # Determine note end time (next onset or a maximum duration)
        if i + 1 < len(onset_times):
            next_onset = onset_times[i + 1]
        else:
            next_onset = total_duration

        # Get the frame index for this onset
        frame_idx = librosa.time_to_frames(onset_time, sr=sr, hop_length=hop_length)

        if frame_idx >= pitches.shape[1]:
            continue

        # Look at a small window of frames around the onset to find pitches
        window_end = min(frame_idx + 4, pitches.shape[1])

        # Collect pitches from frames in this window
        detected_pitches = set()
        best_magnitudes = {}

        for f_idx in range(frame_idx, window_end):
            frame_pitches = pitches[:, f_idx]
            frame_mags = magnitudes[:, f_idx]

            # Get bins with significant magnitude
            nonzero = frame_mags > 0
            if not np.any(nonzero):
                continue

            for bin_idx in np.where(nonzero)[0]:
                freq = frame_pitches[bin_idx]
                mag = frame_mags[bin_idx]

                if freq < min_freq or freq > max_freq:
                    continue

                midi_pitch = int(round(librosa.hz_to_midi(freq)))
                if midi_pitch < 0 or midi_pitch > 127:
                    continue

                if midi_pitch not in best_magnitudes or mag > best_magnitudes[midi_pitch]:
                    best_magnitudes[midi_pitch] = mag
                    detected_pitches.add(midi_pitch)

        if not detected_pitches:
            continue

        # Sort by magnitude and take top N pitches (avoid noise)
        sorted_pitches = sorted(
            detected_pitches,
            key=lambda p: best_magnitudes.get(p, 0),
            reverse=True,
        )[:max_polyphony]

        # Determine note duration
        note_end = min(next_onset, onset_time + 2.0)  # cap at 2 seconds
        note_duration = note_end - onset_time

        if note_duration < min_note_length:
            continue

        for midi_pitch in sorted_pitches:
            # Scale magnitude to MIDI velocity (0-127)
            mag = best_magnitudes[midi_pitch]
            velocity = int(np.clip(mag * 127 * 3, 40, 127))

            notes.append((onset_time, note_end, midi_pitch, velocity))

    return notes
