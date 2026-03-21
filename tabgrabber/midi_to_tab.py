"""Convert MIDI note data to guitar/bass tablature with fret assignment."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pretty_midi

logger = logging.getLogger("tabgrabber")

# Standard tunings as MIDI note numbers (low string to high string)
TUNINGS = {
    "guitar_standard": [40, 45, 50, 55, 59, 64],   # E2 A2 D3 G3 B3 E4
    "guitar_drop_d":   [38, 45, 50, 55, 59, 64],   # D2 A2 D3 G3 B3 E4
    "bass_standard":   [28, 33, 38, 43],             # E1 A1 D2 G2
    "bass_drop_d":     [26, 33, 38, 43],             # D1 A1 D2 G2
}

# String labels for display (high to low, as shown in tab)
STRING_LABELS = {
    "guitar_standard": ["e", "B", "G", "D", "A", "E"],
    "guitar_drop_d":   ["e", "B", "G", "D", "A", "D"],
    "bass_standard":   ["G", "D", "A", "E"],
    "bass_drop_d":     ["G", "D", "A", "D"],
}

MAX_FRET = 24
CHORD_TIME_THRESHOLD = 0.03  # Notes within 30ms are considered simultaneous


@dataclass
class TabNote:
    """A single note placed on the fretboard."""
    time: float        # seconds
    duration: float    # seconds
    string: int        # 0-based index (0 = lowest pitched string)
    fret: int          # 0 = open string
    midi_note: int     # original MIDI note number
    velocity: int = 100


@dataclass
class NoteEvent:
    """A MIDI note before fret assignment."""
    time: float
    duration: float
    pitch: int       # MIDI note number
    velocity: int


@dataclass
class InstrumentConfig:
    """Configuration for a fretted instrument."""
    name: str
    tuning: list[int]
    max_fret: int = MAX_FRET

    @property
    def num_strings(self) -> int:
        return len(self.tuning)

    @property
    def string_labels(self) -> list[str]:
        key = self.name
        if key in STRING_LABELS:
            return STRING_LABELS[key]
        # Generic labels for custom tunings
        return [str(i) for i in range(self.num_strings, 0, -1)]


def get_instrument_config(instrument: str, tuning: list[int] | None = None) -> InstrumentConfig:
    """Get instrument config, optionally with a custom tuning."""
    if tuning is not None:
        return InstrumentConfig(name=f"{instrument}_custom", tuning=tuning)

    default_tunings = {
        "guitar": "guitar_standard",
        "bass": "bass_standard",
    }
    tuning_key = default_tunings.get(instrument)
    if tuning_key is None:
        raise ValueError(f"Unknown instrument '{instrument}'. Use 'guitar' or 'bass'.")
    return InstrumentConfig(name=tuning_key, tuning=TUNINGS[tuning_key])


def load_midi_notes(midi_path: Path) -> tuple[list[NoteEvent], float]:
    """
    Load notes from a MIDI file.

    Returns:
        Tuple of (note_events, tempo_bpm).
    """
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    tempo = midi.estimate_tempo()
    logger.debug(f"Estimated tempo: {tempo:.1f} BPM")

    notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append(NoteEvent(
                time=note.start,
                duration=note.end - note.start,
                pitch=note.pitch,
                velocity=note.velocity,
            ))

    # Sort by time, then pitch
    notes.sort(key=lambda n: (n.time, n.pitch))
    logger.info(f"Loaded {len(notes)} notes from MIDI")
    return notes, tempo


def get_valid_positions(pitch: int, config: InstrumentConfig) -> list[tuple[int, int]]:
    """
    Get all valid (string, fret) positions for a MIDI pitch.

    Returns list of (string_index, fret) tuples, where string_index 0 is the
    lowest pitched string.
    """
    positions = []
    for string_idx, open_note in enumerate(config.tuning):
        fret = pitch - open_note
        if 0 <= fret <= config.max_fret:
            positions.append((string_idx, fret))
    return positions


def group_into_events(notes: list[NoteEvent]) -> list[list[NoteEvent]]:
    """Group notes into simultaneous events (chords) based on time proximity."""
    if not notes:
        return []

    events: list[list[NoteEvent]] = []
    current_group = [notes[0]]

    for note in notes[1:]:
        if abs(note.time - current_group[0].time) <= CHORD_TIME_THRESHOLD:
            current_group.append(note)
        else:
            events.append(current_group)
            current_group = [note]

    events.append(current_group)
    return events


def assign_frets(
    notes: list[NoteEvent],
    config: InstrumentConfig,
) -> list[TabNote]:
    """
    Assign fret positions to MIDI notes using a greedy algorithm.

    Strategy:
    - For each event (single note or chord), find all valid positions.
    - Score positions by proximity to the previous hand position.
    - For chords, ensure no two notes share a string and minimize fret span.
    """
    events = group_into_events(notes)
    tab_notes: list[TabNote] = []
    last_fret_position = 3  # Start near the nut

    for event in events:
        if len(event) == 1:
            # Single note: pick the position closest to current hand position
            note = event[0]
            positions = get_valid_positions(note.pitch, config)
            if not positions:
                logger.debug(f"Note {note.pitch} out of range, skipping")
                continue

            best = min(positions, key=lambda p: _position_score(p, last_fret_position))
            string_idx, fret = best
            tab_notes.append(TabNote(
                time=note.time,
                duration=note.duration,
                string=string_idx,
                fret=fret,
                midi_note=note.pitch,
                velocity=note.velocity,
            ))
            if fret > 0:
                last_fret_position = fret
        else:
            # Chord: assign positions avoiding string conflicts
            chord_notes = _assign_chord(event, config, last_fret_position)
            tab_notes.extend(chord_notes)
            fretted = [n.fret for n in chord_notes if n.fret > 0]
            if fretted:
                last_fret_position = sum(fretted) // len(fretted)

    logger.info(f"Assigned frets for {len(tab_notes)} notes")
    return tab_notes


def _position_score(position: tuple[int, int], last_fret: int) -> float:
    """Score a position — lower is better."""
    string_idx, fret = position
    # Prefer positions close to current hand position
    distance = abs(fret - last_fret) if fret > 0 else abs(last_fret)
    # Slight preference for lower frets (easier to play)
    low_fret_bonus = fret * 0.1
    return distance + low_fret_bonus


def _assign_chord(
    notes: list[NoteEvent],
    config: InstrumentConfig,
    last_fret: int,
) -> list[TabNote]:
    """
    Assign fret positions for a chord (simultaneous notes).

    Uses a greedy approach: assign notes one at a time, from lowest to highest
    pitch, avoiding already-used strings.
    """
    # Sort by pitch (low to high)
    sorted_notes = sorted(notes, key=lambda n: n.pitch)

    used_strings: set[int] = set()
    result: list[TabNote] = []

    for note in sorted_notes:
        positions = get_valid_positions(note.pitch, config)
        # Filter out already-used strings
        available = [(s, f) for s, f in positions if s not in used_strings]

        if not available:
            logger.debug(
                f"Chord note {note.pitch} has no available string, skipping"
            )
            continue

        # Score by distance from last position and chord coherence
        if result:
            chord_frets = [n.fret for n in result if n.fret > 0]
            avg_chord_fret = sum(chord_frets) / len(chord_frets) if chord_frets else last_fret
        else:
            avg_chord_fret = last_fret

        best = min(available, key=lambda p: _chord_position_score(p, avg_chord_fret))
        string_idx, fret = best
        used_strings.add(string_idx)

        result.append(TabNote(
            time=note.time,
            duration=note.duration,
            string=string_idx,
            fret=fret,
            midi_note=note.pitch,
            velocity=note.velocity,
        ))

    return result


def _chord_position_score(position: tuple[int, int], target_fret: float) -> float:
    """Score a chord position — lower is better."""
    string_idx, fret = position
    distance = abs(fret - target_fret) if fret > 0 else abs(target_fret)
    low_fret_bonus = fret * 0.05
    return distance + low_fret_bonus


def midi_to_tab_notes(
    midi_path: Path,
    instrument: str = "guitar",
    tuning: list[int] | None = None,
) -> tuple[list[TabNote], float, InstrumentConfig]:
    """
    Full conversion: MIDI file → assigned tab notes.

    Returns:
        Tuple of (tab_notes, tempo_bpm, instrument_config).
    """
    config = get_instrument_config(instrument, tuning)
    notes, tempo = load_midi_notes(midi_path)
    tab_notes = assign_frets(notes, config)
    return tab_notes, tempo, config
