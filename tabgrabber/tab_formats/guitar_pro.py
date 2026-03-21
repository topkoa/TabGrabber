"""Guitar Pro (.gp5) tablature output format."""

import logging
from pathlib import Path

import guitarpro

from tabgrabber.midi_to_tab import InstrumentConfig, TabNote

logger = logging.getLogger("tabgrabber")

# Map number of strings to Guitar Pro instrument channel
INSTRUMENT_CHANNELS = {
    6: 25,   # Steel guitar
    4: 33,   # Electric bass (finger)
}


def write_guitar_pro(
    notes: list[TabNote],
    config: InstrumentConfig,
    output_path: Path,
    tempo: float = 120.0,
    title: str = "",
) -> None:
    """
    Write a Guitar Pro .gp5 file.

    Uses pyguitarpro to create a properly formatted file with tuning,
    tempo, and note data.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    song = guitarpro.models.Song()
    song.title = title or output_path.stem
    song.tempo = guitarpro.models.Tempo(int(tempo))

    # Create track with correct tuning
    track = song.tracks[0]
    track.name = config.name.replace("_", " ").title()
    track.channel.instrument = INSTRUMENT_CHANNELS.get(config.num_strings, 25)
    track.isPercussionTrack = False

    # Set string tuning (pyguitarpro expects high string first)
    track.strings = [
        guitarpro.models.GuitarString(number=i + 1, value=midi_note)
        for i, midi_note in enumerate(reversed(config.tuning))
    ]

    if not notes:
        logger.warning("No notes to write to Guitar Pro file")
        guitarpro.write(song, str(output_path))
        return

    # Group notes into measures
    seconds_per_beat = 60.0 / tempo
    seconds_per_measure = seconds_per_beat * 4  # 4/4 time

    max_time = max(n.time + n.duration for n in notes)
    num_measures = int(max_time / seconds_per_measure) + 1

    # Build measures
    # Clear default measure
    track.measures = []

    for m_idx in range(num_measures):
        m_start = m_idx * seconds_per_measure
        m_end = m_start + seconds_per_measure

        # Ensure we have a MeasureHeader
        if m_idx >= len(song.measureHeaders):
            header = guitarpro.models.MeasureHeader()
            header.number = m_idx + 1
            header.start = guitarpro.models.Duration.quarterTime * 4 * m_idx + guitarpro.models.Duration.quarterTime
            header.tempo = guitarpro.models.Tempo(int(tempo))
            song.measureHeaders.append(header)

        header = song.measureHeaders[m_idx]
        measure = guitarpro.models.Measure(header, track)

        # Get notes in this measure
        measure_notes = [n for n in notes if m_start <= n.time < m_end]

        if not measure_notes:
            # Empty measure — add a whole rest
            voice = measure.voices[0]
            beat = guitarpro.models.Beat(voice)
            beat.duration = guitarpro.models.Duration()
            beat.duration.value = 1  # whole note
            beat.status = guitarpro.models.BeatStatus.rest
            voice.beats = [beat]
        else:
            # Place notes as beats
            voice = measure.voices[0]
            voice.beats = []

            # Group simultaneous notes
            events = _group_simultaneous(measure_notes)

            for event_notes in events:
                beat = guitarpro.models.Beat(voice)
                # Estimate duration from note duration
                beat_dur = min(n.duration for n in event_notes)
                beat.duration = _seconds_to_gp_duration(beat_dur, tempo)

                for tab_note in event_notes:
                    # pyguitarpro string numbering: 1 = highest string
                    gp_string = config.num_strings - tab_note.string
                    note = guitarpro.models.Note(beat)
                    note.string = gp_string
                    note.value = tab_note.fret
                    note.velocity = tab_note.velocity
                    beat.notes.append(note)

                voice.beats.append(beat)

        track.measures.append(measure)

    guitarpro.write(song, str(output_path))
    logger.info(f"Wrote Guitar Pro file: {output_path}")


def _group_simultaneous(notes: list[TabNote]) -> list[list[TabNote]]:
    """Group notes that occur at the same time."""
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: n.time)
    groups: list[list[TabNote]] = []
    current = [sorted_notes[0]]

    for note in sorted_notes[1:]:
        if abs(note.time - current[0].time) <= 0.03:
            current.append(note)
        else:
            groups.append(current)
            current = [note]
    groups.append(current)
    return groups


def _seconds_to_gp_duration(seconds: float, tempo: float) -> guitarpro.models.Duration:
    """Convert a duration in seconds to a Guitar Pro Duration value."""
    beats = seconds * tempo / 60.0

    dur = guitarpro.models.Duration()
    if beats >= 3.0:
        dur.value = 1      # whole
    elif beats >= 1.5:
        dur.value = 2      # half
    elif beats >= 0.75:
        dur.value = 4      # quarter
    elif beats >= 0.375:
        dur.value = 8      # eighth
    elif beats >= 0.1875:
        dur.value = 16     # sixteenth
    else:
        dur.value = 32     # thirty-second

    return dur
