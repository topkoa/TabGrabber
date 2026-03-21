"""MusicXML tablature output format."""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from tabgrabber.midi_to_tab import InstrumentConfig, TabNote

logger = logging.getLogger("tabgrabber")

# MIDI note to pitch name mapping
NOTE_NAMES = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
NOTE_ALTERS = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]


def write_musicxml(
    notes: list[TabNote],
    config: InstrumentConfig,
    output_path: Path,
    tempo: float = 120.0,
    title: str = "",
) -> None:
    """
    Write a MusicXML file with tablature notation.

    Uses <technical><string> and <fret> elements for tab display.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("score-partwise", version="4.0")

    # Work title
    if title:
        work = ET.SubElement(root, "work")
        ET.SubElement(work, "work-title").text = title

    # Part list
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = config.name.replace("_", " ").title()

    # Part content
    part = ET.SubElement(root, "part", id="P1")

    if not notes:
        # Single empty measure
        measure = ET.SubElement(part, "measure", number="1")
        _add_attributes(measure, config, tempo)
        logger.warning("No notes to write to MusicXML")
        _write_xml(root, output_path)
        return

    # Group notes into measures
    seconds_per_beat = 60.0 / tempo
    seconds_per_measure = seconds_per_beat * 4

    max_time = max(n.time + n.duration for n in notes)
    num_measures = int(max_time / seconds_per_measure) + 1

    for m_idx in range(num_measures):
        m_start = m_idx * seconds_per_measure
        m_end = m_start + seconds_per_measure

        measure = ET.SubElement(part, "measure", number=str(m_idx + 1))

        if m_idx == 0:
            _add_attributes(measure, config, tempo)

        measure_notes = sorted(
            [n for n in notes if m_start <= n.time < m_end],
            key=lambda n: (n.time, -n.string),
        )

        if not measure_notes:
            # Whole rest
            note_el = ET.SubElement(measure, "note")
            ET.SubElement(note_el, "rest")
            ET.SubElement(note_el, "duration").text = str(4)
            ET.SubElement(note_el, "type").text = "whole"
            continue

        prev_time = None
        for tab_note in measure_notes:
            note_el = ET.SubElement(measure, "note")

            # Chord: if same time as previous note, add <chord/> tag
            if prev_time is not None and abs(tab_note.time - prev_time) <= 0.03:
                ET.SubElement(note_el, "chord")

            # Pitch
            pitch = ET.SubElement(note_el, "pitch")
            step = NOTE_NAMES[tab_note.midi_note % 12]
            alter = NOTE_ALTERS[tab_note.midi_note % 12]
            octave = (tab_note.midi_note // 12) - 1

            ET.SubElement(pitch, "step").text = step
            if alter != 0:
                ET.SubElement(pitch, "alter").text = str(alter)
            ET.SubElement(pitch, "octave").text = str(octave)

            # Duration (in divisions)
            beat_duration = tab_note.duration * tempo / 60.0
            divisions = max(1, int(beat_duration * 4))  # 16 divisions per measure
            ET.SubElement(note_el, "duration").text = str(divisions)

            # Note type
            ET.SubElement(note_el, "type").text = _beats_to_type(beat_duration)

            # Technical notation (tab-specific)
            notations = ET.SubElement(note_el, "notations")
            technical = ET.SubElement(notations, "technical")
            # MusicXML string numbering: 1 = highest string
            ET.SubElement(technical, "string").text = str(
                config.num_strings - tab_note.string
            )
            ET.SubElement(technical, "fret").text = str(tab_note.fret)

            prev_time = tab_note.time

    _write_xml(root, output_path)
    logger.info(f"Wrote MusicXML: {output_path}")


def _add_attributes(measure: ET.Element, config: InstrumentConfig, tempo: float) -> None:
    """Add measure attributes (clef, key, time, staff-details)."""
    attributes = ET.SubElement(measure, "attributes")
    ET.SubElement(attributes, "divisions").text = "4"

    key = ET.SubElement(attributes, "key")
    ET.SubElement(key, "fifths").text = "0"

    time = ET.SubElement(attributes, "time")
    ET.SubElement(time, "beats").text = "4"
    ET.SubElement(time, "beat-type").text = "4"

    clef = ET.SubElement(attributes, "clef")
    ET.SubElement(clef, "sign").text = "TAB"
    ET.SubElement(clef, "line").text = "5"

    # Staff details for tab
    staff_details = ET.SubElement(attributes, "staff-details")
    ET.SubElement(staff_details, "staff-lines").text = str(config.num_strings)
    for i, midi_note in enumerate(reversed(config.tuning)):
        tuning = ET.SubElement(staff_details, "staff-tuning", line=str(i + 1))
        step = NOTE_NAMES[midi_note % 12]
        ET.SubElement(tuning, "tuning-step").text = step
        alter = NOTE_ALTERS[midi_note % 12]
        if alter != 0:
            ET.SubElement(tuning, "tuning-alter").text = str(alter)
        ET.SubElement(tuning, "tuning-octave").text = str((midi_note // 12) - 1)

    # Tempo direction
    direction = ET.SubElement(measure, "direction", placement="above")
    direction_type = ET.SubElement(direction, "direction-type")
    metronome = ET.SubElement(direction_type, "metronome")
    ET.SubElement(metronome, "beat-unit").text = "quarter"
    ET.SubElement(metronome, "per-minute").text = str(int(tempo))
    sound = ET.SubElement(direction, "sound", tempo=str(int(tempo)))


def _beats_to_type(beats: float) -> str:
    """Convert beat duration to MusicXML note type name."""
    if beats >= 3.0:
        return "whole"
    elif beats >= 1.5:
        return "half"
    elif beats >= 0.75:
        return "quarter"
    elif beats >= 0.375:
        return "eighth"
    elif beats >= 0.1875:
        return "16th"
    else:
        return "32nd"


def _write_xml(root: ET.Element, output_path: Path) -> None:
    """Write XML tree to file with declaration."""
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    with open(output_path, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)
