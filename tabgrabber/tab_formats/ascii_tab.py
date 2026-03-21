"""ASCII tablature output format."""

import logging
from pathlib import Path

from tabgrabber.midi_to_tab import InstrumentConfig, TabNote

logger = logging.getLogger("tabgrabber")

# Characters per beat at default resolution
CHARS_PER_BEAT = 4
BEATS_PER_MEASURE = 4
MEASURES_PER_LINE = 4


def write_ascii_tab(
    notes: list[TabNote],
    config: InstrumentConfig,
    output_path: Path,
    tempo: float = 120.0,
    title: str = "",
    measures_per_line: int = MEASURES_PER_LINE,
    invert_strings: bool = False,
) -> None:
    """
    Generate ASCII tablature and write to a text file.

    Default (high string on top):
        e|---0---2---3---|
        B|---1---3---0---|
        ...
        E|-----------3---|

    Inverted (low string on top):
        E|-----------3---|
        A|---3-------2---|
        ...
        e|---0---2---3---|
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not notes:
        logger.warning("No notes to write")
        output_path.write_text(f"{title}\n\nNo notes detected.\n")
        return

    # Calculate timing
    seconds_per_beat = 60.0 / tempo
    chars_per_measure = CHARS_PER_BEAT * BEATS_PER_MEASURE
    seconds_per_measure = seconds_per_beat * BEATS_PER_MEASURE

    # Find total duration
    max_time = max(n.time + n.duration for n in notes)
    total_measures = int(max_time / seconds_per_measure) + 1

    # Build tab grid: string_index → list of characters
    num_strings = config.num_strings
    labels = config.string_labels  # default: high to low display order
    if invert_strings:
        labels = list(reversed(labels))

    total_chars = total_measures * chars_per_measure
    # Initialize grids for each string with dashes
    grids = [list("-" * total_chars) for _ in range(num_strings)]

    # Place notes on the grid
    for note in notes:
        char_pos = int(note.time / seconds_per_beat * CHARS_PER_BEAT)
        if char_pos >= total_chars:
            continue

        fret_str = str(note.fret)
        # string index 0 = lowest string
        # default: high-to-low (row 0 = highest string)
        # inverted: low-to-high (row 0 = lowest string)
        if invert_strings:
            display_row = note.string
        else:
            display_row = num_strings - 1 - note.string

        for i, ch in enumerate(fret_str):
            pos = char_pos + i
            if pos < total_chars:
                grids[display_row][pos] = ch

    # Format output with measure bars and line breaks
    lines = []
    if title:
        lines.append(title)
        lines.append(f"Tempo: {tempo:.0f} BPM")
        lines.append("")

    for measure_start in range(0, total_measures, measures_per_line):
        measure_end = min(measure_start + measures_per_line, total_measures)

        for row in range(num_strings):
            label = labels[row]
            line_parts = []
            for m in range(measure_start, measure_end):
                start = m * chars_per_measure
                end = start + chars_per_measure
                segment = "".join(grids[row][start:end])
                line_parts.append(segment)

            line = f"{label}|{'|'.join(line_parts)}|"
            lines.append(line)

        lines.append("")  # blank line between tab lines

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    logger.info(f"Wrote ASCII tab: {output_path}")
