"""Tests for Guitar Pro (.gp5) output.

These are regression tests for pyguitarpro API drift: the writer previously
used `models.Tempo` (removed in 0.10) and passed `Measure(header, track)` with
the arguments the wrong way round, which made .gp5 export fail outright.
"""

import guitarpro
import pytest

from tabgrabber.midi_to_tab import InstrumentConfig, TabNote
from tabgrabber.tab_formats.guitar_pro import _make_tempo, write_guitar_pro

GUITAR = InstrumentConfig(name="guitar", tuning=[40, 45, 50, 55, 59, 64])
BASS = InstrumentConfig(name="bass", tuning=[28, 33, 38, 43])


def _notes():
    """A short single-note run plus a two-note chord."""
    return [
        TabNote(time=0.0, duration=0.5, string=0, fret=3, midi_note=43),
        TabNote(time=0.5, duration=0.5, string=1, fret=2, midi_note=47),
        TabNote(time=1.0, duration=0.5, string=2, fret=0, midi_note=50),
        TabNote(time=1.5, duration=0.5, string=3, fret=2, midi_note=57),
        TabNote(time=1.5, duration=0.5, string=4, fret=3, midi_note=62),
    ]


class TestMakeTempo:
    """The tempo shim has to work on both pyguitarpro generations."""

    def test_returns_something_usable(self):
        assert _make_tempo(120.0) is not None

    def test_accepts_float_tempo(self):
        # Tempo estimates arrive as floats; the file format wants an integer.
        result = _make_tempo(138.6)
        value = getattr(result, "value", result)
        assert value == 138


class TestWriteGuitarPro:
    def test_writes_readable_file(self, tmp_path):
        out = tmp_path / "guitar_tab.gp5"
        write_guitar_pro(_notes(), GUITAR, out, tempo=120.0, title="Test Song")

        assert out.exists() and out.stat().st_size > 0
        song = guitarpro.parse(str(out))
        assert song.title == "Test Song"
        assert len(song.tracks) >= 1

    def test_roundtrip_preserves_notes(self, tmp_path):
        out = tmp_path / "guitar_tab.gp5"
        write_guitar_pro(_notes(), GUITAR, out, tempo=120.0, title="Test Song")

        song = guitarpro.parse(str(out))
        frets = [
            note.value
            for measure in song.tracks[0].measures
            for voice in measure.voices
            for beat in voice.beats
            for note in beat.notes
        ]
        assert sorted(frets) == sorted(n.fret for n in _notes())

    def test_bass_four_strings(self, tmp_path):
        out = tmp_path / "bass_tab.gp5"
        notes = [TabNote(time=0.0, duration=0.5, string=0, fret=5, midi_note=33)]
        write_guitar_pro(notes, BASS, out, tempo=100.0, title="Bass")

        song = guitarpro.parse(str(out))
        assert len(song.tracks[0].strings) == 4

    def test_empty_note_list_still_writes(self, tmp_path):
        out = tmp_path / "empty.gp5"
        write_guitar_pro([], GUITAR, out, tempo=120.0, title="Empty")
        assert out.exists()

    def test_creates_missing_parent_directory(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "guitar_tab.gp5"
        write_guitar_pro(_notes(), GUITAR, out, tempo=120.0, title="Nested")
        assert out.exists()
