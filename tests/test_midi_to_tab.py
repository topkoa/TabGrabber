"""Tests for fret assignment and note-to-tab conversion."""

import pytest

from tabgrabber.midi_to_tab import (
    TUNINGS,
    InstrumentConfig,
    NoteEvent,
    TabNote,
    assign_frets,
    get_instrument_config,
    get_valid_positions,
    group_into_events,
)


class TestGetValidPositions:
    """Test note-to-fret position calculation."""

    def setup_method(self):
        self.guitar = get_instrument_config("guitar")
        self.bass = get_instrument_config("bass")

    def test_open_low_e_guitar(self):
        # MIDI 40 = E2, open low E string
        positions = get_valid_positions(40, self.guitar)
        assert (0, 0) in positions  # string 0, fret 0

    def test_fifth_fret_a_string(self):
        # MIDI 50 = D3, which is fret 5 on A string (MIDI 45) or open D string
        positions = get_valid_positions(50, self.guitar)
        assert (1, 5) in positions  # A string, fret 5
        assert (2, 0) in positions  # D string, open

    def test_note_out_of_range(self):
        # MIDI 20 is way below guitar range
        positions = get_valid_positions(20, self.guitar)
        assert positions == []

    def test_high_note_limited_positions(self):
        # MIDI 84 = C6, only reachable on high E string fret 20
        positions = get_valid_positions(84, self.guitar)
        assert (5, 20) in positions

    def test_bass_open_e(self):
        # MIDI 28 = E1, open low E on bass
        positions = get_valid_positions(28, self.bass)
        assert (0, 0) in positions

    def test_bass_note_out_of_range(self):
        # MIDI 20 is below bass range
        positions = get_valid_positions(20, self.bass)
        assert positions == []


class TestGroupIntoEvents:
    """Test chord grouping logic."""

    def test_single_notes(self):
        notes = [
            NoteEvent(time=0.0, duration=0.5, pitch=40, velocity=100),
            NoteEvent(time=0.5, duration=0.5, pitch=45, velocity=100),
        ]
        events = group_into_events(notes)
        assert len(events) == 2
        assert len(events[0]) == 1
        assert len(events[1]) == 1

    def test_chord(self):
        notes = [
            NoteEvent(time=1.0, duration=0.5, pitch=40, velocity=100),
            NoteEvent(time=1.01, duration=0.5, pitch=45, velocity=100),
            NoteEvent(time=1.02, duration=0.5, pitch=50, velocity=100),
        ]
        events = group_into_events(notes)
        assert len(events) == 1
        assert len(events[0]) == 3

    def test_mixed(self):
        notes = [
            NoteEvent(time=0.0, duration=0.5, pitch=40, velocity=100),
            NoteEvent(time=0.0, duration=0.5, pitch=45, velocity=100),
            NoteEvent(time=1.0, duration=0.5, pitch=50, velocity=100),
        ]
        events = group_into_events(notes)
        assert len(events) == 2
        assert len(events[0]) == 2  # chord
        assert len(events[1]) == 1  # single note

    def test_empty(self):
        assert group_into_events([]) == []


class TestAssignFrets:
    """Test the greedy fret assignment algorithm."""

    def setup_method(self):
        self.guitar = get_instrument_config("guitar")
        self.bass = get_instrument_config("bass")

    def test_single_open_string(self):
        notes = [NoteEvent(time=0.0, duration=0.5, pitch=40, velocity=100)]
        result = assign_frets(notes, self.guitar)
        assert len(result) == 1
        assert result[0].fret == 0
        assert result[0].string == 0  # low E

    def test_no_duplicate_strings_in_chord(self):
        # Three notes that could all go on nearby strings
        notes = [
            NoteEvent(time=0.0, duration=0.5, pitch=40, velocity=100),  # E2
            NoteEvent(time=0.0, duration=0.5, pitch=45, velocity=100),  # A2
            NoteEvent(time=0.0, duration=0.5, pitch=50, velocity=100),  # D3
        ]
        result = assign_frets(notes, self.guitar)
        strings_used = {n.string for n in result}
        assert len(strings_used) == 3  # no duplicates

    def test_out_of_range_skipped(self):
        notes = [NoteEvent(time=0.0, duration=0.5, pitch=20, velocity=100)]
        result = assign_frets(notes, self.guitar)
        assert len(result) == 0

    def test_sequential_notes_prefer_nearby_frets(self):
        # Play a simple scale on one string
        notes = [
            NoteEvent(time=0.0, duration=0.25, pitch=40, velocity=100),  # E2, fret 0
            NoteEvent(time=0.25, duration=0.25, pitch=42, velocity=100),  # F#2, fret 2
            NoteEvent(time=0.5, duration=0.25, pitch=44, velocity=100),  # G#2, fret 4
        ]
        result = assign_frets(notes, self.guitar)
        assert len(result) == 3
        # All should be on the low E string for proximity
        for note in result:
            assert note.string == 0

    def test_bass_assignment(self):
        notes = [NoteEvent(time=0.0, duration=0.5, pitch=28, velocity=100)]  # E1
        result = assign_frets(notes, self.bass)
        assert len(result) == 1
        assert result[0].fret == 0
        assert result[0].string == 0


class TestInstrumentConfig:
    """Test instrument configuration."""

    def test_guitar_standard(self):
        config = get_instrument_config("guitar")
        assert config.num_strings == 6
        assert config.tuning == [40, 45, 50, 55, 59, 64]

    def test_bass_standard(self):
        config = get_instrument_config("bass")
        assert config.num_strings == 4
        assert config.tuning == [28, 33, 38, 43]

    def test_custom_tuning(self):
        config = get_instrument_config("guitar", tuning=[38, 45, 50, 55, 59, 64])
        assert config.tuning[0] == 38  # Drop D

    def test_unknown_instrument(self):
        with pytest.raises(ValueError):
            get_instrument_config("banjo")
