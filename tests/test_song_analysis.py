"""Tests for key membership and chord-track smoothing."""

import pytest

from tabgrabber.song_analysis import (
    MIN_CHORD_WINDOW,
    ChordEvent,
    diatonic_chords,
    smooth_chords,
)


def beats(names, start=0.0, step=0.5, confidence=0.8):
    """A per-beat chord track, the shape _detect_chords produces."""
    return [
        ChordEvent(time=start + i * step, duration=step,
                   chord=name, confidence=confidence)
        for i, name in enumerate(names)
    ]


class TestDiatonicChords:
    def test_major_key(self):
        assert diatonic_chords("D major") == {"D", "Em", "F#m", "G", "A", "Bm"}

    def test_minor_key_includes_borrowed_dominant(self):
        chords = diatonic_chords("A minor")
        assert {"Am", "C", "Dm", "Em", "F", "G"} <= chords
        # The major V is far too common in minor keys to call out of key.
        assert "E" in chords

    def test_flat_root_is_normalised(self):
        assert diatonic_chords("Bb major") == diatonic_chords("A# major")

    def test_unknown_key_is_empty(self):
        assert diatonic_chords("") == set()
        assert diatonic_chords("H major") == set()


class TestSmoothChords:
    def test_empty_input(self):
        assert smooth_chords([], 120.0, "C major") == []

    def test_collapses_a_held_chord_into_one_event(self):
        result = smooth_chords(beats(["C"] * 16), 120.0, "C major")
        assert [c.chord for c in result] == ["C"]

    def test_keeps_genuine_changes(self):
        # Four bars at 120 BPM: two bars of C, then two of G.
        result = smooth_chords(beats(["C"] * 8 + ["G"] * 8), 120.0, "C major")
        assert [c.chord for c in result] == ["C", "G"]

    def test_drops_isolated_out_of_key_noise(self):
        # A single stray F# inside a bar of C should not survive.
        track = beats(["C", "C", "F#", "C", "C", "C", "C", "C"])
        result = smooth_chords(track, 120.0, "C major")
        assert [c.chord for c in result] == ["C"]

    def test_swaps_non_diatonic_winner_for_its_parallel(self):
        # Cm wins on raw weight, but C is diatonic and polled close behind.
        track = beats(["Cm", "Cm", "Cm", "C", "C", "C", "Cm", "C"])
        result = smooth_chords(track, 120.0, "C major")
        assert result[0].chord == "C"

    def test_keeps_non_diatonic_chord_with_no_diatonic_support(self):
        # Nothing suggests the parallel, so the borrowed chord stands.
        track = beats(["Eb"] * 8)
        result = smooth_chords(track, 120.0, "C major")
        assert result[0].chord == "Eb"

    def test_output_is_far_smaller_than_input(self):
        track = beats(["C", "C", "G", "C", "Am", "C", "F", "C"] * 8)
        result = smooth_chords(track, 120.0, "C major")
        assert len(result) < len(track) / 4

    def test_events_are_contiguous_and_ordered(self):
        result = smooth_chords(beats(["C"] * 8 + ["G"] * 8 + ["F"] * 8),
                               120.0, "C major")
        assert len(result) >= 2
        for earlier, later in zip(result, result[1:]):
            assert earlier.time < later.time
            assert earlier.time + earlier.duration == pytest.approx(later.time)

    @pytest.mark.parametrize("tempo", [1.0, 0.0, 300.0, 1000.0])
    def test_window_is_clamped_for_absurd_tempos(self, tempo):
        # Bar length from a nonsense tempo must not produce sub-second chords
        # (too fast) or swallow the whole song in one event (too slow).
        result = smooth_chords(beats(["C"] * 8 + ["G"] * 8), tempo, "C major")
        assert result
        # Merged runs are multiples of the window, so the floor is what matters.
        assert all(c.duration >= MIN_CHORD_WINDOW for c in result)
        assert [c.chord for c in result] == ["C", "G"]

    def test_survives_missing_key(self):
        result = smooth_chords(beats(["C"] * 8 + ["G"] * 8), 120.0, "")
        assert [c.chord for c in result] == ["C", "G"]

    def test_confidence_is_a_fraction(self):
        result = smooth_chords(beats(["C"] * 8), 120.0, "C major")
        assert all(0.0 <= c.confidence <= 1.0 for c in result)
