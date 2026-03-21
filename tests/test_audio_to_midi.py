"""Tests for audio-to-MIDI conversion configuration."""

from tabgrabber.audio_to_midi import INSTRUMENT_FREQ_RANGES


class TestInstrumentFreqRanges:
    """Test instrument frequency range definitions."""

    def test_guitar_range_defined(self):
        assert "guitar" in INSTRUMENT_FREQ_RANGES
        guitar = INSTRUMENT_FREQ_RANGES["guitar"]
        assert guitar["minimum_frequency"] == 82.0
        assert guitar["maximum_frequency"] == 1175.0

    def test_bass_range_defined(self):
        assert "bass" in INSTRUMENT_FREQ_RANGES
        bass = INSTRUMENT_FREQ_RANGES["bass"]
        assert bass["minimum_frequency"] == 41.0
        assert bass["maximum_frequency"] == 400.0

    def test_bass_lower_than_guitar(self):
        guitar = INSTRUMENT_FREQ_RANGES["guitar"]
        bass = INSTRUMENT_FREQ_RANGES["bass"]
        assert bass["minimum_frequency"] < guitar["minimum_frequency"]
        assert bass["maximum_frequency"] < guitar["maximum_frequency"]
