"""Tests for CLI argument parsing."""

import pytest

from tabgrabber.cli import build_parser, parse_tuning


class TestParseTuning:
    """Test tuning string parsing."""

    def test_midi_numbers(self):
        result = parse_tuning("40,45,50,55,59,64")
        assert result == [40, 45, 50, 55, 59, 64]

    def test_note_names(self):
        result = parse_tuning("E2,A2,D3,G3,B3,E4")
        assert result == [40, 45, 50, 55, 59, 64]

    def test_note_with_sharp(self):
        result = parse_tuning("F#2")
        assert result == [42]

    def test_note_with_flat(self):
        result = parse_tuning("Bb3")
        assert result == [58]

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            parse_tuning("xyz")

    def test_whitespace_handling(self):
        result = parse_tuning("E2, A2, D3")
        assert result == [40, 45, 50]


class TestBuildParser:
    """Test CLI argument parser."""

    def test_required_args(self):
        parser = build_parser()
        args = parser.parse_args(["song.mp3", "output/"])
        assert str(args.input_file) == "song.mp3"
        assert str(args.output_dir) == "output"

    def test_default_values(self):
        parser = build_parser()
        args = parser.parse_args(["song.mp3", "output/"])
        assert args.model == "htdemucs_6s"
        assert args.device == "auto"
        assert args.instruments == ["guitar", "bass"]
        assert args.formats == ["ascii"]
        assert args.verbose is False

    def test_custom_instruments(self):
        parser = build_parser()
        args = parser.parse_args(["song.mp3", "output/", "--instruments", "guitar"])
        assert args.instruments == ["guitar"]

    def test_multiple_formats(self):
        parser = build_parser()
        args = parser.parse_args(["song.mp3", "output/", "--format", "ascii", "gp5"])
        assert args.formats == ["ascii", "gp5"]

    def test_verbose(self):
        parser = build_parser()
        args = parser.parse_args(["song.mp3", "output/", "-v"])
        assert args.verbose is True
