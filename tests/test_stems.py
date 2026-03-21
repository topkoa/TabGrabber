"""Tests for stem extraction configuration."""

from tabgrabber.stems import DEFAULT_MODEL


def test_default_model():
    assert DEFAULT_MODEL == "htdemucs_6s"
