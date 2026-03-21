"""Song analysis: key detection, tempo, chord progression, and structure."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger("tabgrabber")

# Chord templates: major and minor triads mapped to chroma bins
# Chroma order: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Major chord template: root, major third (+4), fifth (+7)
# Minor chord template: root, minor third (+3), fifth (+7)
CHORD_TEMPLATES = {}
for i, name in enumerate(NOTE_NAMES):
    major = np.zeros(12)
    major[i] = 1.0
    major[(i + 4) % 12] = 0.8
    major[(i + 7) % 12] = 0.8
    CHORD_TEMPLATES[name] = major / np.linalg.norm(major)

    minor = np.zeros(12)
    minor[i] = 1.0
    minor[(i + 3) % 12] = 0.8
    minor[(i + 7) % 12] = 0.8
    CHORD_TEMPLATES[f"{name}m"] = minor / np.linalg.norm(minor)


# Key profiles (Krumhansl-Kessler)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class ChordEvent:
    """A detected chord at a specific time."""
    time: float          # seconds
    duration: float      # seconds
    chord: str           # e.g. "C", "Am", "G#m"
    confidence: float    # 0-1


@dataclass
class SongSection:
    """A detected section of the song."""
    label: str           # e.g. "Intro", "Verse", "Chorus", "Bridge", "Outro"
    start: float         # seconds
    end: float           # seconds
    chords: list[str] = field(default_factory=list)  # chord names in this section


@dataclass
class SongAnalysis:
    """Complete analysis results for a song."""
    title: str
    key: str              # e.g. "C major", "A minor"
    key_confidence: float
    tempo: float          # BPM
    time_signature: str   # e.g. "4/4"
    duration: float       # seconds
    chords: list[ChordEvent] = field(default_factory=list)
    sections: list[SongSection] = field(default_factory=list)


def analyze_song(audio_path: Path) -> SongAnalysis:
    """
    Perform full song analysis: key, tempo, chords, and structure.

    Args:
        audio_path: Path to the audio file.

    Returns:
        SongAnalysis with all detected features.
    """
    audio_path = Path(audio_path)
    logger.info(f"Analyzing song: {audio_path.name}")

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # Tempo detection
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    logger.info(f"Detected tempo: {tempo:.1f} BPM")

    # Chroma features for key and chord detection
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    # Key detection
    key, key_confidence = _detect_key(chroma)
    logger.info(f"Detected key: {key} (confidence: {key_confidence:.2f})")

    # Chord detection
    chords = _detect_chords(chroma, sr, beat_times)
    logger.info(f"Detected {len(chords)} chord changes")

    # Song structure / section detection
    sections = _detect_sections(y, sr, chroma, chords, duration)
    logger.info(f"Detected {len(sections)} sections")

    title = audio_path.stem.replace("_", " ").replace("-", " ").title()

    return SongAnalysis(
        title=title,
        key=key,
        key_confidence=key_confidence,
        tempo=tempo,
        time_signature="4/4",  # Assume 4/4 — time sig detection is very hard
        duration=duration,
        chords=chords,
        sections=sections,
    )


def _detect_key(chroma: np.ndarray) -> tuple[str, float]:
    """Detect the key using Krumhansl-Kessler key profiles."""
    # Average chroma across all frames
    mean_chroma = np.mean(chroma, axis=1)
    mean_chroma = mean_chroma / (np.linalg.norm(mean_chroma) + 1e-8)

    best_key = "C major"
    best_score = -1.0

    for shift in range(12):
        # Try major key
        profile = np.roll(MAJOR_PROFILE, shift)
        profile = profile / np.linalg.norm(profile)
        score = np.dot(mean_chroma, profile)
        if score > best_score:
            best_score = score
            best_key = f"{NOTE_NAMES[shift]} major"

        # Try minor key
        profile = np.roll(MINOR_PROFILE, shift)
        profile = profile / np.linalg.norm(profile)
        score = np.dot(mean_chroma, profile)
        if score > best_score:
            best_score = score
            best_key = f"{NOTE_NAMES[shift]} minor"

    return best_key, float(best_score)


def _detect_chords(
    chroma: np.ndarray, sr: int, beat_times: np.ndarray,
) -> list[ChordEvent]:
    """Detect chords at each beat using template matching."""
    if len(beat_times) < 2:
        return []

    hop_length = 512
    chords = []
    prev_chord = None

    for i in range(len(beat_times)):
        # Get chroma for this beat region
        start_frame = librosa.time_to_frames(beat_times[i], sr=sr, hop_length=hop_length)
        if i + 1 < len(beat_times):
            end_frame = librosa.time_to_frames(beat_times[i + 1], sr=sr, hop_length=hop_length)
            beat_duration = beat_times[i + 1] - beat_times[i]
        else:
            end_frame = chroma.shape[1]
            beat_duration = 0.5  # estimate

        start_frame = min(start_frame, chroma.shape[1] - 1)
        end_frame = min(end_frame, chroma.shape[1])
        if start_frame >= end_frame:
            continue

        beat_chroma = np.mean(chroma[:, start_frame:end_frame], axis=1)
        norm = np.linalg.norm(beat_chroma)
        if norm < 0.01:  # silence
            continue
        beat_chroma = beat_chroma / norm

        # Match against chord templates
        best_chord = "N"  # no chord
        best_score = 0.3  # minimum threshold

        for chord_name, template in CHORD_TEMPLATES.items():
            score = np.dot(beat_chroma, template)
            if score > best_score:
                best_score = score
                best_chord = chord_name

        if best_chord == "N":
            continue

        # Merge with previous if same chord
        if chords and chords[-1].chord == best_chord:
            chords[-1].duration += beat_duration
            continue

        chords.append(ChordEvent(
            time=beat_times[i],
            duration=beat_duration,
            chord=best_chord,
            confidence=float(best_score),
        ))

    return chords


def _compute_novelty(sim: np.ndarray, kernel_size: int = 8) -> np.ndarray:
    """Compute a novelty curve from a self-similarity matrix using a checkerboard kernel."""
    n = sim.shape[0]
    k = kernel_size

    # Build checkerboard kernel
    half = k // 2
    kernel = np.ones((k, k))
    kernel[:half, :half] = -1
    kernel[half:, half:] = -1

    novelty = np.zeros(n)
    for i in range(half, n - half):
        patch = sim[i - half:i + half, i - half:i + half]
        if patch.shape == (k, k):
            novelty[i] = np.sum(patch * kernel)

    # Rectify (only positive = boundaries) and normalize
    novelty = np.maximum(novelty, 0)
    peak = np.max(novelty)
    if peak > 0:
        novelty = novelty / peak
    return novelty


def _detect_sections(
    y: np.ndarray,
    sr: int,
    chroma: np.ndarray,
    chords: list[ChordEvent],
    duration: float,
) -> list[SongSection]:
    """
    Detect song sections using a self-similarity matrix and novelty detection.

    Uses spectral features to find section boundaries, then labels sections
    based on repetition patterns.
    """
    hop_length = 512

    # Compute MFCCs for structural similarity
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)

    # Aggregate features into ~2-second windows for section-level analysis
    # Create evenly-spaced frame boundaries (~2 seconds apart)
    frames_per_window = int(2.0 * sr / hop_length)
    n_frames = mfcc.shape[1]
    window_boundaries = np.arange(0, n_frames, frames_per_window)
    beat_sync_mfcc = librosa.util.sync(mfcc, window_boundaries, aggregate=np.median)

    # Self-similarity matrix
    sim = librosa.segment.recurrence_matrix(
        beat_sync_mfcc, mode="affinity", sym=True,
    )

    # Novelty curve via checkerboard kernel convolution on the similarity matrix
    novelty = _compute_novelty(sim)

    # Find peaks in novelty curve (section boundaries)
    # Adaptive threshold based on novelty statistics
    threshold = np.mean(novelty) + 0.5 * np.std(novelty)
    min_section_frames = max(4, int(8.0 * sr / (hop_length * beat_sync_mfcc.shape[1] / (len(y) / sr))))

    peaks = []
    for i in range(1, len(novelty) - 1):
        if novelty[i] > novelty[i-1] and novelty[i] > novelty[i+1] and novelty[i] > threshold:
            # Enforce minimum distance between peaks
            if not peaks or (i - peaks[-1]) >= min_section_frames:
                peaks.append(i)

    # Convert peak indices to times
    total_frames = beat_sync_mfcc.shape[1]
    boundary_times = [0.0]
    for p in peaks:
        t = (p / total_frames) * duration
        if t > 2.0 and t < duration - 2.0:  # skip very start/end
            boundary_times.append(t)
    boundary_times.append(duration)

    # Remove boundaries that are too close together (< 8 seconds)
    filtered = [boundary_times[0]]
    for t in boundary_times[1:]:
        if t - filtered[-1] >= 8.0:
            filtered.append(t)
        elif t == boundary_times[-1]:
            # Always include the end
            if t - filtered[-1] >= 4.0:
                filtered.append(t)
            else:
                filtered[-1] = t
    boundary_times = filtered

    if len(boundary_times) < 2:
        boundary_times = [0.0, duration]

    # Build sections and assign chords
    sections = []
    for i in range(len(boundary_times) - 1):
        start = boundary_times[i]
        end = boundary_times[i + 1]

        # Get chords in this section
        section_chords = []
        for c in chords:
            if c.time >= start and c.time < end:
                section_chords.append(c.chord)

        sections.append(SongSection(
            label="",  # labeled below
            start=start,
            end=end,
            chords=section_chords,
        ))

    # Label sections by comparing chord patterns for repetition
    _label_sections(sections)

    return sections


def _label_sections(sections: list[SongSection]) -> None:
    """
    Label sections by detecting repetition patterns.

    Strategy:
    - First and last short sections → Intro/Outro
    - Sections with similar chord patterns → same label (Verse, Chorus)
    - The most repeated pattern = Verse
    - The second most repeated = Chorus
    - Unique sections = Bridge
    """
    if not sections:
        return

    # Create a fingerprint for each section based on chord sequence
    fingerprints = []
    for s in sections:
        # Simplify: take the unique chord sequence (deduplicated consecutive)
        simplified = []
        for c in s.chords:
            if not simplified or simplified[-1] != c:
                simplified.append(c)
        fingerprints.append(tuple(simplified[:8]))  # cap at 8 for comparison

    # Group similar sections
    groups: dict[int, list[int]] = {}  # group_id → list of section indices
    section_group: dict[int, int] = {}  # section_index → group_id
    next_group = 0

    for i, fp_i in enumerate(fingerprints):
        if len(fp_i) == 0:
            continue

        matched = False
        for group_id, members in groups.items():
            rep_fp = fingerprints[members[0]]
            if _chord_similarity(fp_i, rep_fp) > 0.6:
                groups[group_id].append(i)
                section_group[i] = group_id
                matched = True
                break

        if not matched:
            groups[next_group] = [i]
            section_group[i] = next_group
            next_group += 1

    # Rank groups by frequency
    group_sizes = [(gid, len(members)) for gid, members in groups.items()]
    group_sizes.sort(key=lambda x: -x[1])

    # Assign labels
    label_map = {}
    verse_assigned = False
    chorus_assigned = False

    for gid, size in group_sizes:
        if size >= 2 and not verse_assigned:
            label_map[gid] = "Verse"
            verse_assigned = True
        elif size >= 2 and not chorus_assigned:
            label_map[gid] = "Chorus"
            chorus_assigned = True
        elif size >= 2:
            label_map[gid] = "Chorus"
        else:
            label_map[gid] = "Bridge"

    # Apply labels
    for i, s in enumerate(sections):
        if i == 0 and s.end - s.start < 20.0:
            s.label = "Intro"
        elif i == len(sections) - 1 and s.end - s.start < 20.0:
            s.label = "Outro"
        elif i in section_group:
            s.label = label_map.get(section_group[i], "Section")
        else:
            s.label = "Interlude"


def _chord_similarity(a: tuple, b: tuple) -> float:
    """Compare two chord sequences for similarity (0-1)."""
    if not a or not b:
        return 0.0

    # Simple overlap-based similarity
    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    jaccard = intersection / union

    # Also check sequence order similarity
    min_len = min(len(a), len(b))
    matches = sum(1 for i in range(min_len) if a[i] == b[i])
    order_sim = matches / min_len if min_len > 0 else 0.0

    return 0.5 * jaccard + 0.5 * order_sim


def _format_time(seconds: float) -> str:
    """Format seconds as M:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def write_analysis_report(analysis: SongAnalysis, output_path: Path) -> None:
    """Write a human-readable song analysis report to a text file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"{'=' * 60}")
    lines.append(f"  SONG ANALYSIS: {analysis.title}")
    lines.append(f"{'=' * 60}")
    lines.append("")

    # Overview
    lines.append("OVERVIEW")
    lines.append(f"  Key:            {analysis.key} (confidence: {analysis.key_confidence:.0%})")
    lines.append(f"  Tempo:          {analysis.tempo:.0f} BPM")
    lines.append(f"  Time Signature: {analysis.time_signature}")
    lines.append(f"  Duration:       {_format_time(analysis.duration)}")
    lines.append("")

    # Song structure
    lines.append(f"{'-' * 60}")
    lines.append("SONG STRUCTURE")
    lines.append(f"{'-' * 60}")
    lines.append("")

    for i, section in enumerate(analysis.sections):
        time_range = f"[{_format_time(section.start)} - {_format_time(section.end)}]"
        sec_duration = section.end - section.start
        lines.append(f"  {section.label:<12s} {time_range}  ({sec_duration:.0f}s)")

        if section.chords:
            # Show unique chord progression for this section
            unique_chords = []
            for c in section.chords:
                if not unique_chords or unique_chords[-1] != c:
                    unique_chords.append(c)
            chord_str = " - ".join(unique_chords)
            lines.append(f"               Chords: {chord_str}")
        lines.append("")

    # Full chord progression
    lines.append(f"{'-' * 60}")
    lines.append("CHORD PROGRESSION (FULL)")
    lines.append(f"{'-' * 60}")
    lines.append("")

    # Group chords into lines of ~4 bars
    if analysis.chords:
        chord_line = []
        line_start = 0.0
        beats_per_bar = 4
        seconds_per_bar = beats_per_bar * (60.0 / analysis.tempo)

        for chord in analysis.chords:
            chord_line.append(chord.chord)

            # New line roughly every 4 bars
            if chord.time - line_start > seconds_per_bar * 4:
                time_marker = _format_time(line_start)
                lines.append(f"  {time_marker:>5s}  | {' | '.join(chord_line)} |")
                chord_line = []
                line_start = chord.time

        if chord_line:
            time_marker = _format_time(line_start)
            lines.append(f"  {time_marker:>5s}  | {' | '.join(chord_line)} |")
    else:
        lines.append("  No chords detected.")

    lines.append("")
    lines.append(f"{'=' * 60}")
    lines.append(f"  Generated by TabGrabber")
    lines.append(f"{'=' * 60}")

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    logger.info(f"Wrote analysis report: {output_path}")
