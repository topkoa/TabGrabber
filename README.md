# TabGrabber

Extract guitar and bass tablature from audio files using AI-powered stem separation and pitch detection.

TabGrabber takes any audio file, isolates the guitar and bass using Demucs source separation, converts the isolated stems to MIDI via librosa pitch tracking, and generates playable tablature in multiple formats. It also performs song analysis to detect key, tempo, chord progressions, and song structure.

## Features

- **AI Stem Separation** - Uses Facebook's Demucs (htdemucs_6s) to isolate 6 stems: guitar, bass, drums, vocals, piano, other
- **Audio-to-MIDI Conversion** - Polyphonic pitch detection for guitar, monophonic for bass, with instrument-specific frequency filtering
- **Tablature Generation** - Greedy fret assignment algorithm that minimizes hand movement and handles chords
- **Multiple Output Formats** - ASCII tab (.txt), Guitar Pro (.gp5), MusicXML (.xml), Sloppak (.sloppak)
- **Slopsmith Integration** - Package a whole run as a `.sloppak` song — stems, arrangements, and metadata — ready to drop into [slopsmith](https://github.com/byrongamatos/slopsmith) and play on the highway
- **Karaoke Lyrics** - Transcribe the vocal stem with WhisperX (forced word-level alignment) and embed per-word timed lyrics into the `.sloppak` for highway-synced karaoke display
- **Song Analysis** - Detects key, tempo, chord progressions, and song structure (intro/verse/chorus/bridge/outro)
- **GUI with MIDI Player** - Dark-themed Tkinter interface with built-in MIDI playback and backing track support
- **Batch Processing** - Process entire folders of audio files, each getting its own output subfolder

## Installation

Requires Python 3.10-3.12.

```bash
# Clone the repository
git clone https://github.com/topkoa/TabGrabber.git
cd TabGrabber

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| torch, torchaudio | ML framework for Demucs |
| demucs | AI stem separation |
| librosa | Onset detection, pitch tracking, song analysis |
| pretty_midi | MIDI file manipulation |
| pyguitarpro | Guitar Pro file output |
| PyYAML | Sloppak manifest authoring |
| whisperx | Karaoke-style lyrics extraction (Whisper + wav2vec2 alignment) |
| soundfile, numpy | Audio I/O |
| pygame | MIDI playback in GUI (optional) |

> **Note:** torch/torchaudio pins (2.6.0) match the [RocksmithGuitarMute](https://github.com/topkoa/RocksmithGuitarMute) project for shared virtual environment compatibility.

## Usage

### GUI (Recommended)

```bash
# Double-click (no console window):
run_gui.pyw

# Or from command line:
python -m tabgrabber.gui

# Or if pip-installed:
tabgrabber-gui
```

The GUI provides:

- **Process tab** - Select input file/folder and output folder, configure options, and run
- **MIDI Player tab** - Load and play generated MIDI files, with backing track support

#### Processing Options

| Option | Description | Default |
|--------|-------------|---------|
| Demucs Model | Stem separation model | htdemucs_6s |
| Device | cpu, cuda, or auto | auto |
| Instruments | Guitar, Bass, or both | Both |
| Output Formats | ASCII, Guitar Pro, MusicXML, Sloppak | ASCII |
| Invert String Order | Low E on top in text tabs | Off |
| Quality Preset | fast, balanced, high, extreme | fast |
| Onset Threshold | Note onset sensitivity (0-1) | 0.5 |
| Frame Threshold | Pitch confidence threshold (0-1) | 0.3 |

**Quality Presets** trade processing speed for accuracy:

| Preset | Demucs Shifts | Overlap | Segment | Hop Length | FFT Size | Onset Window | Speed |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|-------|
| **fast** | 0 | 0.25 | default | 512 | 2048 | 4 | Fastest |
| **balanced** | 1 | 0.25 | default | 512 | 2048 | 6 | Moderate |
| **high** | 3 | 0.50 | default | 256 | 4096 | 8 | Slow |
| **extreme** | 5 | 0.75 | 40s | 128 | 8192 | 12 | Very slow |

Advanced settings can be expanded in the GUI to fine-tune individual parameters beyond presets.

#### MIDI Player

- Select one or multiple MIDI files (Ctrl+click or "Select All")
- **Play Selected** - Plays MIDI only
- **Play Selected + Backing** - Plays MIDI alongside the backing track (all stems minus guitar/bass)
- Separate volume sliders for MIDI and backing track
- Supports pause/stop and auto-loads files after processing

### CLI

```bash
python -m tabgrabber input_file output_dir [options]
```

#### Examples

```bash
# Basic usage - extract guitar and bass tabs as ASCII
python -m tabgrabber song.mp3 ./output

# Guitar only, Guitar Pro format, with CUDA acceleration
python -m tabgrabber song.wav ./output --instruments guitar --format gp5 --device cuda

# All formats, custom tuning (Drop D)
python -m tabgrabber song.flac ./output --format all --tuning D2,A2,D3,G3,B3,E4

# Adjust detection sensitivity
python -m tabgrabber song.ogg ./output --onset-threshold 0.4 --frame-threshold 0.2

# Use high quality preset
python -m tabgrabber song.mp3 ./output --quality high

# Extreme quality with custom Demucs shifts
python -m tabgrabber song.mp3 ./output --quality extreme --demucs-shifts 8

# Package a slopsmith-playable .sloppak (zip form)
python -m tabgrabber "Artist - Song.mp3" ./output --format sloppak

# Sloppak directory form (hand-editable)
python -m tabgrabber song.mp3 ./output --format sloppak --sloppak-dir
```

#### CLI Options

```
positional arguments:
  input_file            Path to audio file (WAV, MP3, FLAC, OGG)
  output_dir            Directory for output files

optional arguments:
  --model MODEL         Demucs model (default: htdemucs_6s)
  --device {auto,cpu,cuda}
  --instruments {guitar,bass} [...]
  --format {ascii,gp5,musicxml,sloppak,all} [...]
  --sloppak-dir         Emit sloppak as a directory instead of a zip file
  --lyrics              Transcribe vocals with WhisperX and write lyrics.json
  --no-lyrics           Skip lyrics even when sloppak is requested
  --lyrics-model {tiny,base,small,medium,large-v2,large-v3}
                        Whisper model size (default: large-v2)
  --lyrics-language LANG
                        Force language ISO code (default: autodetect)
  --tuning TUNING       Custom tuning, e.g. "E2,A2,D3,G3,B3,E4" or "40,45,50,55,59,64"
  --onset-threshold     Note onset sensitivity 0-1 (default: 0.5)
  --frame-threshold     Pitch confidence 0-1 (default: 0.3)
  --keep-intermediates  Keep stem WAVs and MIDI files
  -v, --verbose         Verbose logging

quality options:
  --quality {fast,balanced,high,extreme}
                        Quality preset (default: fast)
  --demucs-shifts N     Demucs random shifts (overrides preset)
  --demucs-overlap F    Demucs segment overlap 0.0-0.99 (overrides preset)
  --hop-length {128,256,512,1024}
                        Analysis hop length (overrides preset)
  --n-fft {1024,2048,4096,8192}
                        FFT window size (overrides preset)
  --onset-window N      Frames around each onset (overrides preset)
```

## Output Structure

Each processed song gets its own subfolder:

```
output/
  songname/
    stems/
      htdemucs_6s/
        songname/
          guitar.wav
          bass.wav
          drums.wav
          vocals.wav
          piano.wav
          other.wav
    midi/
      guitar.mid
      bass.mid
    tabs/
      guitar_tab.txt
      bass_tab.txt
    backing_track.wav      # All stems minus guitar/bass, mixed
    song_analysis.txt      # Key, tempo, chords, structure
    songname.sloppak       # Optional — only with --format sloppak
```

## Sloppak Output (Slopsmith Integration)

Passing `--format sloppak` packages a TabGrabber run as a `.sloppak` song for [slopsmith](https://github.com/byrongamatos/slopsmith) — the open song format introduced in slopsmith PR #7. A single `.sloppak` contains everything slopsmith needs to render and play the song on the highway, and everything the [Stems plugin](https://github.com/topkoa/slopsmith-plugin-stems) needs for live per-instrument mixing.

### What's inside

```
Artist - Song.sloppak/
  manifest.yaml           # title, artist, duration, stems[], arrangements[]
  stems/
    guitar.ogg            # all six demucs stems, encoded to OGG Vorbis
    bass.ogg
    drums.ogg
    vocals.ogg
    piano.ogg
    other.ogg
  arrangements/
    guitar.json           # slopsmith highway wire format (Lead)
    bass.json             # slopsmith highway wire format (Bass)
```

The arrangement JSON matches slopsmith's `arrangement_to_wire` schema byte-for-byte, so it loads through the same seam as PSARC-sourced songs — no client changes needed. Effect flags (bends, hammer-ons, palm mute, etc.) are left at their defaults since TabGrabber doesn't detect them.

### Title / artist

Filenames of the form `Artist - Title.mp3` are parsed into the manifest's `artist` and `title` fields. Filenames without a ` - ` separator become `title`-only with an empty artist.

### Requirements

- **ffmpeg** on `PATH` — used to encode each demucs stem to OGG Vorbis (`libvorbis -q:a 5`).
- **PyYAML** — installed via `requirements.txt`.

### Usage

Drop the generated `.sloppak` into slopsmith's `dlc/` directory. It shows up in the library with a **SLOPPAK** badge alongside PSARC songs. With the Stems plugin installed, each demucs stem becomes a live-mixable channel.

### Lyrics Extraction (Karaoke Mode)

Every `--format sloppak` run also transcribes the isolated vocal stem into per-word timed lyrics, stored as `lyrics.json` inside the package and referenced from `manifest.yaml`. Slopsmith renders them as karaoke-style lyrics on the highway alongside the note chart.

**How it works**: TabGrabber feeds Demucs's `vocals.wav` into [WhisperX](https://github.com/m-bain/whisperX), which runs Whisper (`large-v2` by default) for transcription and then wav2vec2 forced alignment for phoneme-level word timing. The result is much tighter than plain Whisper's `word_timestamps` on sung vocals — typically sub-150ms drift on clear vocals.

**Output shape** (matches slopsmith's lyrics schema):

```json
[
  {"t": 12.34, "d": 0.28, "w": "hello"},
  {"t": 12.62, "d": 0.41, "w": "darkness"},
  ...
]
```

**First run**: downloads ~3 GB of model weights (Whisper `large-v2` + wav2vec2 English alignment). GPU is strongly recommended but CPU works.

**Flags**:
- `--no-lyrics` skips lyrics entirely (e.g. instrumentals)
- `--lyrics-model medium` trades some accuracy for a smaller/faster run
- `--lyrics-language es` forces a non-English alignment model
- `--lyrics` also works on non-sloppak runs — writes `lyrics.json` next to the tabs

## Song Analysis

TabGrabber analyzes the original audio to detect:

- **Key** - Using Krumhansl-Kessler key profiles against chroma features (e.g. "D major", "A minor")
- **Tempo** - Beat tracking via librosa
- **Chord Progression** - Chroma-to-chord template matching at each beat (major and minor triads)
- **Song Structure** - Section boundaries via self-similarity matrix and novelty detection, labeled as Intro/Verse/Chorus/Bridge/Outro based on chord pattern repetition

Example output:

```
============================================================
  SONG ANALYSIS: One Headlight
============================================================

OVERVIEW
  Key:            D major (confidence: 97%)
  Tempo:          108 BPM
  Time Signature: 4/4
  Duration:       5:12

------------------------------------------------------------
SONG STRUCTURE
------------------------------------------------------------

  Intro        [0:00 - 0:18]  (18s)
               Chords: D - A - G

  Verse        [0:18 - 1:05]  (47s)
               Chords: D - A - G - D - A - G
  ...
```

## How It Works

### Pipeline

```
Audio file (.wav/.mp3/.flac/.ogg)
  1. Demucs htdemucs_6s  -->  6 stem WAV files
  2. librosa piptrack    -->  MIDI files (guitar + bass)
  3. Fret assignment     -->  Tablature files (.txt / .gp5 / .xml)
  4. Song analysis       -->  song_analysis.txt
```

### Fret Assignment Algorithm

The MIDI-to-tab conversion uses a greedy algorithm:

1. Notes are grouped into events (single notes vs chords) based on timing
2. For each note, all valid (string, fret) positions are computed
3. Single notes: pick the position closest to the current hand position
4. Chords: assign from lowest pitch up, avoiding string conflicts and minimizing fret span
5. Bass uses monophonic mode (1 note per onset) for cleaner output

### Supported Tunings

| Tuning | Notes | MIDI Values |
|--------|-------|-------------|
| Guitar Standard | E2 A2 D3 G3 B3 E4 | 40 45 50 55 59 64 |
| Guitar Drop D | D2 A2 D3 G3 B3 E4 | 38 45 50 55 59 64 |
| Bass Standard | E1 A1 D2 G2 | 28 33 38 43 |
| Bass Drop D | D1 A1 D2 G2 | 26 33 38 43 |

Custom tunings can be specified via CLI (`--tuning`) or note name format (e.g. `C#2,G#2,D#3,A#3`).

## Project Structure

```
tabgrabber/
  __init__.py          # Package version
  __main__.py          # python -m tabgrabber entry point
  cli.py               # CLI argument parsing
  pipeline.py          # Orchestrates the 4-stage pipeline
  stems.py             # Demucs stem extraction
  audio_to_midi.py     # librosa onset/pitch detection -> MIDI
  midi_to_tab.py       # MIDI note -> fret assignment
  song_analysis.py     # Key, tempo, chord, structure detection
  lyrics.py            # WhisperX vocal-stem lyrics extraction
  utils.py             # Device detection, logging, validation
  tab_formats/
    ascii_tab.py       # ASCII tablature writer
    guitar_pro.py      # Guitar Pro .gp5 writer
    musicxml.py        # MusicXML writer
    sloppak.py         # Slopsmith .sloppak packager
  gui/
    __main__.py        # python -m tabgrabber.gui entry point
    gui_main.py        # Main GUI window
    midi_player.py     # MIDI player widget with backing track
    theme.py           # Dark theme configuration
    launch_gui.py      # GUI entry point
run_gui.pyw              # Double-click GUI launcher (no console)
```

## License

MIT
