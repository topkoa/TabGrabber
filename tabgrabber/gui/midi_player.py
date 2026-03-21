"""MIDI file player widget using pygame.mixer."""

import logging
import tempfile
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import numpy as np
import soundfile as sf

from tabgrabber.gui.theme import COLORS, FONT_FAMILY, FONT_MONO

logger = logging.getLogger("tabgrabber")

# Try to import pygame for MIDI playback
_pygame_available = False
try:
    import pygame
    import pygame.mixer
    _pygame_available = True
except ImportError:
    logger.debug("pygame not available - MIDI playback disabled")


def _midi_to_wav(midi_path: Path, wav_path: Path, sample_rate: int = 44100) -> None:
    """Render a MIDI file to WAV using pretty_midi's fluidsynth or sine synthesis."""
    import pretty_midi
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    audio = midi.synthesize(fs=sample_rate)
    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    sf.write(str(wav_path), audio.astype(np.float32), sample_rate)


class MidiPlayerWidget(ttk.Frame):
    """A widget for loading and playing MIDI files."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._playing = False
        self._paused = False
        self._current_file: Path | None = None
        self._backing_track: Path | None = None
        self._backing_sound: object | None = None  # pygame.mixer.Sound
        self._backing_channel: object | None = None  # pygame.mixer.Channel
        self._midi_sound: object | None = None  # pygame.mixer.Sound (rendered MIDI)
        self._midi_channel: object | None = None  # pygame.mixer.Channel
        self._midi_files: list[Path] = []
        self._temp_dir = tempfile.mkdtemp(prefix="tabgrabber_player_")
        self._mixer_initialized = False

        self._build_ui()
        self._init_mixer()

    def _init_mixer(self):
        """Initialize pygame mixer for MIDI playback."""
        if not _pygame_available:
            self._set_status("pygame not installed - install with: pip install pygame")
            self._disable_controls()
            return

        try:
            # Reserve extra channels for simultaneous playback
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
            pygame.mixer.set_num_channels(8)
            self._mixer_initialized = True
            self._set_status("Ready - load a MIDI file to play")
        except Exception as e:
            self._set_status(f"Mixer init failed: {e}")
            self._disable_controls()

    def _build_ui(self):
        """Build the player UI."""
        self.configure(style="Section.TFrame")

        # File list section
        file_frame = ttk.Frame(self, style="Section.TFrame")
        file_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        ttk.Label(file_frame, text="MIDI Files (Ctrl+click to select multiple):",
                  style="Section.TLabel").pack(anchor=tk.W)

        # File listbox with scrollbar — EXTENDED selection mode
        list_frame = ttk.Frame(file_frame, style="Section.TFrame")
        list_frame.pack(fill=tk.X, pady=(5, 0))

        self._file_listbox = tk.Listbox(
            list_frame,
            height=6,
            selectmode=tk.EXTENDED,
            bg=COLORS["bg_entry"],
            fg=COLORS["fg"],
            selectbackground=COLORS["bg_accent"],
            selectforeground=COLORS["fg"],
            font=(FONT_MONO, 9),
            borderwidth=1,
            relief="flat",
            activestyle="none",
        )
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                   command=self._file_listbox.yview)
        self._file_listbox.configure(yscrollcommand=scrollbar.set)

        self._file_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._file_listbox.bind("<<ListboxSelect>>", self._on_file_select)

        # Load buttons + Select All
        btn_frame = ttk.Frame(file_frame, style="Section.TFrame")
        btn_frame.pack(fill=tk.X, pady=(5, 0))

        self._load_btn = ttk.Button(btn_frame, text="Load Folder",
                                     command=self._load_folder)
        self._load_btn.pack(side=tk.LEFT, padx=(0, 5))

        self._load_file_btn = ttk.Button(btn_frame, text="Load File",
                                          command=self._load_file)
        self._load_file_btn.pack(side=tk.LEFT, padx=(0, 5))

        self._select_all_btn = ttk.Button(btn_frame, text="Select All",
                                           command=self._select_all)
        self._select_all_btn.pack(side=tk.LEFT)

        # Playback controls
        controls_frame = ttk.Frame(self, style="Section.TFrame")
        controls_frame.pack(fill=tk.X, padx=10, pady=10)

        self._play_btn = ttk.Button(controls_frame, text="Play Selected",
                                     style="Accent.TButton",
                                     command=self._play, width=12)
        self._play_btn.pack(side=tk.LEFT, padx=(0, 5))

        self._pause_btn = ttk.Button(controls_frame, text="Pause",
                                      command=self._pause, width=8)
        self._pause_btn.pack(side=tk.LEFT, padx=(0, 5))

        self._stop_btn = ttk.Button(controls_frame, text="Stop",
                                     command=self._stop, width=8)
        self._stop_btn.pack(side=tk.LEFT, padx=(0, 5))

        # Separator
        ttk.Frame(controls_frame, style="Section.TFrame", width=10).pack(
            side=tk.LEFT, padx=5)

        self._play_backing_btn = ttk.Button(
            controls_frame, text="Play Selected + Backing",
            style="Success.TButton", command=self._play_with_backing, width=22)
        self._play_backing_btn.pack(side=tk.LEFT, padx=(0, 5))

        # Volume controls
        vol_frame = ttk.Frame(self, style="Section.TFrame")
        vol_frame.pack(fill=tk.X, padx=10, pady=(0, 5))

        # MIDI volume
        ttk.Label(vol_frame, text="MIDI Vol:", style="Section.TLabel").pack(
            side=tk.LEFT, padx=(0, 5))
        self._volume_var = tk.DoubleVar(value=0.7)
        self._volume_scale = ttk.Scale(
            vol_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
            variable=self._volume_var, command=self._on_volume_change,
            length=100,
        )
        self._volume_scale.pack(side=tk.LEFT, padx=(0, 15))

        # Backing track volume
        ttk.Label(vol_frame, text="Backing Vol:", style="Section.TLabel").pack(
            side=tk.LEFT, padx=(0, 5))
        self._backing_vol_var = tk.DoubleVar(value=0.7)
        self._backing_vol_scale = ttk.Scale(
            vol_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
            variable=self._backing_vol_var, command=self._on_backing_volume_change,
            length=100,
        )
        self._backing_vol_scale.pack(side=tk.LEFT)

        # Backing track status
        self._backing_status_var = tk.StringVar(value="No backing track loaded")
        ttk.Label(self, textvariable=self._backing_status_var,
                  style="Status.TLabel").pack(fill=tk.X, padx=10, pady=(0, 2))

        # Now playing / status
        self._status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self, textvariable=self._status_var,
                                  style="Status.TLabel")
        status_label.pack(fill=tk.X, padx=10, pady=(0, 10))

    def _disable_controls(self):
        """Disable playback controls."""
        for btn in (self._play_btn, self._pause_btn, self._stop_btn,
                    self._play_backing_btn):
            btn.configure(state="disabled")
        self._volume_scale.configure(state="disabled")
        self._backing_vol_scale.configure(state="disabled")

    def _set_status(self, text: str):
        """Update status label."""
        self._status_var.set(text)

    def _load_folder(self):
        """Open folder dialog and load all MIDI files from it."""
        from tkinter import filedialog
        folder = filedialog.askdirectory(title="Select folder with MIDI files")
        if not folder:
            return

        folder_path = Path(folder)
        self._populate_file_list(folder_path)

    def _load_file(self):
        """Open file dialog to load a single MIDI file."""
        from tkinter import filedialog
        filepath = filedialog.askopenfilename(
            title="Select MIDI file",
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")],
        )
        if not filepath:
            return

        self._file_listbox.delete(0, tk.END)
        self._midi_files = [Path(filepath)]
        self._file_listbox.insert(tk.END, filepath)
        self._file_listbox.selection_set(0)
        self._set_status(f"Loaded: {Path(filepath).name}")

    def _populate_file_list(self, folder: Path):
        """Find all MIDI files in folder and populate the listbox."""
        self._file_listbox.delete(0, tk.END)
        self._midi_files = []

        midi_files = sorted(folder.glob("**/*.mid")) + sorted(folder.glob("**/*.midi"))
        if not midi_files:
            self._set_status(f"No MIDI files found in {folder.name}")
            return

        self._midi_files = midi_files
        for f in midi_files:
            self._file_listbox.insert(tk.END, str(f))

        # Select all by default
        self._select_all()
        self._set_status(f"Found {len(midi_files)} MIDI file(s) - all selected")

    def _select_all(self):
        """Select all items in the listbox."""
        self._file_listbox.selection_set(0, tk.END)
        count = self._file_listbox.size()
        self._set_status(f"Selected {count} MIDI file(s)")

    def _get_selected_files(self) -> list[Path]:
        """Get list of selected MIDI file paths."""
        selection = self._file_listbox.curselection()
        return [Path(self._file_listbox.get(i)) for i in selection]

    def _on_file_select(self, event):
        """Handle file list selection."""
        selected = self._get_selected_files()
        if len(selected) == 1:
            self._current_file = selected[0]
            self._set_status(f"Selected: {self._current_file.name}")
        elif len(selected) > 1:
            names = ", ".join(f.stem for f in selected)
            self._set_status(f"Selected {len(selected)} files: {names}")

    def _render_midi_to_wav(self, midi_files: list[Path]) -> Path:
        """Render one or more MIDI files to a single mixed WAV."""
        sample_rate = 44100
        mixed = None

        for midi_path in midi_files:
            import pretty_midi
            midi = pretty_midi.PrettyMIDI(str(midi_path))
            audio = midi.synthesize(fs=sample_rate)

            if mixed is None:
                mixed = audio.astype(np.float64)
            else:
                # Pad to match lengths
                if len(audio) > len(mixed):
                    mixed = np.pad(mixed, (0, len(audio) - len(mixed)))
                elif len(audio) < len(mixed):
                    audio = np.pad(audio, (0, len(mixed) - len(audio)))
                mixed += audio.astype(np.float64)

        if mixed is None:
            raise ValueError("No MIDI data to render")

        # Normalize
        peak = np.max(np.abs(mixed))
        if peak > 0:
            mixed = mixed / peak * 0.9

        wav_path = Path(self._temp_dir) / "midi_mix.wav"
        sf.write(str(wav_path), mixed.astype(np.float32), sample_rate)
        return wav_path

    def _play(self):
        """Play the selected MIDI file(s)."""
        if not self._mixer_initialized:
            return

        if self._paused:
            if self._midi_channel is not None:
                self._midi_channel.unpause()
            else:
                pygame.mixer.music.unpause()
            self._paused = False
            self._playing = True
            self._set_status("Playing...")
            return

        selected = self._get_selected_files()
        if not selected:
            self._set_status("No file selected")
            return

        try:
            self._stop()
            self._set_status("Rendering MIDI to audio...")
            self.update_idletasks()

            wav_path = self._render_midi_to_wav(selected)
            self._midi_sound = pygame.mixer.Sound(str(wav_path))
            self._midi_sound.set_volume(self._volume_var.get())
            self._midi_channel = self._midi_sound.play()

            self._playing = True
            names = " + ".join(f.stem for f in selected)
            self._set_status(f"Playing: {names}")
        except Exception as e:
            self._set_status(f"Playback error: {e}")
            logger.error(f"MIDI playback error: {e}")

    def _pause(self):
        """Pause/unpause playback."""
        if not self._mixer_initialized or not self._playing:
            return

        if self._paused:
            if self._midi_channel is not None:
                self._midi_channel.unpause()
            if self._backing_channel is not None:
                self._backing_channel.unpause()
            self._paused = False
            self._set_status("Playing...")
        else:
            if self._midi_channel is not None:
                self._midi_channel.pause()
            if self._backing_channel is not None:
                self._backing_channel.pause()
            self._paused = True
            self._set_status("Paused")

    def _play_with_backing(self):
        """Play all MIDI files and backing track simultaneously."""
        if not self._mixer_initialized:
            return

        selected = self._get_selected_files()
        if not selected:
            self._set_status("No MIDI files selected")
            return

        if self._backing_track is None or not self._backing_track.exists():
            self._set_status("No backing track available - process a song first")
            return

        try:
            self._stop()
            self._set_status("Rendering MIDI to audio...")
            self.update_idletasks()

            # Render all selected MIDIs to a single WAV
            wav_path = self._render_midi_to_wav(selected)
            self._midi_sound = pygame.mixer.Sound(str(wav_path))
            self._midi_sound.set_volume(self._volume_var.get())

            # Load backing track
            self._backing_sound = pygame.mixer.Sound(str(self._backing_track))
            self._backing_sound.set_volume(self._backing_vol_var.get())

            # Start both at the same time
            self._backing_channel = self._backing_sound.play()
            self._midi_channel = self._midi_sound.play()

            self._playing = True
            names = " + ".join(f.stem for f in selected)
            self._set_status(f"Playing with backing: {names}")
        except Exception as e:
            self._set_status(f"Playback error: {e}")
            logger.error(f"Backing playback error: {e}")

    def _stop(self):
        """Stop all playback."""
        if not self._mixer_initialized:
            return

        pygame.mixer.music.stop()

        if self._midi_channel is not None:
            self._midi_channel.stop()
            self._midi_channel = None

        if self._backing_channel is not None:
            self._backing_channel.stop()
            self._backing_channel = None

        self._playing = False
        self._paused = False
        self._set_status("Stopped")

    def _on_volume_change(self, value):
        """Handle MIDI volume slider change."""
        if self._midi_sound is not None:
            self._midi_sound.set_volume(float(value))
        elif self._mixer_initialized:
            pygame.mixer.music.set_volume(float(value))

    def _on_backing_volume_change(self, value):
        """Handle backing track volume slider change."""
        if self._backing_sound is not None:
            self._backing_sound.set_volume(float(value))

    def load_output_dir(self, output_dir: Path):
        """Programmatically load MIDI files and backing track from pipeline output.

        Searches recursively so it works with per-song subfolders:
            output_dir/
              song1/midi/*.mid
              song1/backing_track.wav
              song2/midi/*.mid
              ...
        """
        output_dir = Path(output_dir)
        self._file_listbox.delete(0, tk.END)
        self._midi_files = []

        # Find all MIDI files recursively
        midi_files = sorted(output_dir.rglob("*.mid")) + sorted(output_dir.rglob("*.midi"))
        if midi_files:
            self._midi_files = midi_files
            for f in midi_files:
                try:
                    display = str(f.relative_to(output_dir))
                except ValueError:
                    display = str(f)
                self._file_listbox.insert(tk.END, str(f))
            self._select_all()
            self._set_status(f"Found {len(midi_files)} MIDI file(s)")
        else:
            self._set_status(f"No MIDI files found in {output_dir}")

        # Look for backing track (use the most recent one found)
        backing_files = sorted(output_dir.rglob("backing_track.wav"))
        if backing_files:
            self._backing_track = backing_files[-1]
            self._backing_status_var.set(f"Backing track: {self._backing_track.parent.name}/{self._backing_track.name}")
        else:
            self._backing_track = None
            self._backing_status_var.set("No backing track found")

    def cleanup(self):
        """Clean up pygame resources and temp files."""
        if self._mixer_initialized:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except Exception:
                pass

        # Clean up temp WAVs
        import shutil
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass
