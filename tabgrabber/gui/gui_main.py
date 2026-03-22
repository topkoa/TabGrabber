"""Main GUI window for TabGrabber."""

import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

from tabgrabber import __version__
from tabgrabber.gui.theme import COLORS, FONT_FAMILY, FONT_MONO, apply_dark_theme
from tabgrabber.gui.midi_player import MidiPlayerWidget
from tabgrabber.pipeline import PipelineOptions, QUALITY_PRESETS, process
from tabgrabber.utils import get_device, SUPPORTED_AUDIO_EXTENSIONS

logger = logging.getLogger("tabgrabber")


class TabGrabberGUI:
    """Main application window."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"TabGrabber v{__version__}")
        self.root.geometry("800x900")
        self.root.minsize(700, 700)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._style = apply_dark_theme(self.root)
        self._processing = False
        self._log_queue: queue.Queue = queue.Queue()

        self._build_ui()
        self._poll_log_queue()

    def _build_ui(self):
        """Build the full UI layout."""
        # Main scrollable container
        canvas = tk.Canvas(self.root, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient=tk.VERTICAL, command=canvas.yview)
        self._scroll_frame = ttk.Frame(canvas, style="TFrame")

        self._scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Make the inner frame stretch to canvas width
        def _on_canvas_configure(event):
            canvas.itemconfig(canvas.find_withtag("all")[0], width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        parent = self._scroll_frame

        # Header
        self._build_header(parent)

        # Notebook with tabs
        self._notebook = ttk.Notebook(parent)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tab 1: Processing
        process_tab = ttk.Frame(self._notebook, style="TFrame")
        self._notebook.add(process_tab, text="  Process  ")
        self._build_process_tab(process_tab)

        # Tab 2: MIDI Player
        player_tab = ttk.Frame(self._notebook, style="TFrame")
        self._notebook.add(player_tab, text="  MIDI Player  ")
        self._build_player_tab(player_tab)

    def _build_header(self, parent):
        """Build the title header."""
        header = ttk.Frame(parent, style="TFrame")
        header.pack(fill=tk.X, padx=10, pady=(10, 5))

        ttk.Label(header, text="TabGrabber", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Extract guitar & bass tabs from audio using AI",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W)

        # Separator
        sep = ttk.Frame(header, style="Header.TFrame", height=2)
        sep.pack(fill=tk.X, pady=(10, 0))

    def _build_process_tab(self, parent):
        """Build the processing tab."""
        # File selection
        self._build_file_section(parent)

        # Options
        self._build_options_section(parent)

        # Progress
        self._build_progress_section(parent)

        # Control buttons
        self._build_controls(parent)

        # Activity log
        self._build_log_section(parent)

    def _build_file_section(self, parent):
        """Build input/output file selection."""
        frame = ttk.LabelFrame(parent, text="File Selection", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)

        # Input
        ttk.Label(frame, text="Input:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self._input_var = tk.StringVar()
        input_entry = ttk.Entry(frame, textvariable=self._input_var, width=50)
        input_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)

        btn_frame_in = ttk.Frame(frame)
        btn_frame_in.grid(row=0, column=2, pady=2)
        ttk.Button(btn_frame_in, text="File", width=6,
                   command=self._select_input_file).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame_in, text="Folder", width=6,
                   command=self._select_input_folder).pack(side=tk.LEFT, padx=1)

        # Output
        ttk.Label(frame, text="Output:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self._output_var = tk.StringVar()
        output_entry = ttk.Entry(frame, textvariable=self._output_var, width=50)
        output_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)

        ttk.Button(frame, text="Browse", width=8,
                   command=self._select_output_folder).grid(row=1, column=2, pady=2)

        frame.columnconfigure(1, weight=1)

    def _build_options_section(self, parent):
        """Build processing options."""
        frame = ttk.LabelFrame(parent, text="Processing Options", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)

        # Row 0: Model + Device
        ttk.Label(frame, text="Demucs Model:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self._model_var = tk.StringVar(value="htdemucs_6s")
        model_combo = ttk.Combobox(frame, textvariable=self._model_var, width=18,
                                    state="readonly",
                                    values=["htdemucs_6s", "htdemucs", "htdemucs_ft",
                                            "mdx_extra", "mdx_extra_q"])
        model_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(frame, text="Device:").grid(row=0, column=2, sticky=tk.W, padx=(15, 0), pady=2)
        self._device_var = tk.StringVar(value="auto")
        device_combo = ttk.Combobox(frame, textvariable=self._device_var, width=8,
                                     state="readonly", values=["auto", "cpu", "cuda"])
        device_combo.grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)

        # Row 1: Instruments
        ttk.Label(frame, text="Instruments:").grid(row=1, column=0, sticky=tk.W, pady=2)
        inst_frame = ttk.Frame(frame)
        inst_frame.grid(row=1, column=1, columnspan=3, sticky=tk.W, padx=5, pady=2)

        self._guitar_var = tk.BooleanVar(value=True)
        self._bass_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(inst_frame, text="Guitar", variable=self._guitar_var).pack(
            side=tk.LEFT, padx=(0, 15))
        ttk.Checkbutton(inst_frame, text="Bass", variable=self._bass_var).pack(
            side=tk.LEFT)

        # Row 2: Output formats
        ttk.Label(frame, text="Output Formats:").grid(row=2, column=0, sticky=tk.W, pady=2)
        fmt_frame = ttk.Frame(frame)
        fmt_frame.grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=5, pady=2)

        self._fmt_ascii_var = tk.BooleanVar(value=True)
        self._fmt_gp5_var = tk.BooleanVar(value=False)
        self._fmt_xml_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(fmt_frame, text="ASCII Tab (.txt)",
                         variable=self._fmt_ascii_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(fmt_frame, text="Guitar Pro (.gp5)",
                         variable=self._fmt_gp5_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(fmt_frame, text="MusicXML (.xml)",
                         variable=self._fmt_xml_var).pack(side=tk.LEFT)

        # Row 3: Tab options
        ttk.Label(frame, text="Tab Options:").grid(row=3, column=0, sticky=tk.W, pady=2)
        tab_opts_frame = ttk.Frame(frame)
        tab_opts_frame.grid(row=3, column=1, columnspan=3, sticky=tk.W, padx=5, pady=2)

        self._invert_strings_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab_opts_frame, text="Invert string order (low E on top)",
                         variable=self._invert_strings_var).pack(side=tk.LEFT)

        # Row 4: Quality preset
        ttk.Label(frame, text="Quality Preset:").grid(row=4, column=0, sticky=tk.W, pady=2)
        preset_frame = ttk.Frame(frame)
        preset_frame.grid(row=4, column=1, columnspan=3, sticky=tk.W, padx=5, pady=2)

        self._quality_var = tk.StringVar(value="fast")
        quality_combo = ttk.Combobox(preset_frame, textvariable=self._quality_var, width=14,
                                      state="readonly",
                                      values=["fast", "balanced", "high", "extreme"])
        quality_combo.pack(side=tk.LEFT)
        quality_combo.bind("<<ComboboxSelected>>", self._on_quality_changed)

        self._quality_desc = ttk.Label(preset_frame, text="Quick results, lower accuracy",
                                        style="Subtitle.TLabel")
        self._quality_desc.pack(side=tk.LEFT, padx=(10, 0))

        # Row 5: Thresholds
        ttk.Label(frame, text="Onset Threshold:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self._onset_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(frame, from_=0.1, to=0.9, increment=0.05,
                     textvariable=self._onset_var, width=6).grid(
            row=5, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(frame, text="Frame Threshold:").grid(row=5, column=2, sticky=tk.W,
                                                         padx=(15, 0), pady=2)
        self._frame_var = tk.DoubleVar(value=0.3)
        ttk.Spinbox(frame, from_=0.1, to=0.9, increment=0.05,
                     textvariable=self._frame_var, width=6).grid(
            row=5, column=3, sticky=tk.W, padx=5, pady=2)

        # Row 6: Advanced toggle
        self._advanced_visible = tk.BooleanVar(value=False)
        self._advanced_toggle = ttk.Checkbutton(
            frame, text="Show advanced quality settings",
            variable=self._advanced_visible,
            command=self._toggle_advanced,
        )
        self._advanced_toggle.grid(row=6, column=0, columnspan=4, sticky=tk.W, pady=(5, 2))

        # Advanced settings frame (hidden by default)
        self._advanced_frame = ttk.Frame(frame)

        # Demucs params
        ttk.Label(self._advanced_frame, text="Demucs Shifts:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self._shifts_var = tk.IntVar(value=0)
        ttk.Spinbox(self._advanced_frame, from_=0, to=10, increment=1,
                     textvariable=self._shifts_var, width=6).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(self._advanced_frame, text="Demucs Overlap:").grid(row=0, column=2, sticky=tk.W,
                                                                      padx=(15, 0), pady=2)
        self._overlap_var = tk.DoubleVar(value=0.25)
        ttk.Spinbox(self._advanced_frame, from_=0.0, to=0.99, increment=0.05,
                     textvariable=self._overlap_var, width=6).grid(
            row=0, column=3, sticky=tk.W, padx=5, pady=2)

        # Audio-to-MIDI params
        ttk.Label(self._advanced_frame, text="Hop Length:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self._hop_var = tk.IntVar(value=512)
        ttk.Combobox(self._advanced_frame, textvariable=self._hop_var, width=6,
                      state="readonly", values=[128, 256, 512, 1024]).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(self._advanced_frame, text="FFT Size:").grid(row=1, column=2, sticky=tk.W,
                                                                 padx=(15, 0), pady=2)
        self._nfft_var = tk.IntVar(value=2048)
        ttk.Combobox(self._advanced_frame, textvariable=self._nfft_var, width=6,
                      state="readonly", values=[1024, 2048, 4096, 8192]).grid(
            row=1, column=3, sticky=tk.W, padx=5, pady=2)

        ttk.Label(self._advanced_frame, text="Onset Window:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self._onset_window_var = tk.IntVar(value=4)
        ttk.Spinbox(self._advanced_frame, from_=1, to=16, increment=1,
                     textvariable=self._onset_window_var, width=6).grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=2)

    def _build_progress_section(self, parent):
        """Build progress bar and status."""
        frame = ttk.Frame(parent, style="TFrame")
        frame.pack(fill=tk.X, padx=10, pady=5)

        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_bar = ttk.Progressbar(
            frame, variable=self._progress_var, maximum=100, mode="determinate"
        )
        self._progress_bar.pack(fill=tk.X, pady=(0, 5))

        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(frame, textvariable=self._status_var, style="Status.TLabel").pack(
            anchor=tk.W)

    def _build_controls(self, parent):
        """Build start/cancel buttons."""
        frame = ttk.Frame(parent, style="TFrame")
        frame.pack(fill=tk.X, padx=10, pady=5)

        self._start_btn = ttk.Button(
            frame, text="Start Processing", style="Accent.TButton",
            command=self._start_processing
        )
        self._start_btn.pack(side=tk.LEFT, padx=(0, 10))

        self._cancel_btn = ttk.Button(
            frame, text="Cancel", command=self._cancel_processing, state="disabled"
        )
        self._cancel_btn.pack(side=tk.LEFT)

        # Open output folder button
        self._open_output_btn = ttk.Button(
            frame, text="Open Output Folder", command=self._open_output_folder
        )
        self._open_output_btn.pack(side=tk.RIGHT)

    def _build_log_section(self, parent):
        """Build the activity log."""
        frame = ttk.LabelFrame(parent, text="Activity Log", padding=5)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self._log_text = tk.Text(
            frame,
            height=12,
            bg=COLORS["bg_entry"],
            fg=COLORS["fg_secondary"],
            font=(FONT_MONO, 9),
            wrap=tk.WORD,
            borderwidth=0,
            state="disabled",
        )
        log_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL,
                                    command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_scroll.set)

        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure log text tags for colored output
        self._log_text.tag_configure("info", foreground=COLORS["fg_secondary"])
        self._log_text.tag_configure("success", foreground=COLORS["success"])
        self._log_text.tag_configure("warning", foreground=COLORS["warning"])
        self._log_text.tag_configure("error", foreground=COLORS["error"])

    def _build_player_tab(self, parent):
        """Build the MIDI player tab."""
        self._midi_player = MidiPlayerWidget(parent, style="Section.TFrame")
        self._midi_player.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # --- Quality preset helpers ---

    def _on_quality_changed(self, event=None):
        """Update advanced params and description when preset changes."""
        preset = self._quality_var.get()
        params = QUALITY_PRESETS.get(preset, QUALITY_PRESETS["fast"])

        self._shifts_var.set(params["demucs_shifts"])
        self._overlap_var.set(params["demucs_overlap"])
        self._hop_var.set(params["hop_length"])
        self._nfft_var.set(params["n_fft"])
        self._onset_window_var.set(params["onset_window"])

        descriptions = {
            "fast": "Quick results, lower accuracy",
            "balanced": "Good balance of speed and quality",
            "high": "Best accuracy, significantly slower",
            "extreme": "Maximum accuracy, very slow — best for final output",
        }
        self._quality_desc.configure(text=descriptions.get(preset, ""))

    def _toggle_advanced(self):
        """Show/hide advanced quality settings."""
        if self._advanced_visible.get():
            self._advanced_frame.grid(row=7, column=0, columnspan=4, sticky=tk.EW, pady=(2, 5))
        else:
            self._advanced_frame.grid_remove()

    # --- File selection handlers ---

    def _select_input_file(self):
        exts = " ".join(f"*{ext}" for ext in SUPPORTED_AUDIO_EXTENSIONS)
        filepath = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", exts), ("All files", "*.*")],
        )
        if filepath:
            self._input_var.set(filepath)

    def _select_input_folder(self):
        folder = filedialog.askdirectory(title="Select folder with audio files")
        if folder:
            self._input_var.set(folder)

    def _select_output_folder(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self._output_var.set(folder)

    def _open_output_folder(self):
        output = self._output_var.get()
        if output and Path(output).exists():
            import os
            os.startfile(output)
        else:
            messagebox.showwarning("Warning", "Output folder does not exist yet.")

    # --- Processing ---

    def _get_selected_instruments(self) -> list[str]:
        instruments = []
        if self._guitar_var.get():
            instruments.append("guitar")
        if self._bass_var.get():
            instruments.append("bass")
        return instruments

    def _get_selected_formats(self) -> list[str]:
        formats = []
        if self._fmt_ascii_var.get():
            formats.append("ascii")
        if self._fmt_gp5_var.get():
            formats.append("gp5")
        if self._fmt_xml_var.get():
            formats.append("musicxml")
        return formats

    def _validate_inputs(self) -> bool:
        input_path = self._input_var.get().strip()
        output_path = self._output_var.get().strip()

        if not input_path:
            messagebox.showerror("Error", "Please select an input file or folder.")
            return False
        if not output_path:
            messagebox.showerror("Error", "Please select an output folder.")
            return False
        if not Path(input_path).exists():
            messagebox.showerror("Error", f"Input path does not exist:\n{input_path}")
            return False
        if not self._get_selected_instruments():
            messagebox.showerror("Error", "Please select at least one instrument.")
            return False
        if not self._get_selected_formats():
            messagebox.showerror("Error", "Please select at least one output format.")
            return False
        return True

    def _start_processing(self):
        if self._processing:
            return
        if not self._validate_inputs():
            return

        self._processing = True
        self._cancel_requested = False
        self._start_btn.configure(state="disabled")
        self._cancel_btn.configure(state="normal")
        self._progress_var.set(0)

        input_path = Path(self._input_var.get().strip())
        output_dir = Path(self._output_var.get().strip())

        # Collect audio files to process
        if input_path.is_file():
            audio_files = [input_path]
        else:
            audio_files = []
            for ext in SUPPORTED_AUDIO_EXTENSIONS:
                audio_files.extend(input_path.glob(f"*{ext}"))
            audio_files.sort()

        if not audio_files:
            self._log("No supported audio files found.", "warning")
            self._finish_processing()
            return

        self._log(f"Found {len(audio_files)} audio file(s) to process.", "info")

        opts = PipelineOptions(
            model=self._model_var.get(),
            device=self._device_var.get(),
            instruments=self._get_selected_instruments(),
            formats=self._get_selected_formats(),
            onset_threshold=self._onset_var.get(),
            frame_threshold=self._frame_var.get(),
            invert_strings=self._invert_strings_var.get(),
            demucs_shifts=self._shifts_var.get(),
            demucs_overlap=self._overlap_var.get(),
            hop_length=self._hop_var.get(),
            n_fft=self._nfft_var.get(),
            onset_window=self._onset_window_var.get(),
        )

        # Run in background thread
        thread = threading.Thread(
            target=self._process_worker,
            args=(audio_files, output_dir, opts),
            daemon=True,
        )
        thread.start()

    def _process_worker(self, audio_files: list[Path], output_dir: Path,
                         opts: PipelineOptions):
        """Background worker for processing."""
        total = len(audio_files)

        for i, audio_file in enumerate(audio_files):
            if self._cancel_requested:
                self._log_queue.put(("Cancelled by user.", "warning"))
                break

            self._log_queue.put((f"[{i+1}/{total}] Processing: {audio_file.name}", "info"))

            # Always create a subfolder per file
            file_output = output_dir / audio_file.stem

            try:
                # Set up a logging handler that routes to the GUI
                gui_handler = _QueueLogHandler(self._log_queue)
                gui_handler.setLevel(logging.INFO)
                tab_logger = logging.getLogger("tabgrabber")
                tab_logger.addHandler(gui_handler)

                result = process(audio_file, file_output, opts)

                tab_logger.removeHandler(gui_handler)

                tab_count = sum(len(v) for v in result.tabs.values())
                self._log_queue.put((
                    f"Completed: {audio_file.name} - "
                    f"{len(result.midi)} MIDI, {tab_count} tab file(s)",
                    "success"
                ))

            except Exception as e:
                self._log_queue.put((f"Error processing {audio_file.name}: {e}", "error"))

            progress = ((i + 1) / total) * 100
            self._log_queue.put(("__progress__", progress))

        self._log_queue.put(("__done__", None))

    def _cancel_processing(self):
        self._cancel_requested = True
        self._status_var.set("Cancelling...")

    def _finish_processing(self):
        self._processing = False
        self._start_btn.configure(state="normal")
        self._cancel_btn.configure(state="disabled")
        self._status_var.set("Ready")

        # Auto-load MIDI files into player
        output = self._output_var.get().strip()
        if output:
            self._midi_player.load_output_dir(Path(output))

    # --- Logging ---

    def _log(self, message: str, tag: str = "info"):
        """Append message to the log widget (main thread only)."""
        self._log_text.configure(state="normal")
        self._log_text.insert(tk.END, message + "\n", tag)
        self._log_text.see(tk.END)
        self._log_text.configure(state="disabled")

    def _poll_log_queue(self):
        """Check for log messages from worker thread."""
        try:
            while True:
                msg, data = self._log_queue.get_nowait()
                if msg == "__progress__":
                    self._progress_var.set(data)
                    self._status_var.set(f"Processing... {data:.0f}%")
                elif msg == "__done__":
                    self._log("Processing finished.", "success")
                    self._finish_processing()
                else:
                    self._log(msg, data)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    # --- Lifecycle ---

    def _on_close(self):
        if self._processing:
            if not messagebox.askyesno("Confirm", "Processing is in progress. Quit anyway?"):
                return
            self._cancel_requested = True

        self._midi_player.cleanup()
        self.root.destroy()

    def run(self):
        """Start the GUI event loop."""
        self.root.mainloop()


class _QueueLogHandler(logging.Handler):
    """Logging handler that sends records to a queue for GUI display."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self._queue = log_queue

    def emit(self, record):
        msg = self.format(record)
        if record.levelno >= logging.ERROR:
            tag = "error"
        elif record.levelno >= logging.WARNING:
            tag = "warning"
        else:
            tag = "info"
        self._queue.put((msg, tag))
