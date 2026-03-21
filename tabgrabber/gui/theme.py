"""Dark theme configuration for TabGrabber GUI."""

import tkinter as tk
from tkinter import ttk

# Color palette (matches RocksmithGuitarMute style)
COLORS = {
    "bg": "#1e1e1e",
    "bg_section": "#2d2d2d",
    "bg_header": "#333333",
    "bg_entry": "#404040",
    "bg_button": "#404040",
    "bg_button_hover": "#505050",
    "bg_accent": "#0078d4",
    "bg_accent_hover": "#1a8ae8",
    "bg_danger": "#602020",
    "fg": "#ffffff",
    "fg_secondary": "#cccccc",
    "fg_disabled": "#666666",
    "border": "#555555",
    "trough": "#404040",
    "progress": "#0078d4",
    "success": "#2ea043",
    "warning": "#d29922",
    "error": "#f85149",
}

FONT_FAMILY = "Segoe UI"
FONT_MONO = "Consolas"


def apply_dark_theme(root: tk.Tk) -> ttk.Style:
    """Apply dark theme to the application."""
    root.configure(bg=COLORS["bg"])

    style = ttk.Style(root)
    style.theme_use("clam")

    # Frame
    style.configure("TFrame", background=COLORS["bg"])
    style.configure("Section.TFrame", background=COLORS["bg_section"])
    style.configure("Header.TFrame", background=COLORS["bg_header"])

    # Label
    style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["fg"],
                     font=(FONT_FAMILY, 10))
    style.configure("Title.TLabel", font=(FONT_FAMILY, 18, "bold"))
    style.configure("Subtitle.TLabel", foreground=COLORS["fg_secondary"],
                     font=(FONT_FAMILY, 9))
    style.configure("Header.TLabel", background=COLORS["bg_header"],
                     foreground=COLORS["fg"], font=(FONT_FAMILY, 11, "bold"))
    style.configure("Section.TLabel", background=COLORS["bg_section"],
                     foreground=COLORS["fg"])
    style.configure("Status.TLabel", foreground=COLORS["fg_secondary"],
                     font=(FONT_FAMILY, 9))

    # Button
    style.configure("TButton", background=COLORS["bg_button"],
                     foreground=COLORS["fg"], font=(FONT_FAMILY, 9),
                     borderwidth=1, relief="flat", padding=(10, 5))
    style.map("TButton",
              background=[("active", COLORS["bg_button_hover"]),
                          ("disabled", COLORS["bg"])],
              foreground=[("disabled", COLORS["fg_disabled"])])

    style.configure("Accent.TButton", background=COLORS["bg_accent"],
                     foreground=COLORS["fg"], font=(FONT_FAMILY, 10, "bold"))
    style.map("Accent.TButton",
              background=[("active", COLORS["bg_accent_hover"]),
                          ("disabled", COLORS["trough"])])

    style.configure("Success.TButton", background=COLORS["success"],
                     foreground=COLORS["fg"])
    style.configure("Danger.TButton", background=COLORS["bg_danger"],
                     foreground=COLORS["fg"])

    # Entry
    style.configure("TEntry", fieldbackground=COLORS["bg_entry"],
                     foreground=COLORS["fg"], insertcolor=COLORS["fg"],
                     borderwidth=1, relief="flat")

    # Combobox
    style.configure("TCombobox", fieldbackground=COLORS["bg_entry"],
                     foreground=COLORS["fg"], selectbackground=COLORS["bg_accent"],
                     arrowcolor=COLORS["fg"])
    style.map("TCombobox",
              fieldbackground=[("readonly", COLORS["bg_entry"])],
              selectbackground=[("readonly", COLORS["bg_accent"])])

    # Combobox dropdown
    root.option_add("*TCombobox*Listbox.background", COLORS["bg_entry"])
    root.option_add("*TCombobox*Listbox.foreground", COLORS["fg"])
    root.option_add("*TCombobox*Listbox.selectBackground", COLORS["bg_accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", COLORS["fg"])

    # Checkbutton
    style.configure("TCheckbutton", background=COLORS["bg"],
                     foreground=COLORS["fg"], font=(FONT_FAMILY, 9))
    style.configure("Section.TCheckbutton", background=COLORS["bg_section"])
    style.map("TCheckbutton",
              background=[("active", COLORS["bg"])])

    # Progressbar
    style.configure("TProgressbar", troughcolor=COLORS["trough"],
                     background=COLORS["progress"], thickness=20)

    # Spinbox
    style.configure("TSpinbox", fieldbackground=COLORS["bg_entry"],
                     foreground=COLORS["fg"], arrowcolor=COLORS["fg"],
                     borderwidth=1)

    # Scale (slider)
    style.configure("TScale", background=COLORS["bg"],
                     troughcolor=COLORS["trough"])

    # Labelframe
    style.configure("TLabelframe", background=COLORS["bg_section"],
                     foreground=COLORS["fg"], borderwidth=1,
                     relief="solid")
    style.configure("TLabelframe.Label", background=COLORS["bg_section"],
                     foreground=COLORS["fg"], font=(FONT_FAMILY, 10, "bold"))

    # Notebook (tabs)
    style.configure("TNotebook", background=COLORS["bg"], borderwidth=0)
    style.configure("TNotebook.Tab", background=COLORS["bg_button"],
                     foreground=COLORS["fg"], padding=(12, 6),
                     font=(FONT_FAMILY, 9))
    style.map("TNotebook.Tab",
              background=[("selected", COLORS["bg_section"]),
                          ("active", COLORS["bg_button_hover"])],
              foreground=[("selected", COLORS["fg"])])

    return style
