"""Double-click this file to launch TabGrabber GUI (no console window)."""

import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from tabgrabber.gui.launch_gui import main

main()
