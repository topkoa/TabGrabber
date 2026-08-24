"""Double-click this file to launch the TabGrabber web UI (no console window)."""

import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from tabgrabber.web.server import main

main()
