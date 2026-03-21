"""Entry point for launching the TabGrabber GUI."""

import logging
import sys


def main():
    """Launch the TabGrabber GUI application."""
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    from tabgrabber.gui.gui_main import TabGrabberGUI
    app = TabGrabberGUI()
    app.run()


if __name__ == "__main__":
    main()
