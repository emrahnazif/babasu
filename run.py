"""Entry point for development (`python run.py`) and for PyInstaller."""

import os
import sys

# Windowed PyInstaller builds have no console: stdout/stderr are None, and
# anything that writes or checks isatty() (uvicorn's logging, print) crashes.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

from app.desktop import main  # noqa: E402

if __name__ == "__main__":
    main()
