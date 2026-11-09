"""App-wide settings and resource paths."""

from pathlib import Path

import platformdirs

APP_NAME = "babasu"

WINDOW_SIZE = (1280, 800)
WINDOW_MIN_SIZE = (900, 600)

HOST = "127.0.0.1"

# Header navigation: label and link for each item.
NAV = [
    {"label": "Home", "href": "/"},
    {"label": "Datasets", "href": "/datasets"},
]

# Works in dev and in PyInstaller builds (bundled modules get __file__ under _MEIPASS).
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# The user's workspace: settings now, datasets and models later.
# macOS ~/Library/Application Support/babasu, Windows %LOCALAPPDATA%\babasu,
# Linux ~/.local/share/babasu.
DATA_DIR = Path(platformdirs.user_data_dir(APP_NAME, appauthor=False))
