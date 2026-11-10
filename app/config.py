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
    {"label": "Projects", "href": "/projects"},
]

# Works in dev and in PyInstaller builds (bundled modules get __file__ under _MEIPASS).
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
BRAND_DIR = STATIC_DIR / "brand"  # logo.svg; app-icon.svg and the files scripts/app_icons.py renders from it
APP_ICON = BRAND_DIR / "app-icon.png"

# The user's workspace: settings, database and media files.
# macOS ~/Library/Application Support/babasu, Windows %LOCALAPPDATA%\babasu,
# Linux ~/.local/share/babasu.
DATA_DIR = Path(platformdirs.user_data_dir(APP_NAME, appauthor=False))
DB_FILE = DATA_DIR / "babasu.db"
MEDIA_DIR = DATA_DIR / "media"
