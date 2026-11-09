"""App-wide settings and resource paths."""

from pathlib import Path

APP_NAME = "babasu"

WINDOW_SIZE = (1280, 800)
WINDOW_MIN_SIZE = (900, 600)

HOST = "127.0.0.1"

# Works in dev and in PyInstaller builds (bundled modules get __file__ under _MEIPASS).
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
