"""App settings, stored as JSON in the workspace."""

from typing import Literal

from pydantic import BaseModel

from app import config

SETTINGS_FILE = config.DATA_DIR / "settings.json"

Theme = Literal["system", "light", "dark"]


class Settings(BaseModel):
    theme: Theme = "system"


def get() -> Settings:
    if SETTINGS_FILE.exists():
        return Settings.model_validate_json(SETTINGS_FILE.read_text())
    return Settings()


def save(settings: Settings) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_FILE.with_suffix(".tmp")
    tmp.write_text(settings.model_dump_json(indent=2))
    tmp.replace(SETTINGS_FILE)  # atomic: never leaves a half-written file
