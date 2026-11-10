"""App settings, stored as JSON in the workspace."""

from typing import Literal

from pydantic import BaseModel, Field

from app import config

SETTINGS_FILE = config.DATA_DIR / "settings.json"

Theme = Literal["system", "light", "dark"]


SIDEBAR_MIN_WIDTH = 240
SIDEBAR_MAX_WIDTH = 600


class Settings(BaseModel):
    theme: Theme = "system"
    # Set by dragging the sidebar border and the collapse button, not by the Settings form.
    sidebar_width: int = Field(320, ge=SIDEBAR_MIN_WIDTH, le=SIDEBAR_MAX_WIDTH)
    sidebar_collapsed: bool = False


def get() -> Settings:
    if SETTINGS_FILE.exists():
        return Settings.model_validate_json(SETTINGS_FILE.read_text())
    return Settings()


def save(settings: Settings) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_FILE.with_suffix(".tmp")
    tmp.write_text(settings.model_dump_json(indent=2))
    tmp.replace(SETTINGS_FILE)  # atomic: never leaves a half-written file


def update(**changes) -> Settings:
    """Apply changes, validated by the Settings model, and save. Raises ValidationError."""
    updated = Settings.model_validate(get().model_dump() | changes)
    save(updated)
    return updated
