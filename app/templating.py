import re
from functools import cache

from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from app import config, settings

ICONS_DIR = config.STATIC_DIR / "icons"


@cache
def icon(name: str) -> Markup:
    """Inline an SVG from static/icons, so it takes the surrounding text colour."""
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValueError(f"invalid icon name: {name!r}")
    return Markup((ICONS_DIR / f"{name}.svg").read_text())


templates = Jinja2Templates(directory=config.TEMPLATES_DIR)
templates.env.globals["app_name"] = config.APP_NAME
templates.env.globals["nav"] = config.NAV
templates.env.globals["icon"] = icon
templates.env.globals["settings"] = settings.get
templates.env.globals["sidebar_min_width"] = settings.SIDEBAR_MIN_WIDTH
templates.env.globals["sidebar_max_width"] = settings.SIDEBAR_MAX_WIDTH
