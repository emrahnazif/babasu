import re
from contextlib import closing
from functools import cache

from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from app import agent, config, db, jobs, media, settings

ICONS_DIR = config.STATIC_DIR / "icons"


@cache
def icon(name: str) -> Markup:
    """Inline an SVG from static/icons, so it takes the surrounding text colour."""
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValueError(f"invalid icon name: {name!r}")
    return Markup((ICONS_DIR / f"{name}.svg").read_text())


def jobs_overview() -> dict:
    """Shell data for base.html: the jobs drawer and count are rendered with every page,
    so they're right before js/jobs.js runs."""
    with closing(db.connect()) as conn:
        return jobs.overview(conn)


def agent_chat() -> dict:
    """Shell data for the chat in the sidebar: the current conversation, rendered with every page."""
    with closing(db.connect()) as conn:
        turns = agent.turns(conn, agent.latest_conversation(conn))
    return {"turns": turns, "configured": agent.configured()}


templates = Jinja2Templates(directory=config.TEMPLATES_DIR)
templates.env.globals["app_name"] = config.APP_NAME
templates.env.globals["nav"] = config.NAV
templates.env.globals["icon"] = icon
templates.env.globals["settings"] = settings.get
templates.env.globals["sidebar_min_width"] = settings.SIDEBAR_MIN_WIDTH
templates.env.globals["sidebar_max_width"] = settings.SIDEBAR_MAX_WIDTH
templates.env.globals["jobs_overview"] = jobs_overview
templates.env.globals["image_formats"] = media.FORMATS_TEXT
templates.env.globals["image_suffixes"] = ",".join(sorted(media.SUFFIXES))
templates.env.globals["agent_chat"] = agent_chat
