import re
from contextlib import closing
from datetime import UTC, datetime
from functools import cache

from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from app import agent, config, db, jobs, media, settings

ICONS_DIR = config.STATIC_DIR / "icons"


@cache
def _inline_svg(path) -> Markup:
    return Markup(path.read_text())


def icon(name: str) -> Markup:
    """Inline an SVG from static/icons, so it takes the surrounding text colour."""
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValueError(f"invalid icon name: {name!r}")
    return _inline_svg(ICONS_DIR / f"{name}.svg")


def logo() -> Markup:
    """The babasu logo, inlined like the icons; where it's shown sets its colour and line width."""
    return _inline_svg(config.BRAND_DIR / "logo.svg")


def jobs_overview() -> dict:
    """Shell data for base.html: the jobs drawer and count are rendered with every page,
    so they're right before js/jobs.js runs."""
    with closing(db.connect()) as conn:
        return jobs.overview(conn)


def agent_chat() -> dict:
    """Shell data for the chat in the sidebar: the current conversation and its agent, and the
    agents to pick from, rendered with every page."""
    with closing(db.connect()) as conn:
        conversation_id = agent.current_conversation(conn)
        turns = agent.turns(conn, conversation_id)
        agent_id = agent.conversation_agent(conn, conversation_id)
    s = settings.get()
    found = settings.find_agent(agent_id, s)
    return {"turns": turns, "agents": s.agents, "agent_id": found[0].id if found else None}


def ago(timestamp: str) -> str:
    """A short "how long ago" for a stored UTC time: now, 5m, 3h, Yesterday, 4d, then the date."""
    then = datetime.fromisoformat(timestamp)
    seconds = (datetime.now(UTC) - then).total_seconds()
    if seconds < 60:
        return "now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    if seconds < 2 * 86400:
        return "Yesterday"
    if seconds < 7 * 86400:
        return f"{int(seconds // 86400)}d"
    local = then.astimezone()
    return f"{local:%b} {local.day}"


def period(timestamp: str) -> str:
    """Which stretch of time a stored UTC time falls in, for grouping lists: Today, Yesterday,
    Previous 7 days, Previous 30 days, Older (by local calendar days)."""
    days = (datetime.now().astimezone().date() - datetime.fromisoformat(timestamp).astimezone().date()).days
    if days <= 0:
        return "Today"
    if days == 1:
        return "Yesterday"
    if days < 7:
        return "Previous 7 days"
    if days < 30:
        return "Previous 30 days"
    return "Older"


templates = Jinja2Templates(directory=config.TEMPLATES_DIR)
templates.env.filters["ago"] = ago
templates.env.filters["period"] = period
templates.env.globals["app_name"] = config.APP_NAME
templates.env.globals["nav"] = config.NAV
templates.env.globals["icon"] = icon
templates.env.globals["logo"] = logo
templates.env.globals["settings"] = settings.get
templates.env.globals["sidebar_min_width"] = settings.SIDEBAR_MIN_WIDTH
templates.env.globals["sidebar_max_width"] = settings.SIDEBAR_MAX_WIDTH
templates.env.globals["jobs_overview"] = jobs_overview
templates.env.globals["image_formats"] = media.FORMATS_TEXT
templates.env.globals["image_suffixes"] = ",".join(sorted(media.SUFFIXES))
templates.env.globals["agent_chat"] = agent_chat
templates.env.globals["unified_title_bar"] = False  # macOS desktop window: app/desktop.py
