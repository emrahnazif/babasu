import re
from contextlib import closing
from datetime import UTC, datetime
from functools import cache
from urllib.parse import urlencode

from fastapi import Request
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from app import agent, annotators, config, db, jobs, media, projects, settings

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


def under(path: str, href: str) -> bool:
    """Whether a page is at href or below it: /projects/3 is under /projects."""
    return path == href or path.startswith(href + "/")


def nav() -> list[list[dict]]:
    """The navigation: the app's sections (config.NAV), then Views and the links added to it."""
    return config.NAV + [[{"label": "Views", "href": "/views"}] + settings.nav()]


def section_of(path: str, groups: list[list[dict]]) -> dict | None:
    """The navigation item (of nav()'s groups) a path is in, if any: the one whose link is the
    longest match."""
    found = [item for group in groups for item in group if under(path, item["href"])]
    return max(found, key=lambda item: len(item["href"]), default=None)


def page_name(title: str) -> str:
    """A page's own name from its title: "COCO val · babasu" -> "COCO val"."""
    return title.strip().removesuffix(f" · {config.APP_NAME}")


def agent_chat(request: Request, page_title: str) -> dict:
    """Shell data for the chat in the sidebar: the current conversation and its agent, and the
    agents to pick from, rendered with every page. The open chat follows you: this page (its path
    and title) becomes where it was last open, so it reopens here from the history. While the agent
    is answering (`running`), the reply so far is left out: chat.js follows it from its first event."""
    with closing(db.connect()) as conn:
        conversation_id = agent.current_conversation(conn)
        if conversation_id is not None:
            page = request.url.path + (f"?{request.url.query}" if request.url.query else "")
            agent.set_page(conn, conversation_id, page, page_name(page_title))
        turns = agent.turns(conn, conversation_id)
        agent_id = agent.conversation_agent(conn, conversation_id)
    running = agent.running(conversation_id)
    if running and turns and turns[-1]["role"] == "agent":
        turns = turns[:-1]
    s = settings.get()
    found = settings.find_agent(agent_id, s)
    return {"turns": turns, "running": running, "agents": s.agents, "agent_id": found[0].id if found else None}


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


def query(params: dict, **changes) -> str:
    """A link's query string: params with changes, leaving out empty values (None, "").
    query({"q": "night"}, show="labelled") -> "?q=night&show=labelled"; query({}) -> "?"."""
    kept = {k: v for k, v in (params | changes).items() if v not in (None, "")}
    return "?" + urlencode(kept)


def plural(n: int, word: str) -> str:
    """A count with its noun: "1 image", "1,240 images", "3 classes"."""
    return f"{n:,} {word if n == 1 else word + ('es' if word.endswith('s') else 's')}"


def filesize(n: int) -> str:
    """Bytes for people, in decimal units as downloads are shown: "820 bytes", "56.1 MB", "3.45 GB"."""
    if n < 1000:
        return f"{n:,} bytes"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1000
        if n < 1000 or unit == "TB":
            break
    return f"{n:.{0 if n >= 100 else 1 if n >= 10 else 2}f} {unit}"


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


def options(items, value: str, label: str) -> list[tuple]:
    """(value, label) pairs from dicts or objects, for the select() macro: models | options("id", "name")."""
    get = lambda item, key: item[key] if isinstance(item, dict) else getattr(item, key)
    return [(get(item, value), get(item, label)) for item in items]


templates = Jinja2Templates(directory=config.TEMPLATES_DIR)
templates.env.filters["options"] = options
templates.env.filters["ago"] = ago
templates.env.filters["period"] = period
templates.env.filters["filesize"] = filesize
templates.env.filters["plural"] = plural
templates.env.globals["app_name"] = config.APP_NAME
templates.env.globals["nav"] = nav
templates.env.globals["section_of"] = section_of
templates.env.globals["under"] = under
templates.env.globals["query"] = query
templates.env.globals["icon"] = icon
templates.env.globals["logo"] = logo
templates.env.globals["settings"] = settings.get
templates.env.globals["sidebar_min_width"] = settings.SIDEBAR_MIN_WIDTH
templates.env.globals["sidebar_max_width"] = settings.SIDEBAR_MAX_WIDTH
templates.env.globals["jobs_overview"] = jobs_overview
templates.env.globals["image_formats"] = media.FORMATS_TEXT
templates.env.globals["tasks"] = projects.TASKS
templates.env.globals["annotators"] = annotators.FINDERS
templates.env.globals["image_suffixes"] = ",".join(sorted(media.SUFFIXES))
templates.env.globals["agent_chat"] = agent_chat
templates.env.globals["unified_title_bar"] = False  # macOS desktop window: app/desktop.py
