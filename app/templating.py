from contextlib import closing
from functools import cache
from urllib.parse import urlencode

from fastapi import Request
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from app import agent, annotators, config, db, formatting, jobs, media, projects, settings


@cache
def logo() -> Markup:
    """The babasu logo, inlined like the icons; where it's shown sets its colour and line width."""
    return Markup((config.BRAND_DIR / "logo.svg").read_text())


def jobs_overview() -> dict:
    """Shell data for base.html: the jobs drawer and count are rendered with every page,
    so they're right before js/jobs.js runs."""
    with closing(db.connect()) as conn:
        return jobs.overview(conn)


def under(path: str, href: str) -> bool:
    """Whether a page is at href or below it: /projects/3 is under /projects."""
    return path == href or path.startswith(href + "/")


def section_of(path: str) -> dict | None:
    """The navigation item (of config.NAV) a path is in, if any: the one whose link is the
    longest match."""
    found = [item for group in config.NAV for item in group if under(path, item["href"])]
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
        for turn in turns:
            if turn.get("project_id") is not None:
                turn["shown"] = dropped_images(conn, turn["project_id"])
            for part in turn.get("parts", []):
                if part.get("images"):
                    part["shown"] = projects.images(conn, part["images"])
        agent_id = agent.conversation_agent(conn, conversation_id)
    running = agent.running(conversation_id)
    if running and turns and turns[-1]["role"] == "agent":
        turns = turns[:-1]
    s = settings.get()
    found = settings.find_agent(agent_id, s)
    return {"turns": turns, "running": running, "agents": s.agents, "agent_id": found[0].id if found else None}


SHOWN_DROPPED = 8  # images the chat shows under the message they were dropped with; the rest are counted


def dropped_images(conn, project_id: int) -> dict:
    """The images dropped with a message, in the project made for them, as partials/chat-images.html
    draws them: the first few, and how many more."""
    first = projects.sample(conn, project_id=project_id, first=SHOWN_DROPPED)
    return projects.images(conn, first) | {"more": max(projects.image_count(conn, project_id) - SHOWN_DROPPED, 0)}


def query(params: dict, **changes) -> str:
    """A link's query string: params with changes, leaving out empty values (None, "").
    query({"q": "night"}, show="labelled") -> "?q=night&show=labelled"; query({}) -> "?"."""
    kept = {k: v for k, v in (params | changes).items() if v not in (None, "")}
    return "?" + urlencode(kept)


def options(items, value: str, label: str) -> list[tuple]:
    """(value, label) pairs from dicts or objects, for the select() macro: models | options("id", "name")."""
    get = lambda item, key: item[key] if isinstance(item, dict) else getattr(item, key)
    return [(get(item, value), get(item, label)) for item in items]


templates = Jinja2Templates(directory=config.TEMPLATES_DIR)
templates.env.filters.update(formatting.FILTERS)
templates.env.globals.update(formatting.GLOBALS)
templates.env.filters["options"] = options
templates.env.globals["app_name"] = config.APP_NAME
templates.env.globals["nav"] = config.NAV
templates.env.globals["section_of"] = section_of
templates.env.globals["under"] = under
templates.env.globals["query"] = query
templates.env.globals["logo"] = logo
templates.env.globals["settings"] = settings.get
templates.env.globals["sidebar_min_width"] = settings.SIDEBAR_MIN_WIDTH
templates.env.globals["sidebar_max_width"] = settings.SIDEBAR_MAX_WIDTH
templates.env.globals["jobs_overview"] = jobs_overview
templates.env.globals["media_formats"] = media.FORMATS_TEXT
templates.env.globals["tasks"] = projects.TASKS
templates.env.globals["annotators"] = annotators.FINDERS
templates.env.globals["media_suffixes"] = ",".join(sorted(media.SUFFIXES))
templates.env.globals["agent_chat"] = agent_chat
