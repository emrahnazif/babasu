"""Settings pages. Save posts the form, then redirects back to where the user came from."""

from typing import Annotated, get_args
from urllib.parse import urlencode

from fastapi import APIRouter, Body, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import ValidationError

from app import agent, config, settings
from app.routes import safe_next
from app.templating import templates

router = APIRouter(prefix="/settings")

GENERAL = {"label": "General", "href": "/settings/general"}
AGENT = {"label": "Agent", "href": "/settings/agent"}
SECTIONS = [GENERAL, AGENT]


@router.get("/general", response_class=HTMLResponse)
def general(request: Request, next: str | None = None):
    return templates.TemplateResponse(
        request,
        "settings/general.html",
        {
            "sections": SECTIONS,
            "section": GENERAL,
            "next": safe_next(next),
            "current": settings.get(),
            "themes": get_args(settings.Theme),
            "data_dir": config.DATA_DIR,
        },
    )


@router.post("/general")
def save_general(
    theme: Annotated[settings.Theme, Form()],
    next: Annotated[str | None, Form()] = None,
):
    settings.update(theme=theme)
    return RedirectResponse(safe_next(next), status_code=303)


@router.get("/agent", response_class=HTMLResponse)
def agent_page(request: Request, next: str | None = None, test: str | None = None, ok: bool = False):
    return templates.TemplateResponse(
        request,
        "settings/agent.html",
        {"sections": SECTIONS, "section": AGENT, "next": safe_next(next), "current": settings.get(),
         "test": test, "test_ok": ok},
    )


@router.post("/agent")
def save_agent(
    base_url: Annotated[str, Form()] = "",
    model: Annotated[str, Form()] = "",
    api_key: Annotated[str, Form()] = "",
    action: Annotated[str, Form()] = "save",  # "test": save, then try the connection
    next: Annotated[str | None, Form()] = None,
):
    changes = {"agent_base_url": base_url.strip(), "agent_model": model.strip()}
    if api_key.strip():  # blank keeps the saved key, which the form never shows
        changes["agent_api_key"] = api_key.strip()
    settings.update(**changes)
    if action != "test":
        return RedirectResponse(safe_next(next), status_code=303)
    ok, message = agent.test_connection()
    query = urlencode({"next": safe_next(next), "test": message, "ok": ok})
    return RedirectResponse(f"/settings/agent?{query}", status_code=303)


SIDEBAR_FIELDS = {"sidebar_width", "sidebar_collapsed"}


@router.post("/sidebar", status_code=204)
def save_sidebar(change: dict = Body()):
    """Called from sidebar.js after a resize or collapse, so the next page renders the same way."""
    if not change or not change.keys() <= SIDEBAR_FIELDS:
        raise HTTPException(422, f"expected only {sorted(SIDEBAR_FIELDS)}")
    try:
        settings.update(**change)
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False))
