"""Settings pages. Save posts the form, then redirects back to where the user came from."""

from typing import Annotated, get_args

from fastapi import APIRouter, Body, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import ValidationError

from app import config, settings
from app.routes import safe_next
from app.templating import templates

router = APIRouter(prefix="/settings")

GENERAL = {"label": "General", "href": "/settings/general"}
SECTIONS = [GENERAL]


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
