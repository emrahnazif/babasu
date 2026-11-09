"""Settings pages. Save posts the form, then redirects back to where the user came from."""

from typing import Annotated, get_args

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app import config, settings
from app.templating import templates

router = APIRouter(prefix="/settings")

GENERAL = {"label": "General", "href": "/settings/general"}
SECTIONS = [GENERAL]


def _safe_next(next: str | None) -> str:
    """Only allow local paths as the return target."""
    if next and next.startswith("/") and not next.startswith("//"):
        return next
    return "/"


@router.get("/general", response_class=HTMLResponse)
def general(request: Request, next: str | None = None):
    return templates.TemplateResponse(
        request,
        "settings/general.html",
        {
            "sections": SECTIONS,
            "section": GENERAL,
            "next": _safe_next(next),
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
    settings.save(settings.get().model_copy(update={"theme": theme}))
    return RedirectResponse(_safe_next(next), status_code=303)
