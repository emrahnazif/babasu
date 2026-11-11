"""HTML page routes."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.templating import templates

router = APIRouter()


@router.get("/")
def home():
    return RedirectResponse("/projects")


# Pages that aren't built yet: each shows a short note. A section's name is already in the
# breadcrumb (its switcher); a page outside the sections passes its own crumbs.
def _coming(request: Request, title: str, crumbs: list | None = None):
    return templates.TemplateResponse(request, "pages/coming.html", {"title": title, "crumbs": crumbs or []})


@router.get("/workflows", response_class=HTMLResponse)
def workflows(request: Request):
    return _coming(request, "Workflows")


@router.get("/studio", response_class=HTMLResponse)
def studio(request: Request):
    return _coming(request, "Studio")


@router.get("/models", response_class=HTMLResponse)
def models(request: Request):
    return _coming(request, "Models")


@router.get("/library", response_class=HTMLResponse)
def library(request: Request):
    return _coming(request, "Library", [("Library", None)])
