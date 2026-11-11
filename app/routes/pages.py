"""HTML page routes."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.templating import templates

router = APIRouter()


@router.get("/")
def home():
    return RedirectResponse("/projects")


# Sections that aren't built yet: each shows a short note (its name is the breadcrumb's switcher).
def _coming(request: Request, title: str):
    return templates.TemplateResponse(request, "pages/coming.html", {"title": title})


@router.get("/workflows", response_class=HTMLResponse)
def workflows(request: Request):
    return _coming(request, "Workflows")


@router.get("/studio", response_class=HTMLResponse)
def studio(request: Request):
    return _coming(request, "Studio")


@router.get("/models", response_class=HTMLResponse)
def models(request: Request):
    return _coming(request, "Models")


@router.get("/data-lake", response_class=HTMLResponse)
def data_lake(request: Request):
    return _coming(request, "Data lake")
