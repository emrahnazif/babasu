"""HTML page routes."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app.templating import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "pages/home.html", {"message": "Hello, world"})


@router.get("/datasets", response_class=HTMLResponse)
def datasets(request: Request):
    return templates.TemplateResponse(request, "pages/datasets.html")
