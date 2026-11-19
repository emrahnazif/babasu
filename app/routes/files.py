"""Files on this computer before they're imported or attached: what a drop zone shows of the files
and folders dropped on it (js/drop-zone.js). Reading them isn't uploading them: the app runs on
this computer, and nothing is stored until the zone's form is sent."""

from contextlib import suppress
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from app import imports, media
from app.routes import UI_ONLY
from app.templating import query, templates

router = APIRouter(tags=[UI_ONLY])
SHOWN = 12  # images a drop zone shows; the rest are counted


@router.post("/files/preview", response_class=HTMLResponse)
def preview(request: Request, paths: Annotated[list[str], Form()] = []):
    """The images the paths hold (a folder's with its subfolders', as an import finds them), as a
    drop zone shows them (partials/dropped.html): how many, the first few, and what isn't found or
    isn't an image."""
    images, others, missing = imports.find_images(paths)
    return templates.TemplateResponse(request, "partials/dropped.html", {
        "count": len(images), "missing": missing, "others": [name for _, name in others],
        "shown": [{"src": "/files/thumb" + query({"path": file})} for file, _ in images[:SHOWN]]})


@router.get("/files/thumb")
def thumb(path: str) -> Response:
    """A thumbnail of an image file on this computer, made as it's asked for (nothing is kept)."""
    file = Path(path)
    if file.suffix.lower() in media.SUFFIXES:
        with suppress(media.NotAnImage):  # missing or unreadable too
            # The browser keeps it a while: a zone shows the same first images again after each drop.
            return Response(media.thumbnail(file), media_type="image/webp", headers={"Cache-Control": "private, max-age=3600"})
    raise HTTPException(404, "No such image")
