"""Vision models (mozo's): the Models page (forms post, then redirect back), and the same data
and actions as JSON under /api/vision."""

import sqlite3
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from mozo import weights

from app import config, db, vision
from app.templating import templates

router = APIRouter()
DB = Annotated[sqlite3.Connection, Depends(db.get)]


def _exists(family: str, variant: str) -> None:
    if not vision.exists(family, variant):
        raise HTTPException(404, "mozo publishes no such model")


def _start_load(conn: sqlite3.Connection, family: str, variant: str) -> int:
    _exists(family, variant)
    return vision.start_load(conn, family, variant)


def _remove(family: str, variant: str) -> int:
    _exists(family, variant)
    try:
        return vision.remove(family, variant)
    except weights.WeightsError as e:
        raise HTTPException(409, str(e)) from None


def _back(error: str | None = None) -> RedirectResponse:
    return RedirectResponse("/models" + (f"?{urlencode({'error': error})}" if error else ""), status_code=303)


# --- Pages ---


@router.get("/models", response_class=HTMLResponse)
def models_page(request: Request, conn: DB, error: str | None = None):
    context = {
        "catalogue": vision.catalogue(conn),
        "loaded": vision.loaded(),
        "device": config.DEVICE,
        "error": error,
    }
    return templates.TemplateResponse(request, "models/list.html", context)


@router.post("/models/{family}/{variant}/download")
def download(family: str, variant: str, conn: DB):
    """Starts the download job; progress shows in the jobs drawer (sent in the background with JS)."""
    _exists(family, variant)
    vision.start_download(conn, family, variant)
    return _back()


@router.post("/models/{family}/{variant}/load")
def load(family: str, variant: str, conn: DB):
    _start_load(conn, family, variant)
    return _back()


@router.post("/models/{family}/{variant}/remove")
def remove(family: str, variant: str):
    _exists(family, variant)
    try:
        vision.remove(family, variant)
    except weights.WeightsError as e:
        return _back(str(e))
    return _back()


@router.post("/models/{family}/{variant}/unload")
def unload(family: str, variant: str):
    vision.unload(family, variant)
    return _back()


@router.post("/models/unload-all")
def unload_all():
    vision.unload_all()
    return _back()


# --- API ---


@router.get("/api/vision/models")
def api_models(conn: DB) -> list[dict]:
    """Every model mozo publishes, by family, with each variant's licence, size, bytes left to download and state."""
    return vision.catalogue(conn)


@router.post("/api/vision/models/{family}/{variant}/download", status_code=202)
def api_download(family: str, variant: str, conn: DB) -> dict:
    """Starts a job that downloads the model's files; follow it at /api/jobs/{job_id}."""
    _exists(family, variant)
    return {"job_id": vision.start_download(conn, family, variant)}


@router.delete("/api/vision/models/{family}/{variant}/weights")
def api_remove(family: str, variant: str) -> dict:
    """Deletes the model's downloaded files; 409 while it's downloading."""
    return {"freed": _remove(family, variant)}


@router.post("/api/vision/models/{family}/{variant}/load", status_code=202)
def api_load(family: str, variant: str, conn: DB) -> dict:
    """Starts a job that loads the model, after downloading it if it isn't downloaded."""
    return {"job_id": _start_load(conn, family, variant)}


@router.get("/api/vision/loaded")
def api_loaded() -> list[dict]:
    """The loaded models, with device, runtime, when last used and estimated memory in bytes."""
    return vision.loaded()


@router.delete("/api/vision/loaded/{family}/{variant}")
def api_unload(family: str, variant: str) -> dict:
    """Unloads a model from memory (it loads again when something uses it); {unloaded: true/false}."""
    return {"unloaded": vision.unload(family, variant)}


@router.delete("/api/vision/loaded")
def api_unload_all() -> dict:
    """Unloads every model from memory, e.g. to free it for something else; {unloaded: how many}."""
    return {"unloaded": vision.unload_all()}
