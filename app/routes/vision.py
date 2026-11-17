"""Vision models (mozo's): the Models page (forms post, then redirect back), and the same data
and actions as JSON under /api/vision."""

import sqlite3
from typing import Annotated, Literal
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from mozo import weights
from pydantic import Field

from app import config, db, vision
from app.formatting import filesize, plural
from app.routes import Answer, Done, Started, started
from app.templating import templates

router = APIRouter()
DB = Annotated[sqlite3.Connection, Depends(db.get)]


# --- What the API answers ---


class VariantListed(Answer):
    variant: str = Field(description="Which of the family's models, e.g. tiny.")
    state: Literal["not downloaded", "downloading", "downloaded", "loading", "loaded"] = Field(
        description="downloading and loading: a job is under way.")
    to_download: int = Field(description="Bytes still to download for this computer (0 once downloaded).")


class ModelVariant(VariantListed):
    license: str = Field(description="Its licence.")
    size: int = Field(description="Its weights, in bytes.")


class FamilyListed(Answer):
    family: str = Field(description="The model family, e.g. grounding_dino.")
    task: str = Field(description="What it does, e.g. object_detection.")
    variants: list[VariantListed]


class ModelFamily(FamilyListed):
    description: str = Field(description="What it is and how it works, as mozo describes it.")
    variants: list[ModelVariant]


class LoadedModel(Answer):
    family: str
    variant: str
    device: str = Field(description="What it runs on, e.g. mps or cpu.")
    runtime: str | None = Field(description="What runs it, e.g. torch-fp32 or coreml-fp16; null if the model doesn't say.")
    last_used: str = Field(description="When something last used it, UTC (ISO 8601).")
    size: int | None = Field(description="Its memory, estimated from its weights' size in bytes; null if unknown.")


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


@router.get("/api/vision/models", response_model=list[FamilyListed])
def api_models(conn: DB, task: Annotated[str | None, Query(description="Only the models for one task, e.g. object_detection.")] = None) -> list[dict]:
    """Every model mozo publishes (?task= those for one task), by family, with each variant's state
    and bytes left to download (a family's answer has what it is, and each variant's licence and
    size). 422 for a task none is for, naming the tasks."""
    if task is not None and task not in vision.tasks():
        raise HTTPException(422, f"No model is for {task!r}; the tasks: {', '.join(vision.tasks())}")
    return [{"family": f["family"], "task": f["task"],
             "variants": [{k: v[k] for k in VariantListed.model_fields} for v in f["variants"]]}
            for f in vision.catalogue(conn, task)]


@router.get("/api/vision/models/{family}", response_model=ModelFamily)
def api_model_family(family: str, conn: DB) -> dict:
    """One model family: what it is and how it works, its task, and each variant's licence, size,
    state and bytes left to download."""
    found = vision.catalogue(conn, family=family)
    if not found:
        raise HTTPException(404, "mozo publishes no such model family")
    return found[0]


class Freed(Done):
    freed: int = Field(description="Bytes deleted.")


class Unloaded(Done):
    unloaded: int = Field(description="How many models it unloaded.")


@router.post("/api/vision/models/{family}/{variant}/download", status_code=202, response_model=Started,
             response_model_exclude_unset=True)
def api_download(family: str, variant: str, conn: DB) -> dict:
    """Starts a job that downloads the model's files; follow it at /api/jobs/{job_id}."""
    _exists(family, variant)
    size = weights.download_size(family, variant, device=config.DEVICE)
    job_id = vision.start_download(conn, family, variant)
    return {"job_id": job_id, "did": f"Started downloading {family} {variant} ({filesize(size)})" if size
                                      else f"{family} {variant} is downloaded already"}


@router.delete("/api/vision/models/{family}/{variant}/weights", response_model=Freed)
def api_remove(family: str, variant: str) -> dict:
    """Deletes the model's downloaded files; 409 while it's downloading."""
    freed = _remove(family, variant)
    return {"freed": freed, "did": f"Deleted {family} {variant}'s files ({filesize(freed)})"}


@router.post("/api/vision/models/{family}/{variant}/load", status_code=202, response_model=Started,
             response_model_exclude_unset=True)
def api_load(family: str, variant: str, conn: DB) -> dict:
    """Starts a job that loads the model, after downloading it if it isn't downloaded."""
    job_id = _start_load(conn, family, variant)
    return {"job_id": job_id} | started(conn, job_id, f"Started loading {family} {variant}")


@router.get("/api/vision/loaded", response_model=list[LoadedModel])
def api_loaded() -> list[dict]:
    """The loaded models, with device, runtime, when last used and estimated memory in bytes."""
    return vision.loaded()


@router.delete("/api/vision/loaded/{family}/{variant}", response_model=Unloaded)
def api_unload(family: str, variant: str) -> dict:
    """Unloads a model from memory (it loads again when something uses it)."""
    unloaded = int(vision.unload(family, variant))
    return {"unloaded": unloaded, "did": f"Unloaded {family} {variant}" if unloaded else f"{family} {variant} wasn't loaded"}


@router.delete("/api/vision/loaded", response_model=Unloaded)
def api_unload_all() -> dict:
    """Unloads every model from memory, e.g. to free it for something else."""
    unloaded = vision.unload_all()
    return {"unloaded": unloaded, "did": f"Unloaded {plural(unloaded, 'model')}"}
