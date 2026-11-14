"""Vision models: mozo's published models, downloaded, loaded and unloaded for any use.
Workflow runs, tools and annotation all load through the one manager here.

Downloading is a job (io lane) with progress in bytes. Loading is a job too (gpu lane): it
imports torch and takes seconds, and only a downloaded model loads. Everything loads through the
load job or inside another job (a run's models); a request never loads a model (the click tool
starts the load job and says so, segment.py). The gpu lane runs one job at a time, so a load can
wait behind a run: what it waits behind is said (loading()).
"""

import json

from mozo import MODEL_REGISTRY, weights
from mozo.manager import ModelManager
from mozo.registry import get_model_info

from app import config, jobs

DOWNLOAD = "download_model"  # job kinds
LOAD = "load_model"

# The loaded models, shared by everything in the app so a model is loaded once. Load through
# load(): mozo keys a model by its arguments, so every caller must pass the same device.
models = ModelManager()


def setup() -> None:
    """Keep mozo's downloads in the workspace. Called once at startup."""
    weights.configure(cache=config.WEIGHTS_DIR)


def catalogue(conn) -> list[dict]:
    """Every model mozo publishes, by family: its task and description, and for each variant
    its licence, the size of its weights, the bytes still to download for config.DEVICE (the
    weights and their small companions: licence, labels; 0 once downloaded), and its state:
    "downloading" or "loading" (a job is under way), "loaded", "downloaded" or "not downloaded".
    Read from mozo's registry and manifest, so nothing is downloaded and torch isn't imported."""
    under_way = busy(conn)
    in_memory = {(m["family"], m["variant"]) for m in loaded()}
    return [
        {
            "family": family,
            "task": info["task_type"],
            "description": info["description"],
            "variants": [_variant(family, variant, under_way, in_memory) for variant in info["variants"]],
        }
        for family, info in MODEL_REGISTRY.items()
    ]


def _variant(family: str, variant: str, under_way: dict, in_memory: set) -> dict:
    to_download = weights.download_size(family, variant, device=config.DEVICE)
    key = (family, variant)
    state = under_way.get(key) or ("loaded" if key in in_memory else "downloaded" if to_download == 0 else "not downloaded")
    return {
        "variant": variant,
        "license": weights.license_of(family, variant),
        "size": weights.model_size(family, variant, device=config.DEVICE),
        "to_download": to_download,
        "state": state,
    }


def exists(family: str, variant: str) -> bool:
    """Is this a model mozo publishes?"""
    try:
        _check(family, variant)
    except ValueError:
        return False
    return True


def _check(family: str, variant: str) -> None:
    """ValueError, naming what there is, unless mozo publishes this model."""
    get_model_info(family, variant)


def start_download(conn, family: str, variant: str) -> int:
    """Queue the download of a model's files for config.DEVICE; returns the job id."""
    _check(family, variant)
    return jobs.start(conn, DOWNLOAD, {"family": family, "variant": variant})


def ensure(conn, family: str, variant: str) -> int | None:
    """None if the model is downloaded; else the job downloading it: the one under way, or a new
    one. What needs a model waits for this (a job kind's `needs`), so it's downloaded once, first."""
    if downloaded(family, variant):
        return None
    return jobs.find_active(conn, DOWNLOAD, {"family": family, "variant": variant}) or start_download(conn, family, variant)


def downloads_first(conn, job_id: int) -> list[dict]:
    """The model downloads a job waits for: [{model ("family variant"), size (bytes to download)}]."""
    rows = conn.execute(
        """SELECT waited.params FROM job, json_each(job.waits_for) AS w JOIN job AS waited ON waited.id = w.value
           WHERE job.id = ? AND waited.kind = ?""", (job_id, DOWNLOAD))
    models = [json.loads(row["params"]) for row in rows]
    return [{"model": f"{m['family']} {m['variant']}", "size": weights.download_size(m["family"], m["variant"], device=config.DEVICE)}
            for m in models]


@jobs.kind(DOWNLOAD, lane="io", title="Download model", unit="bytes")
def download(ctx: jobs.Context, family: str, variant: str) -> dict:
    ctx.advance(0, f"{family} {variant}")
    try:
        weights.fetch(family, variant, device=config.DEVICE, progress=ctx.progress,
                      cancelled=lambda: ctx.cancelled)
    except weights.DownloadCancelled:
        pass  # asked to stop: the job ends cancelled (or interrupted), finished files stay
    return {"model": f"{family} {variant}"}


def remove(family: str, variant: str) -> int:
    """Delete a model's downloaded files; returns the bytes freed. A loaded model keeps working
    (its weights are in memory); its next load downloads again. Raises weights.WeightsError
    while the model is downloading."""
    return weights.remove(family, variant)


def load(family: str, variant: str, **kwargs):
    """The loaded model, loading it on first use, on config.DEVICE. Everything that runs a vision
    model gets it here (a workflow run passes this to mozo as `models=`), so each loads once."""
    return models.get_model(family, variant, device=config.DEVICE, **kwargs)


def unload(family: str, variant: str) -> bool:
    """Release a loaded model and its memory; returns whether it was loaded."""
    return models.unload(family, variant, device=config.DEVICE)


def loaded() -> list[dict]:
    """The loaded models: family, variant, the device and runtime they run on, when last used,
    and an estimate of their memory in bytes (from the weights' size; None if unknown)."""
    return [{k: r[k] for k in ("family", "variant", "device", "runtime", "last_used", "size")}
            for r in models.resident()]


def unload_all() -> int:
    """Release every loaded model; returns how many. Also called at shutdown, after the jobs stop."""
    return len(models.clear())


def is_loaded(family: str, variant: str) -> bool:
    return any((r["family"], r["variant"], r["device"]) == (family, variant, config.DEVICE) for r in models.resident())


def downloaded(family: str, variant: str) -> bool:
    return weights.is_cached(family, variant, device=config.DEVICE)


def busy(conn) -> dict[tuple[str, str], str]:
    """(family, variant) -> "downloading" or "loading", for models with a job queued or running."""
    return {(p["family"], p["variant"]): state
            for kind, state in ((DOWNLOAD, "downloading"), (LOAD, "loading"))
            for p in jobs.active_params(conn, kind)}


def start_load(conn, family: str, variant: str) -> int:
    """Queue loading a model, after downloading it if it isn't; returns the job id."""
    _check(family, variant)
    return jobs.start(conn, LOAD, {"family": family, "variant": variant})


def loading(conn, family: str, variant: str) -> dict:
    """The job loading a model: the one under way, or a new one (as ensure() for downloads): {id,
    status (queued or running), started (it's new), ahead: the title of the job it waits behind
    in the gpu lane, or None}."""
    job_id = jobs.find_active(conn, LOAD, {"family": family, "variant": variant})
    started = job_id is None
    job = jobs.get(conn, job_id or start_load(conn, family, variant))
    blocking = jobs.ahead(conn, job)
    return {"id": job["id"], "status": job["status"], "started": started, "ahead": blocking and blocking["title"]}


def _download_first(conn, params: dict) -> list[int]:
    job = ensure(conn, params["family"], params["variant"])
    return [job] if job else []


@jobs.kind(LOAD, lane="gpu", title="Load model", needs=_download_first)
def load_job(ctx: jobs.Context, family: str, variant: str) -> dict:
    ctx.advance(0, f"{family} {variant}")  # no total: torch loading reports none
    load(family, variant)
    return {"model": f"{family} {variant}"}
