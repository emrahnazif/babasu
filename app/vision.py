"""Vision models: mozo's published models, downloaded, loaded and unloaded for any use.
Workflow runs, tools and annotation all load through the one manager here.

Downloading is a job (io lane) with progress in bytes. Loading is a job too (gpu lane): it
imports torch and takes seconds, and only a downloaded model loads.
"""

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


def downloaded(family: str, variant: str) -> bool:
    return weights.is_cached(family, variant, device=config.DEVICE)


def busy(conn) -> dict[tuple[str, str], str]:
    """(family, variant) -> "downloading" or "loading", for models with a job queued or running."""
    return {(p["family"], p["variant"]): state
            for kind, state in ((DOWNLOAD, "downloading"), (LOAD, "loading"))
            for p in jobs.active_params(conn, kind)}


def start_load(conn, family: str, variant: str) -> int:
    """Queue loading a downloaded model; returns the job id. ValueError if it isn't downloaded."""
    _check(family, variant)
    if not downloaded(family, variant):
        raise ValueError(f"{family} {variant} isn't downloaded; download it first")
    return jobs.start(conn, LOAD, {"family": family, "variant": variant})


@jobs.kind(LOAD, lane="gpu", title="Load model")
def load_job(ctx: jobs.Context, family: str, variant: str) -> dict:
    ctx.advance(0, f"{family} {variant}")  # no total: it's downloaded, and torch loading reports none
    load(family, variant)
    return {"model": f"{family} {variant}"}
