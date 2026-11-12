"""Workflows and their runs: a mozo workflow over stored images, as a job, with its results kept
in babasu.

Workflows are mozo's JSON files: the built-ins ship with the app (app/workflows), the user's are in
the workspace (config.WORKFLOWS_DIR). A workflow's id is its file name; a built-in's can't be
taken by a saved one. A workflow says what to keep with save_to_babasu nodes: one run of a
workflow with two of them (say SAM 3, and Grounding DINO → EdgeTAM) keeps two labelled results
per image.
"""

import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

from mozo.workflow import Detections, Workflow, WorkflowError, annotation_line, node

from app import config, db, jobs, media, vision

BUILTIN_DIR = config.BASE_DIR / "workflows"
KIND = "workflow_run"  # job kind

# The built-in workflows' ids (their names are in their files).
BUILTIN = ("sam3", "grounding-dino", "grounding-dino-edgetam")


@node(category="Output")
def save_to_babasu(detections: Detections, label: str = "") -> Detections:
    """Keep these detections in babasu, per image, under `label` (empty: the node's id).

    The node only marks what to keep and passes it on: the run job stores what arrives here,
    since it knows the run, the image and the database. mozo's save_annotations writes the same
    results to a file instead.
    """
    return detections


def _files() -> dict[str, Path]:
    """Workflow id -> its file: the built-ins, then the saved ones whose id isn't a built-in's."""
    files = {id: BUILTIN_DIR / f"{id}.json" for id in BUILTIN}
    saved = sorted(config.WORKFLOWS_DIR.glob("*.json")) if config.WORKFLOWS_DIR.is_dir() else []
    return files | {path.stem: path for path in saved if path.stem not in files}


def _entry(id: str, workflow: Workflow | None = None) -> dict:
    """A workflow's id, name (its file's, or its id), description and whether it's built in."""
    name, description = (workflow.name, workflow.description) if workflow else (None, None)
    return {"id": id, "name": name or id, "description": description, "builtin": id in BUILTIN}


def workflows() -> list[dict]:
    """Every workflow: id, name, description, whether it's built in, and the models it uses (as
    detail() lists them), or `error`: why mozo doesn't load it."""
    out = []
    for id, path in _files().items():
        try:
            workflow = Workflow.load(path)
        except WorkflowError as e:
            out.append(_entry(id) | {"error": str(e)})
            continue
        out.append(_entry(id, workflow) | {"models": [use._asdict() for use in workflow.check(device=config.DEVICE).models]})
    return out


def load(id: str) -> Workflow | None:
    """The workflow read from its file, or None if there's no such workflow. A file that doesn't
    load raises mozo's WorkflowError."""
    path = _files().get(id)
    return Workflow.load(path) if path else None


def detail(id: str, workflow: Workflow) -> dict:
    """A loaded workflow with its graph (mozo's {nodes, edges}), the models it uses (each with
    its licence and the bytes still to download for config.DEVICE, 0 once downloaded) and the
    files it names that are missing."""
    found = workflow.check(device=config.DEVICE)
    return _entry(id, workflow) | {
        "graph": workflow.to_dict(),
        "models": [use._asdict() for use in found.models],
        "missing_files": [{"node": node, "parameter": name, "path": path} for node, name, path in found.missing],
    }


# --- Runs ---


def run_name(label: str, images: int) -> str:
    """A run's name: "<label> · N images"."""
    return f"{label} · {images:,} image{'' if images == 1 else 's'}"


def to_download(workflow_id: str) -> list[str]:
    """The models the workflow uses that aren't downloaded yet (for config.DEVICE), as
    "family variant"; empty when it's ready to run."""
    return [f"{use.family} {use.variant}" for use in load(workflow_id).check(device=config.DEVICE).models if use.size]


def create(conn: sqlite3.Connection, workflow_id: str, media_ids: list[int], project_id: int | None = None,
           name: str | None = None, classes: list[int] | None = None, annotates: bool = False) -> int:
    """Create a run of a workflow over stored images (no repeats), without its job (launch() starts
    one), and commit it; returns its id. `classes`: the project classes its prompts stand for, in
    order; `annotates`: its results become the project's annotations. ValueError: no such
    workflow, or unknown images."""
    workflow = load(workflow_id)
    if workflow is None:
        raise ValueError(f"There is no workflow {workflow_id!r}")
    found = conn.execute("SELECT count(*) FROM media WHERE id IN (SELECT value FROM json_each(?))",
                         (json.dumps(media_ids),)).fetchone()[0]
    if not media_ids or found != len(media_ids):
        raise ValueError("Give stored images (media ids) to run on")
    with conn:
        return conn.execute(
            """INSERT INTO run (workflow_id, workflow, project_id, name, classes, annotates)
               VALUES (?, ?, ?, ?, ?, ?) RETURNING id""",
            (workflow_id, json.dumps(workflow.to_dict()), project_id, name or run_name(workflow.name or workflow_id, len(media_ids)),
             classes and json.dumps(classes), annotates),
        ).fetchone()[0]


def launch(conn: sqlite3.Connection, run_id: int, kind: str, params: dict, project_id: int | None = None) -> int:
    """Start the job that does a run's work (a job kind built on execute(), given the run id) and
    return its id."""
    job_id = jobs.start(conn, kind, {"run_id": run_id} | params, project_id)
    with conn:
        conn.execute("UPDATE run SET job_id = ? WHERE id = ?", (job_id, run_id))
    return job_id


def start(conn: sqlite3.Connection, workflow_id: str, media_ids: list[int], overrides: dict | None = None,
          project_id: int | None = None, name: str | None = None, prompts: list[str] | None = None) -> tuple[int, int]:
    """Create a run of a workflow over stored images and start its job; returns (run id, job id).
    `prompts`: what to look for, given to every prompted node (see execute()); `overrides`:
    parameter values as mozo's run takes them, by name or one node's ("dino.threshold").
    ValueError as create(). The workflow's own problems (a model not downloaded, a missing file,
    prompts it can't take) fail the job, which says why."""
    media_ids = list(dict.fromkeys(media_ids))
    run_id = create(conn, workflow_id, media_ids, project_id, name)
    params = {"media_ids": media_ids, "overrides": overrides, "prompts": prompts}
    return run_id, launch(conn, run_id, KIND, params, project_id)


def _kept(workflow: Workflow) -> dict[str, str]:
    """The save_to_babasu nodes: node id -> the label its results are kept under."""
    return {step.id: step.parameters.get("label") or step.id
            for step in workflow.steps.values() if step.spec.name == "save_to_babasu"}


# Called as each image finishes, in the transaction that stores its results: (connection, the
# image's media row: id, sha256, ext, width, height, its results: label -> detections). What it
# writes commits with the results, or not at all; raising fails the job before the image counts.
OnImage = Callable[[sqlite3.Connection, sqlite3.Row, dict[str, Detections]], None]


@jobs.kind(KIND, lane="gpu", title="Run workflow")
def run_job(ctx: jobs.Context, run_id: int, media_ids: list[int], overrides: dict | None = None,
            prompts: list[str] | None = None) -> dict:
    return execute(ctx, run_id, media_ids, overrides, prompts)


def execute(ctx: jobs.Context, run_id: int, media_ids: list[int], overrides: dict | None = None,
            prompts: list[str] | None = None, on_image: OnImage | None = None) -> dict:
    """A run's work, inside a job: the workflow over the images, keeping each image's results.
    `prompts` go to every prompted node (mozo's prompt_parameters: a detection's class_id is its
    prompt's index), so one list reaches SAM 3 and Grounding DINO side by side. A job kind built
    on runs (auto-annotate) does more per image with `on_image`.

    Crash-safe and safe to retry: each image's results are committed together as it finishes,
    and a retry skips the images the run already has."""
    with closing(db.connect()) as conn:
        row = conn.execute("SELECT workflow, name FROM run WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("The run was deleted")
        with conn:
            conn.execute("UPDATE run SET job_id = ? WHERE id = ?", (ctx.job_id, run_id))
        ctx.advance(0, row["name"])
        workflow = Workflow.from_dict(json.loads(row["workflow"]))
        kept = _kept(workflow)
        if not kept:
            raise ValueError("The workflow has no save_to_babasu node, so a run would keep nothing")
        if prompts is not None:
            if not workflow.prompt_parameters:
                raise ValueError("The workflow takes no prompts")
            overrides = {key: prompts for key in workflow.prompt_parameters} | (overrides or {})
        overrides = overrides or {}
        found = workflow.check(device=config.DEVICE, **overrides)
        if found.missing:
            raise ValueError("Missing files: " + ", ".join(path for _, _, path in found.missing))
        if found.downloads:  # a run never starts a download: that's the Models page's job, with progress
            left = ", ".join(f"{family} {variant}" for _, family, variant, _ in found.downloads)
            raise ValueError(f"Download these models first (Models page): {left}")

        done = {r[0] for r in conn.execute("SELECT DISTINCT media_id FROM run_result WHERE run_id = ?", (run_id,))}
        images = {
            str(media.path(r["sha256"], r["ext"])): r
            for r in conn.execute(
                "SELECT id, sha256, ext, width, height FROM media WHERE id IN (SELECT value FROM json_each(?))",
                (json.dumps([m for m in media_ids if m not in done]),),
            )
        }
        ctx.total(len(media_ids))
        ctx.advance(len(media_ids) - len(images))
        failed = 0

        def skip(item, event) -> None:
            nonlocal failed
            failed += 1
            ctx.log(f"Failed on {item} at {event.node}: {event.error}")
            ctx.advance()

        # Closing the results stops the pipeline: no image starts a node after that.
        with closing(workflow.run_many(list(images), models=vision.load, on_error=skip, **overrides)) as results:
            for path, outputs in results:
                image = images[path]
                found = {label: outputs[node] for node, label in kept.items()}
                with conn:
                    conn.executemany(
                        "INSERT OR REPLACE INTO run_result (run_id, media_id, label, result) VALUES (?, ?, ?, ?)",
                        [(run_id, image["id"], label,
                          annotation_line(detections, label=image["sha256"], width=image["width"], height=image["height"]))
                         for label, detections in found.items()],
                    )
                    if on_image:
                        on_image(conn, image, found)
                ctx.advance()
                if ctx.cancelled:
                    break
    return {"images": ctx.done - failed, "failed": failed}


_RUNS = """SELECT run.id, run.name, run.workflow_id, run.project_id, run.job_id, run.created_at,
                  run.classes, run.annotates, job.status AS status,
                  (SELECT count(DISTINCT media_id) FROM run_result WHERE run_id = run.id) AS images
           FROM run LEFT JOIN job ON job.id = run.job_id"""


def _dict(row: sqlite3.Row) -> dict:
    return dict(row) | {"classes": row["classes"] and json.loads(row["classes"]), "annotates": bool(row["annotates"])}


def list_all(conn: sqlite3.Connection) -> list[dict]:
    """Every run, newest first: its name, workflow, project, when, the project classes its prompts
    stood for (None: other prompts or none) and whether it annotated the project, its job's
    status, and how many images have results."""
    return [_dict(r) for r in conn.execute(f"{_RUNS} ORDER BY run.id DESC")]


def _run(conn: sqlite3.Connection, run_id: int) -> dict | None:
    row = conn.execute(f"{_RUNS} WHERE run.id = ?", (run_id,)).fetchone()
    return row and _dict(row)


def _results(conn: sqlite3.Connection, run_id: int) -> list[dict]:
    """Per image and label: media id, sha256, the image's file name in the run's project (without
    one, the start of its sha256), and the result, mozo's annotation line (detections with boxes, classes, scores
    and masks; the image's width and height)."""
    rows = conn.execute(
        """SELECT r.media_id, m.sha256, coalesce(pm.filename, substr(m.sha256, 1, 12)) AS filename, r.label, r.result
           FROM run_result r JOIN run ON run.id = r.run_id JOIN media m ON m.id = r.media_id
           LEFT JOIN project_media pm ON pm.project_id = run.project_id AND pm.media_id = r.media_id
           WHERE r.run_id = ? ORDER BY r.media_id, r.label""",
        (run_id,),
    ).fetchall()
    return [dict(r) | {"result": json.loads(r["result"])} for r in rows]


def get(conn: sqlite3.Connection, run_id: int) -> dict | None:
    """The run with all its results (see _results)."""
    run = _run(conn, run_id)
    return None if run is None else run | {"results": _results(conn, run_id)}


def summary(conn: sqlite3.Connection, run_id: int, images: int = 20) -> dict | None:
    """The run in brief, per result label: objects found per class with their average score, and
    for the first `images` images (`image_count`: all of them) each object's class, score and box
    in pixels. No masks."""
    run = _run(conn, run_id)
    if run is None:
        return None
    labels: dict[str, dict] = {}
    for r in _results(conn, run_id):
        found = [{"class": d.get("class_name"), "score": d.get("confidence"), "box": d.get("bbox")}
                 for d in r["result"]["detections"]]
        label = labels.setdefault(r["label"], {"scores": {}, "image_count": 0, "images": []})
        for f in found:
            label["scores"].setdefault(f["class"], []).append(f["score"] or 0)
        label["image_count"] += 1
        if len(label["images"]) < images:
            label["images"].append({"image": r["filename"], "objects": found})
    for label in labels.values():
        label["classes"] = {name: {"objects": len(s), "average_score": round(sum(s) / len(s), 3)}
                            for name, s in label.pop("scores").items()}
    return run | {"results": labels}


def rename(conn: sqlite3.Connection, run_id: int, name: str) -> bool:
    with conn:
        return conn.execute("UPDATE run SET name = ? WHERE id = ?", (name, run_id)).rowcount > 0


def delete(conn: sqlite3.Connection, run_id: int) -> list[Path] | None:
    """Delete a run and its results; returns the image files no project or run uses any more, for
    the caller to delete after this commits. None if there's no such run. ValueError while its job
    is queued or running."""
    row = _run(conn, run_id)
    if row is None:
        return None
    if row["status"] in jobs.ACTIVE:
        raise ValueError("The run's job is still going; cancel it first")
    with conn:
        used = [r[0] for r in conn.execute("SELECT DISTINCT media_id FROM run_result WHERE run_id = ?", (run_id,))]
        conn.execute("DELETE FROM run WHERE id = ?", (run_id,))
        return media.remove_unused(conn, used)
