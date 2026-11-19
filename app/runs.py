"""Workflows and their runs: a mozo workflow over stored images, as a job, with its results kept
in babasu.

Workflows are mozo's JSON files: the built-ins ship with the app (app/workflows), the user's are in
the workspace (config.WORKFLOWS_DIR). A workflow's id is its file name; a built-in's can't be
taken by a saved one. A workflow says what to keep with save_to_babasu nodes: one run of a
workflow with two of them (say SAM 3, and Grounding DINO → EdgeTAM) keeps two labelled results
per image.
"""

import json
import math
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

from mozo.workflow import Detections, Workflow, WorkflowError, annotation_line, catalogue, node

from app import config, db, history, jobs, media, projects, settings, vision

BUILTIN_DIR = config.BASE_DIR / "workflows"
KIND = "workflow_run"  # job kind

# The built-in workflows' ids (their names are in their files).
BUILTIN = ("sam3", "grounding-dino", "grounding-dino-edgetam")
FINDER = "sam3"  # what a run runs when no workflow is named: it finds and outlines what its prompts name


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
    return {"id": id, "name": name or id, "description": description or "", "builtin": id in BUILTIN}


def named(text: str) -> list[dict]:
    """The workflows whose id or name contains `text` (ignoring case): id and name, read from their
    files without loading them."""
    found = []
    for id, path in _files().items():
        name = json.loads(path.read_text()).get("name") or id
        if text.casefold() in id.casefold() or text.casefold() in name.casefold():
            found.append({"id": id, "name": name})
    return found


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
        out.append(_listed(id, workflow))
    return out


def _listed(id: str, workflow: Workflow) -> dict:
    """A workflow as the list shows it: what it does and is (built in, the default finder), whether
    it takes prompts, what a run keeps, what it still downloads, its models."""
    models = workflow.check(device=config.DEVICE).models
    return _entry(id, workflow) | {"default": id == FINDER, "prompts": bool(workflow.prompt_parameters),
                                   "keeps": list(_kept(workflow).values()), "to_download": sum(m.size for m in models),
                                   "models": [use._asdict() for use in models]}


def load(id: str) -> Workflow | None:
    """The workflow read from its file, or None if there's no such workflow. A file that doesn't
    load raises mozo's WorkflowError."""
    path = _files().get(id)
    return Workflow.load(path) if path else None


def document(id: str) -> dict | None:
    """A workflow's document as its file has it (loaded or not: the builder opens a broken one to
    fix it); None if there's no such workflow."""
    path = _files().get(id)
    return json.loads(path.read_text()) if path else None


def detail(id: str, workflow: Workflow) -> dict:
    """A loaded workflow with its graph (mozo's {nodes, edges}), what a run keeps (its
    save_to_babasu labels), which parameters take the prompts, the models it uses (each with its
    licence and the bytes still to download for config.DEVICE, 0 once downloaded) and the files
    it names that are missing."""
    found = workflow.check(device=config.DEVICE)
    return _entry(id, workflow) | {
        "graph": workflow.to_dict(),
        "kept": list(_kept(workflow).values()),
        "prompt_parameters": list(workflow.prompt_parameters),
        "models": [use._asdict() for use in found.models],
        "missing_files": [{"node": node, "parameter": name, "path": path} for node, name, path in found.missing],
    }


def _loaded(document: dict) -> Workflow:
    try:
        return Workflow.from_dict(document)
    except WorkflowError as e:
        raise ValueError(f"The workflow doesn't load: {e}") from None


def resolve(workflow_id: str | None, document: dict | None = None) -> tuple[str | None, Workflow]:
    """What a run runs: a saved workflow (its id; none named, FINDER), or one not saved yet (its
    document: a draft tried on a few images; it has no id). ValueError: no such workflow, or a
    document mozo doesn't load."""
    if document is not None:
        return None, _loaded(document)
    workflow_id = workflow_id or FINDER
    workflow = load(workflow_id)
    if workflow is None:
        raise ValueError(f"There is no workflow {workflow_id!r}")
    return workflow_id, workflow


def nodes(names: list[str] | None = None) -> list[dict]:
    """What workflows are built from (mozo's catalogue): each node's name, category and what it
    does; with `names`, those nodes in full: inputs and outputs (name: type) and parameters (name,
    kind, default, options; `prompts`: it takes the prompts). ValueError naming unknown ones."""
    found = catalogue()
    if names is None:
        return [{"name": n["name"], "category": n["category"], "description": n["description"]} for n in found]
    unknown = set(names) - {n["name"] for n in found}
    if unknown:
        raise ValueError(f"No such nodes: {', '.join(sorted(unknown))}; without names, all of them are listed")
    return [{"name": n["name"], "category": n["category"], "description": n["description"],
             "inputs": [f"{p['name']}: {p['type']}" for p in n["inputs"]],
             "outputs": [f"{p['name']}: {p['type']}" for p in n["outputs"]],
             "parameters": [{k: v for k, v in p.items() if v not in (None, [], "", False) or k == "default"}
                            for p in n["parameters"]]}
            for n in found if n["name"] in names]


def check_workflow(document: dict) -> dict:
    """What a workflow document is, without saving it: what a run keeps (its save_to_babasu
    labels), which parameters take the prompts, its models (size: bytes left to download for
    config.DEVICE) and missing files. ValueError if mozo doesn't load it (why)."""
    workflow = _loaded(document)
    found = workflow.check(device=config.DEVICE)
    return {"kept": list(_kept(workflow).values()), "prompt_parameters": list(workflow.prompt_parameters),
            "models": [{"family": m.family, "variant": m.variant, "size": m.size} for m in found.models],
            "missing_files": [path for _, _, path in found.missing]}


def save_workflow(workflow_id: str, document: dict) -> dict:
    """Save a workflow in the workspace as `<id>.json` (new, or replacing a saved one) and return
    it as workflows() lists it. ValueError: a bad id, a built-in's id, a document mozo refuses, or
    one that keeps nothing (no save_to_babasu)."""
    if not settings.SLUG.fullmatch(workflow_id):
        raise ValueError("A workflow's id is lowercase letters, digits and dashes, e.g. sam3-vs-dino")
    if workflow_id in BUILTIN:
        raise ValueError(f"{workflow_id!r} is a built-in workflow; save under another id")
    workflow = _loaded(document)
    if not _kept(workflow):
        raise ValueError("Nothing reaches a save_to_babasu node, so a run would keep nothing")
    settings.write_file(config.WORKFLOWS_DIR / f"{workflow_id}.json", json.dumps(workflow.to_dict(), indent=2, ensure_ascii=False))
    return _listed(workflow_id, workflow)


def create_workflow(document: dict) -> dict:
    """Save a new workflow (as save_workflow does), its id made from its name, unique: a workflow
    made in the builder, or a copy of one. ValueError as save_workflow, or if it has no name."""
    name = (document.get("name") or "").strip()
    if not name:
        raise ValueError("A new workflow needs a name")
    return save_workflow(settings.slug(name, set(_files())), document)


def delete_workflow(workflow_id: str) -> bool:
    """Delete a saved workflow (a built-in can't be); False if there's no such saved one. Its runs
    keep their own copy of it."""
    path = _files().get(workflow_id)
    if workflow_id in BUILTIN or path is None:
        return False
    with closing(db.connect()) as conn, conn:
        history.trash(conn, path)
    return True


# --- Runs ---


def run_name(label: str, images: int) -> str:
    """A run's name: "<label> · N images"."""
    return f"{label} · {images:,} image{'' if images == 1 else 's'}"


def _missing(workflow: Workflow) -> list:
    """The models a workflow uses that aren't downloaded (for config.DEVICE): mozo's model uses,
    their size the bytes left to download."""
    return [use for use in workflow.check(device=config.DEVICE).models if use.size]


def to_download(workflow_id: str) -> list[dict]:
    """The models the workflow uses that aren't downloaded yet (for config.DEVICE): [{model
    ("family variant"), size (bytes)}]; a run downloads them first. Empty when it's ready to run."""
    return [{"model": f"{use.family} {use.variant}", "size": use.size} for use in _missing(load(workflow_id))]


def needs(conn: sqlite3.Connection, params: dict) -> list[int]:
    """What a run's job waits for (a job kind's `needs`): the downloads of the models its workflow
    uses that aren't downloaded, as the run keeps the workflow."""
    row = conn.execute("SELECT workflow FROM run WHERE id = ?", (params["run_id"],)).fetchone()
    if row is None:
        return []
    missing = _missing(_loaded(json.loads(row["workflow"])))
    return [job for use in missing if (job := vision.ensure(conn, use.family, use.variant))]


def create(conn: sqlite3.Connection, workflow_id: str | None, workflow: Workflow, media_ids: list[int],
           name: str | None = None, classes: list[str] | None = None, annotates: bool = False) -> int:
    """Create a run of a workflow (resolve: its id, None for one not saved) over stored images (no
    repeats) of one project, without its job (launch() starts one), and commit it; returns its id.
    `classes`: the class names its prompts stand for, in order; `annotates`: its results become the
    project's annotations. ValueError: unknown images, or images of several projects."""
    rows = conn.execute("SELECT count(*), count(DISTINCT project_id), min(project_id) FROM media "
                        "WHERE id IN (SELECT value FROM json_each(?))", (json.dumps(media_ids),)).fetchone()
    if not media_ids or rows[0] != len(media_ids):
        raise ValueError("Give stored images (media ids) to run on")
    if rows[1] > 1:
        raise ValueError("Give images of one project to run on")
    project_id = rows[2]
    with conn:
        return conn.execute(
            """INSERT INTO run (workflow_id, workflow, project_id, name, classes, annotates)
               VALUES (?, ?, ?, ?, ?, ?) RETURNING id""",
            (workflow_id, json.dumps(workflow.to_dict()), project_id,
             name or run_name(workflow.name or workflow_id or "Draft", len(media_ids)),
             classes and json.dumps(classes), annotates),
        ).fetchone()[0]


def launch(conn: sqlite3.Connection, run_id: int, kind: str, params: dict) -> int:
    """Start the job that does a run's work (a job kind built on execute(), given the run id) and
    return its id."""
    project_id = conn.execute("SELECT project_id FROM run WHERE id = ?", (run_id,)).fetchone()[0]
    job_id = jobs.start(conn, kind, {"run_id": run_id} | params, project_id)
    with conn:
        conn.execute("UPDATE run SET job_id = ? WHERE id = ?", (job_id, run_id))
    return job_id


def start(conn: sqlite3.Connection, workflow_id: str | None, media_ids: list[int], overrides: dict | None = None,
          name: str | None = None, prompts: list[str] | None = None,
          document: dict | None = None) -> tuple[int, int, str]:
    """Create a run of a workflow (a saved one, or a `document` not saved yet: trying a draft)
    over stored images and start its job; returns (run id, job id, its name).
    `prompts`: what to look for, given to every prompted node (see execute()); none given, a run
    of a workflow that takes prompts finds its project's class prompts. `overrides`:
    parameter values as mozo's run takes them, by name or one node's ("dino.threshold").
    ValueError as create(). The models it uses that aren't downloaded are downloaded first (the job
    waits). The workflow's own problems (a missing file, prompts it can't take) fail the job,
    which says why."""
    media_ids = list(dict.fromkeys(media_ids))
    workflow_id, workflow = resolve(workflow_id, document)
    name = name or run_name(workflow.name or workflow_id or "Draft", len(media_ids))
    run_id = create(conn, workflow_id, workflow, media_ids, name)
    if prompts is None and workflow.prompt_parameters:
        project_id = conn.execute("SELECT project_id FROM run WHERE id = ?", (run_id,)).fetchone()[0]
        prompts = [c["prompt"] for c in projects.classes(conn, project_id)]
    params = {"media_ids": media_ids, "overrides": overrides, "prompts": prompts}
    return run_id, launch(conn, run_id, KIND, params), name


def matching(conn: sqlite3.Connection, project_id: int | None, filter) -> list[int]:
    """A project's images matching `filter` (every project's, without one). ValueError if there's no
    such project or none match."""
    if project_id is not None and projects.get(conn, project_id) is None:
        raise ValueError(f"There is no project {project_id}")
    media_ids = projects.sample(conn, filter, project_id)
    if not media_ids:
        raise ValueError("No images match that filter")
    return media_ids


def found_on(conn: sqlite3.Connection, run_id: int, found: str | None = None) -> list[int]:
    """A run's images: those it found `found` on (a class name, any case), or all it has results
    for. ValueError if there's no such run or none."""
    if conn.execute("SELECT 1 FROM run WHERE id = ?", (run_id,)).fetchone() is None:
        raise ValueError(f"There is no run {run_id}")
    media_ids = [r[0] for r in conn.execute(
        """SELECT DISTINCT media_id FROM run_result r WHERE run_id = :run AND (:found IS NULL OR EXISTS
           (SELECT 1 FROM json_each(r.objects) WHERE lower(key) = lower(:found))) ORDER BY media_id""",
        {"run": run_id, "found": found})]
    if not media_ids:
        raise ValueError(f"Run {run_id} found no {found}" if found else f"Run {run_id} has no results yet")
    return media_ids


def _kept(workflow: Workflow) -> dict[str, str]:
    """The save_to_babasu nodes: node id -> the label its results are kept under."""
    return {step.id: step.parameters.get("label") or step.id
            for step in workflow.steps.values() if step.spec.name == "save_to_babasu"}


# Called as each image finishes, in the transaction that stores its results: (connection, the
# image's media row: id, sha256, ext, width, height, its results: label -> detections). What it
# writes commits with the results, or not at all; raising fails the job before the image counts.
OnImage = Callable[[sqlite3.Connection, sqlite3.Row, dict[str, Detections]], None]


@jobs.kind(KIND, lane="gpu", title="Run workflow", needs=needs)
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
                        "INSERT OR REPLACE INTO run_result (run_id, media_id, label, result, objects) VALUES (?, ?, ?, ?, ?)",
                        [(run_id, image["id"], label,
                          annotation_line(detections, label=image["sha256"], width=image["width"], height=image["height"]),
                          _objects(detections))
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


def list_all(conn: sqlite3.Connection, workflow_id: str | None = None, limit: int | None = None) -> list[dict]:
    """Every run (of one workflow if given; the newest `limit`), newest first: its name, workflow,
    project, when, the project classes its prompts stood for (None: other prompts or none) and
    whether it annotated the project, its job's status, and how many images have results."""
    rows = conn.execute(f"{_RUNS} WHERE ? IS NULL OR run.workflow_id = ? ORDER BY run.id DESC LIMIT ?",
                        (workflow_id, workflow_id, limit or -1))
    return [_dict(r) for r in rows]


def _run(conn: sqlite3.Connection, run_id: int) -> dict | None:
    row = conn.execute(f"{_RUNS} WHERE run.id = ?", (run_id,)).fetchone()
    return row and _dict(row)


def _objects(detections: Detections) -> str:
    """What a result found per class (its name; "class N" without one): [how many, the sum of their
    scores], as it's stored with the result."""
    found: dict[str, list] = {}
    for d in detections:
        counted = found.setdefault(d.class_name or f"class {d.class_id}", [0, 0.0])
        counted[0] += 1
        counted[1] += d.confidence or 0
    return json.dumps(found)


PAGE_SIZE = 30  # images a page of a run (each with its results, one a label)


def _results(conn: sqlite3.Connection, run_id: int, page: int | None = None, shapes: bool = False) -> list[dict]:
    """Per image and label, by image (all of them, or one page of PAGE_SIZE images): media id,
    sha256, the image's file name and
    the objects found per class; with `shapes`, the result too, mozo's annotation line (detections
    with boxes, classes, scores and masks; the image's width and height)."""
    images = "" if page is None else \
        "AND r.media_id IN (SELECT DISTINCT media_id FROM run_result WHERE run_id = :run ORDER BY media_id LIMIT :size OFFSET :skip)"
    rows = conn.execute(
        f"""SELECT r.media_id, m.sha256, m.name AS filename, r.label, r.objects
                   {", r.result" if shapes else ""}
            FROM run_result r JOIN media m ON m.id = r.media_id
            WHERE r.run_id = :run {images} ORDER BY r.media_id, r.label""",
        {"run": run_id, "size": PAGE_SIZE, "skip": ((page or 1) - 1) * PAGE_SIZE},
    ).fetchall()
    return [dict(r) | {"objects": {name: n for name, (n, _) in json.loads(r["objects"]).items()}}
            | ({"result": json.loads(r["result"])} if shapes else {}) for r in rows]


def finds(conn: sqlite3.Connection, run_id: int, media_id: int) -> dict | None:
    """What a run found on one image: the image's file name (as _results names it) and its
    detections, all labels' in order (class_name, confidence, bbox [x1, y1, x2, y2], …); None if
    the run has no result for the image."""
    rows = conn.execute(
        """SELECT m.name AS filename, r.result
           FROM run_result r JOIN media m ON m.id = r.media_id
           WHERE r.run_id = ? AND r.media_id = ? ORDER BY r.label""",
        (run_id, media_id),
    ).fetchall()
    if not rows:
        return None
    return {"filename": rows[0]["filename"], "detections": [d for r in rows for d in json.loads(r["result"])["detections"]]}


def get(conn: sqlite3.Connection, run_id: int) -> dict | None:
    """The run with all its results, shapes included (see _results)."""
    run = _run(conn, run_id)
    return None if run is None else run | {"results": _results(conn, run_id, shapes=True)}


def found(conn: sqlite3.Connection, run_id: int) -> dict[str, dict[str, dict]]:
    """What a run found, per label and class: how many objects, on how many images, and their
    average score."""
    rows = conn.execute(
        """SELECT r.label, o.key AS name, sum(json_extract(o.value, '$[0]')) AS objects, count(DISTINCT r.media_id) AS images,
                  round(sum(json_extract(o.value, '$[1]')) / sum(json_extract(o.value, '$[0]')), 3) AS average_score
           FROM run_result r, json_each(r.objects) o
           WHERE r.run_id = ? GROUP BY r.label, name ORDER BY r.label, objects DESC""",
        (run_id,),
    )
    by_label: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_label.setdefault(r["label"], {})[r["name"]] = {"objects": r["objects"], "images": r["images"],
                                                          "average_score": r["average_score"]}
    return by_label


def results_page(conn: sqlite3.Connection, run_id: int, page: int = 1, shapes: bool = False) -> dict | None:
    """The run, what it found (`found`), and one page of its images' results (see _results). None
    if there's no such run."""
    run = _run(conn, run_id)
    if run is None:
        return None
    pages = max(1, math.ceil(run["images"] / PAGE_SIZE))
    page = min(max(page, 1), pages)
    return run | {"found": found(conn, run_id), "page": page, "pages": pages,
                  "results": _results(conn, run_id, page, shapes)}


def rename(conn: sqlite3.Connection, run_id: int, name: str) -> bool:
    with conn:
        return conn.execute("UPDATE run SET name = ? WHERE id = ?", (name, run_id)).rowcount > 0


def delete(conn: sqlite3.Connection, run_id: int) -> dict | None:
    """Delete a run and its results; the run as it was (name, images), or None if there's no such run. ValueError while its job is
    queued or running."""
    row = _run(conn, run_id)
    if row is None:
        return None
    if row["status"] in jobs.ACTIVE:
        raise ValueError("The run's job is still going; cancel it first")
    with conn:
        conn.execute("DELETE FROM run WHERE id = ?", (run_id,))
    return row
