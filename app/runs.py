"""Workflows and running them: a mozo workflow over stored images, as a job, keeping in babasu
what reaches its save nodes.

Workflows are mozo's JSON files: the built-ins ship with the app (app/workflows), the user's are in
the workspace (config.WORKFLOWS_DIR). A workflow's id is its file name; a built-in's can't be
taken by a saved one. A workflow says what to keep with its save nodes, and each keeps what reaches
it: save_to_babasu as the input image's annotations, save_image_to_babasu as a new image. Two of
them keep twice; the workflow decides, not the app.
"""

import io
import json
import random
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

import pixelflow as pf
from mozo.workflow import Detections, Image, Workflow, WorkflowError, catalogue, node

from app import config, db, history, jobs, media, projects, settings, vision

BUILTIN_DIR = config.BASE_DIR / "workflows"
KIND = "workflow_run"  # job kind

# The built-in workflows' ids (their names are in their files).
BUILTIN = ("sam3", "grounding-dino", "grounding-dino-edgetam")
FINDER = "sam3"  # what a run runs when no workflow is named: it finds and outlines what its prompts name


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
                                   "keeps": _kept(workflow), "to_download": sum(m.size for m in models),
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
    """A loaded workflow with its graph (mozo's {nodes, edges}), what a run keeps (_kept), which parameters take the prompts, the models it uses (each with its
    licence and the bytes still to download for config.DEVICE, 0 once downloaded) and the files
    it names that are missing."""
    found = workflow.check(device=config.DEVICE)
    return _entry(id, workflow) | {
        "graph": workflow.to_dict(),
        "kept": _kept(workflow),
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
    """What a workflow document is, without saving it: what a run keeps (_kept), which parameters take the prompts, its models (size: bytes left to download for
    config.DEVICE) and missing files. ValueError if mozo doesn't load it (why)."""
    workflow = _loaded(document)
    found = workflow.check(device=config.DEVICE)
    return {"kept": _kept(workflow), "prompt_parameters": list(workflow.prompt_parameters),
            "models": [{"family": m.family, "variant": m.variant, "size": m.size} for m in found.models],
            "missing_files": [path for _, _, path in found.missing]}


def save_workflow(workflow_id: str, document: dict) -> dict:
    """Save a workflow in the workspace as `<id>.json` (new, or replacing a saved one) and return
    it as workflows() lists it. ValueError: a bad id, a built-in's id, a document mozo refuses, or
    one that keeps nothing (no save node)."""
    if not settings.SLUG.fullmatch(workflow_id):
        raise ValueError("A workflow's id is lowercase letters, digits and dashes, e.g. sam3-vs-dino")
    if workflow_id in BUILTIN:
        raise ValueError(f"{workflow_id!r} is a built-in workflow; save under another id")
    workflow = _loaded(document)
    if not _kept(workflow):
        raise ValueError("Nothing reaches a save node (save_to_babasu, save_image_to_babasu), so a run would keep nothing")
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
    """Delete a saved workflow (a built-in can't be); False if there's no such saved one."""
    path = _files().get(workflow_id)
    if workflow_id in BUILTIN or path is None:
        return False
    with closing(db.connect()) as conn, conn:
        history.trash(conn, path)
    return True


# --- Running ---


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
    uses that aren't downloaded."""
    missing = _missing(_loaded(params["workflow"]))
    return [job for use in missing if (job := vision.ensure(conn, use.family, use.variant))]


def pending(conn: sqlite3.Connection, media_ids: list[int], workflow: str | None, classes: list[str]) -> dict[int, list[str]]:
    """Image -> those of `classes` (names) the workflow hasn't looked for on it yet (media.analysed),
    in their order; images with none left are left out. A draft (no id) has looked for nothing."""
    path = json.dumps(workflow or "")  # '$."sam3"': the workflow's list in analysed
    rows = conn.execute(
        """SELECT m.id, c.value FROM media m, json_each(?) c WHERE m.id IN (SELECT value FROM json_each(?))
            AND NOT EXISTS (SELECT 1 FROM json_each(m.analysed, '$.' || ?) d WHERE lower(d.value) = lower(c.value))
            ORDER BY c.key""", (json.dumps(classes), json.dumps(media_ids), path))
    todo: dict[int, list[str]] = {}
    for media_id, name in rows:
        todo.setdefault(media_id, []).append(name)
    return {m: todo[m] for m in media_ids if m in todo}


def start(conn: sqlite3.Connection, media_ids: list[int], workflow_id: str | None = None, classes: list[str] | None = None,
          prompts: list[str] | None = None, again: bool = False, min_score: float = 0.0, overrides: dict | None = None,
          document: dict | None = None, count: int | None = None, tried: bool = False) -> dict:
    """Run a workflow (a saved one, the default finder unless named, or a `document` not saved yet:
    trying a draft) over stored images, as a job: what reaches its save nodes is kept (run_job).
    `classes`: the names it looks for, each by its prompt (`prompts`, in the same order; else its
    prompt in the images' project, or for images of several projects or a class it hasn't had, its
    name), given to every prompted node; a name an image's project hasn't had becomes a class there.
    Each image runs only for the classes this workflow hasn't looked for on it (`count` of those at
    random), unless `again`, which replaces what its models found before (annotations with a score)
    of those classes. `tried`: a Try (a project page shows its images: annotators.latest_try).
    Returns {job_id, images}. ValueError: no images, or nothing left to do. The models it uses that
    aren't downloaded are downloaded first (the job waits)."""
    workflow_id, workflow = resolve(workflow_id, document)
    media_ids = list(dict.fromkeys(media_ids))
    if not media_ids:
        raise ValueError("No images to run on")
    if classes and not workflow.prompt_parameters:
        raise ValueError(f"{workflow.name or workflow_id or 'The workflow'} takes no prompts, so it can't look for classes")
    if classes and not again and workflow_id:
        media_ids = list(pending(conn, media_ids, workflow_id, classes))
        if not media_ids:
            raise ValueError("Nothing to do: these images are done for these classes (again does them again)")
    if count is not None and count < len(media_ids):
        media_ids = random.sample(media_ids, count)
    found = [r[0] for r in conn.execute(
        "SELECT DISTINCT project_id FROM media WHERE id IN (SELECT value FROM json_each(?))", (json.dumps(media_ids),))]
    if classes and prompts is None and len(found) == 1:
        listed = {c["name"].casefold(): c["prompt"] for c in projects.including(projects.classes(conn, found[0]), classes)}
        prompts = [listed[c.casefold()] for c in classes]
    params = {"media_ids": media_ids, "workflow": workflow.to_dict(), "workflow_id": workflow_id, "classes": classes or [],
              "prompts": prompts or classes or None, "again": again, "min_score": min_score, "overrides": overrides,
              "tried": tried}
    job_id = jobs.start(conn, KIND, params, found[0] if len(found) == 1 else None)
    return {"job_id": job_id, "images": len(media_ids)}


@node(category="Output")
def save_to_babasu(detections: Detections) -> Detections:
    """Keep these detections in babasu as the image's annotations, in its project.

    The node only marks what to keep and passes it on: the run job stores what arrives here,
    since it knows the image and the database. mozo's save_annotations writes them to a file instead.
    """
    return detections


@node(category="Output")
def save_image_to_babasu(image: Image, project: int | None = None, label: str = "") -> Image:
    """Keep this image in babasu as a new image (a drawn, cropped or edited copy): in `project` (its
    id), or the input image's project when empty, named after the input and `label`.

    As save_to_babasu, the node only marks it; the run job stores it. mozo's save_image writes files.
    """
    return image


def _saves(workflow: Workflow) -> list:
    """The workflow's save nodes, in order."""
    return [step for step in workflow.steps.values() if step.spec.name in ("save_to_babasu", "save_image_to_babasu")]


def _kept(workflow: Workflow) -> list[str]:
    """What a run of the workflow keeps, a save node at a time: "annotations", or "image (label)"."""
    return ["annotations" if step.spec.name == "save_to_babasu" else f"image ({step.parameters.get('label') or step.id})"
            for step in _saves(workflow)]


def annotations(detections: Detections, classes: list[str], min_score: float = 0.0) -> list[projects.Annotation]:
    """Detections as annotations, box and polygon (from the mask, as pixelflow outlines it) with
    the confidence as score. A detection's class: the class its prompt stood for (`classes`, by its
    class_id, a prompt's index), else the name the model gave it."""
    found = []
    for d in detections.masks_to_segments():
        if d.confidence is not None and d.confidence < min_score:
            continue
        index = None if d.class_id is None else int(d.class_id)
        name = classes[index] if index is not None and 0 <= index < len(classes) else d.class_name or f"class {index}"
        bbox = None
        if d.bbox:
            x1, y1, x2, y2 = d.bbox
            bbox = (x1, y1, x2 - x1, y2 - y1)
        if bbox is None and not d.segments:
            continue
        found.append(projects.Annotation(class_=name, bbox=bbox, segments=d.segments, score=d.confidence))
    return found


def _png(image) -> bytes:
    """An image a workflow made (an array, as pixelflow has them), as a PNG to store."""
    buffer = io.BytesIO()
    pf.to_pil(image).save(buffer, format="PNG")
    return buffer.getvalue()


@jobs.kind(KIND, lane="gpu", title="Run workflow", needs=needs)
def run_job(ctx: jobs.Context, media_ids: list[int], workflow: dict, workflow_id: str | None, classes: list[str],
            prompts: list[str] | None = None, again: bool = False, min_score: float = 0.0,
            overrides: dict | None = None, tried: bool = False) -> dict:
    """A workflow over images, keeping what reaches its save nodes as each image finishes (one
    transaction per image): save_to_babasu's detections become the image's annotations (with
    `again`, first replacing those a model found before of `classes`); save_image_to_babasu's image
    becomes a new image. The image is marked analysed for `classes` by this workflow (not a draft,
    which has no id). `tried` only marks a Try. Crash-safe and safe to retry: a retry skips the images already analysed for them."""
    ctx.advance(0, "Starting…")
    flow = _loaded(workflow)
    saves = _saves(flow)
    saves_annotations = any(step.spec.name == "save_to_babasu" for step in saves)
    if not saves:
        raise ValueError("The workflow has no save node (save_to_babasu, save_image_to_babasu), so a run would keep nothing")
    overrides = overrides or {}
    if prompts is not None:
        overrides = {key: prompts for key in flow.prompt_parameters} | overrides
    found = flow.check(device=config.DEVICE, **overrides)
    if found.missing:
        raise ValueError("Missing files: " + ", ".join(path for _, _, path in found.missing))
    if found.downloads:  # a run never starts a download: that's the Models page's job, with progress
        left = ", ".join(f"{family} {variant}" for _, family, variant, _ in found.downloads)
        raise ValueError(f"Download these models first (Models page): {left}")

    with closing(db.connect()) as conn:
        todo = set(pending(conn, media_ids, workflow_id, classes)) if classes and not again and workflow_id else set(media_ids)
        images = {str(media.path(r["sha256"], r["ext"])): r for r in conn.execute(
            "SELECT id, project_id, sha256, ext, name, annotations, analysed FROM media WHERE id IN (SELECT value FROM json_each(?))",
            (json.dumps(sorted(todo)),))}
        ctx.total(len(media_ids))
        ctx.advance(len(media_ids) - len(images))
        failed, objects, made, known = 0, Counter(), 0, {}  # known: each project's classes, read once
        replaced = {c.casefold() for c in classes} if again else set()

        def skip(item, event) -> None:
            nonlocal failed
            failed += 1
            ctx.log(f"Failed on {item} at {event.node}: {event.error}")
            ctx.advance()

        # Closing the results stops the pipeline: no image starts a node after that.
        with closing(flow.run_many(list(images), models=vision.load, on_error=skip, **overrides)) as results:
            for path, outputs in results:
                image = images[path]
                kept = [projects.Annotation(**a) for a in json.loads(image["annotations"])
                        if not (a.get("score") is not None and a["class"].casefold() in replaced)]
                with conn:
                    for step in saves:
                        if step.spec.name == "save_to_babasu":
                            new = annotations(outputs[step.id], classes, min_score)
                            kept += new
                            objects.update(a.class_ for a in new)
                        else:
                            label = step.parameters.get("label") or step.id
                            name = f"{Path(image['name']).stem}-{label}.png"
                            target = step.parameters.get("project") or image["project_id"]
                            made += media.store(conn, target, _png(outputs[step.id]), name)[1]
                    if saves_annotations:
                        stored = known.setdefault(image["project_id"], projects.stored_classes(conn, image["project_id"]))
                        projects.save(conn, image["id"], image["project_id"], [a.stored() for a in kept], stored)
                    if classes and workflow_id:
                        analysed = json.loads(image["analysed"])
                        done = analysed.setdefault(workflow_id, [])
                        done += [c for c in classes if c.casefold() not in {d.casefold() for d in done}]
                        conn.execute("UPDATE media SET analysed = ? WHERE id = ?", (json.dumps(analysed), image["id"]))
                ctx.advance()
                if ctx.cancelled:
                    break
    return {"images": ctx.done - failed, "failed": failed, "found": dict(objects)} | ({"images_saved": made} if made else {})
