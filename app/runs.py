"""Workflows and running them: a mozo workflow over stored images, as a job, keeping in babasu
what reaches its save nodes.

Workflows are mozo's JSON files: the built-ins ship with the app (app/workflows), the user's are in
the workspace (config.WORKFLOWS_DIR). A workflow's id is its file name; a built-in's can't be
taken by a saved one. A workflow says what to keep with its save nodes, and each keeps what reaches
it: save_to_babasu as the input image's annotations, save_image_to_babasu as a new image,
save_video_to_babasu a video's frames as a new video. Two of them keep twice; the workflow decides,
not the app.

A video is run on the frames it's worked at: the workflow's source reads it at its sample_fps
(mozo's read_media rate), and what's found is kept on the frame it's on (run.frame).
"""

import io
import json
import random
import secrets
import sqlite3
import time
from collections import Counter
from contextlib import closing
from pathlib import Path

import pixelflow as pf
from mozo.workflow import Detections, Image, Workflow, WorkflowError, catalogue, node
from mozo.workflow.node import Context

from app import config, db, history, jobs, media, projects, settings, vision

BUILTIN_DIR = config.BASE_DIR / "workflows"
KIND = "workflow_run"  # job kind

# The built-in workflows' ids (their names are in their files).
BUILTIN = ("sam3", "sam3-track", "grounding-dino", "grounding-dino-edgetam")
SAVES = ("save_to_babasu", "save_image_to_babasu", "save_video_to_babasu")  # what a run keeps: its save nodes
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
        raise ValueError(f"Nothing reaches a save node ({', '.join(SAVES)}), so a run would keep nothing")
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
def save_to_babasu(detections: Detections, run: Context) -> Detections:
    """Keep these detections in babasu as the image's annotations, in its project; a video's, on
    the frame they're on.

    The node only marks what to keep and passes it on: the run job stores what arrives here,
    since it knows the image and the database. On a video, each detection says its frame (in its
    metadata, from run.frame). mozo's save_annotations writes them to a file instead.
    """
    if run.frame is None:
        return detections
    marked = pf.Detections()
    for d in detections.detections:
        copy = d.copy(share_masks=True)  # only the metadata changes: a frame's masks aren't copied
        copy.metadata = {**(d.metadata or {}), "frame": run.frame}
        marked.add_detection(copy)
    return marked


@node(category="Output")
def save_image_to_babasu(image: Image, project: int | None = None, label: str = "") -> Image:
    """Keep this image in babasu as a new image (a drawn, cropped or edited copy): in `project` (its
    id), or the input image's project when empty, named after the input and `label`.

    As save_to_babasu, the node only marks it; the run job stores it. mozo's save_image writes files.
    """
    return image


@node(category="Output", ordered=True)
def save_video_to_babasu(image: Image, project: int | None = None, label: str = "") -> Image:
    """Keep a video's frames in babasu as a new video (drawn or edited copies of them), at the rate
    the video was read at: in `project` (its id), or the input video's project when empty, named
    after the input and `label`. On an image it keeps nothing (save_image_to_babasu keeps one).

    As save_to_babasu, the node only marks each frame; the run job writes the video. mozo's
    save_video writes a file.
    """
    return image


def _saves(workflow: Workflow) -> list:
    """The workflow's save nodes, in order."""
    return [step for step in workflow.steps.values() if step.spec.name in SAVES]


def _kept(workflow: Workflow) -> list[str]:
    """What a run of the workflow keeps, a save node at a time: "annotations", "image (label)" or
    "video (label)"."""
    return ["annotations" if step.spec.name == "save_to_babasu"
            else f"{step.spec.name.removeprefix('save_').removesuffix('_to_babasu')} ({step.parameters.get('label') or step.id})"
            for step in _saves(workflow)]


def annotations(detections: Detections, classes: list[str], min_score: float = 0.0) -> list[projects.Annotation]:
    """Detections as annotations, box and polygon (from the mask, as pixelflow outlines it) with
    the confidence as score; on a video, the frame save_to_babasu marked and the tracker's id as
    track. A detection's class: the class its prompt stood for (`classes`, by its class_id, a
    prompt's index), else the name the model gave it."""
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
        frame = (d.metadata or {}).get("frame")
        track = None if frame is None or d.tracker_id is None else int(d.tracker_id)
        found.append(projects.Annotation(class_=name, bbox=bbox, segments=d.segments, score=d.confidence,
                                         frame=frame, track=track))
    return found


def _png(image) -> bytes:
    """An image a workflow made (an array, as pixelflow has them), as a PNG to store."""
    buffer = io.BytesIO()
    pf.to_pil(image).save(buffer, format="PNG")
    return buffer.getvalue()


FLUSH_SECONDS = 5  # how often a video's new finds are kept while it runs, so they show as it goes


@jobs.kind(KIND, lane="gpu", title="Run workflow", needs=needs)
def run_job(ctx: jobs.Context, media_ids: list[int], workflow: dict, workflow_id: str | None, classes: list[str],
            prompts: list[str] | None = None, again: bool = False, min_score: float = 0.0,
            overrides: dict | None = None, tried: bool = False) -> dict:
    """A workflow over images and videos, keeping what reaches its save nodes as it goes:
    save_to_babasu's detections become the image's annotations (a video's, on their frames);
    save_image_to_babasu's image becomes a new image, save_video_to_babasu's frames a new video
    (images_saved, videos_saved). With `again`, and always on a video (it's looked at again
    whole), what a model found before of `classes` is replaced first. An image is kept in one
    transaction, a video's new finds every FLUSH_SECONDS. An image or video is marked analysed for
    `classes` by this workflow (not a draft, which has no id) once it's done; a video only if every
    frame was. `tried` only marks a Try. Crash-safe and safe to retry: a retry skips the images
    already analysed for them."""
    ctx.advance(0, "Starting…")
    flow = _loaded(workflow)
    saves = _saves(flow)
    if not saves:
        raise ValueError(f"The workflow has no save node ({', '.join(SAVES)}), so a run would keep nothing")
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
        rows = conn.execute("""SELECT id, project_id, sha256, ext, name, sample_fps FROM media
                               WHERE id IN (SELECT value FROM json_each(?)) ORDER BY id""",
                            (json.dumps(sorted(todo)),)).fetchall()
        videos = [r for r in rows if r["sample_fps"] is not None]
        images = {str(media.path(r["sha256"], r["ext"])): r for r in rows if r["sample_fps"] is None}
        # About how many frames a video's source yields (its worked frames: media.worked), for the progress.
        frames = {v["id"]: len(media.worked(media.frames(v["sha256"], v["ext"]), v["sample_fps"])) for v in videos}
        ctx.total(len(media_ids) - len(videos) + sum(frames.values()))
        ctx.advance(len(media_ids) - len(rows))
        keep = _Keeper(conn, saves, classes, min_score)
        done = failed = 0
        reading = ""  # the video whose frames are being read: a frame that fails is named by it

        def skip(item, event) -> None:
            nonlocal failed
            failed += 1
            ctx.log(f"Failed on {item if isinstance(item, str) else 'a frame of ' + reading} at {event.node}: {event.error}")
            ctx.advance()

        # Closing the results stops the pipeline: no image starts a node after that.
        with closing(flow.run_many(list(images), models=vision.load, on_error=skip, **overrides)) as results:
            for path, outputs in results:
                image = images[path]
                with conn:
                    keep.save(image, keep.found(image, outputs), replace=again)
                    keep.analysed(image, workflow_id)
                done += 1
                ctx.advance()
                if ctx.cancelled:
                    break

        if videos and not _reads_at_a_rate(flow):
            failed += len(videos)
            ctx.log("Failed on the videos: the workflow's input node doesn't read a video at a rate (read_media does)")
            ctx.advance(sum(frames.values()))
            videos = []
        for video in videos:
            if ctx.cancelled:
                break
            reading, before = video["name"], failed
            _run_video(ctx, flow, video, overrides, keep, skip)
            with conn:
                if failed == before and not ctx.cancelled:
                    keep.analysed(video, workflow_id)
            done += 1
    return keep.summary(done, failed)


def _reads_at_a_rate(flow: Workflow) -> bool:
    """Whether the workflow's source reads a video at a number of frames a second (read_media's rate)."""
    try:
        flow.parameter(f"{flow.source}.rate")
        return True
    except (KeyError, ValueError):
        return False


def _run_video(ctx: jobs.Context, flow: Workflow, video: sqlite3.Row, overrides: dict, keep: "_Keeper", skip) -> None:
    """A workflow over a video, its source reading it at its sample_fps: what a model found before
    of the classes replaced first, the new finds kept every FLUSH_SECONDS, the videos its
    save_video_to_babasu nodes write stored at the end. `skip`: on_error, for a frame that fails."""
    with keep.conn:
        keep.save(video, [], replace=True)
    found, kept, kept_at = [], 0, time.monotonic()

    def flush() -> None:
        nonlocal kept, kept_at
        if len(found) > kept:  # only what's new: the column is rewritten whole
            with keep.conn:
                keep.save(video, found[kept:], replace=False)
            kept = len(found)
        kept_at = time.monotonic()

    settings = overrides | {flow.file_parameter: str(media.path(video["sha256"], video["ext"])),
                            f"{flow.source}.rate": video["sample_fps"], f"{flow.source}.stride": 1}
    with _Writing(video, keep.saves) as writing, \
            closing(flow.process(models=vision.load, on_error=skip, **settings)) as results:
        for index, (_, outputs) in enumerate(results):
            found += keep.found(video, outputs, index)
            writing.write(outputs)
            ctx.advance()
            if time.monotonic() - kept_at >= FLUSH_SECONDS:
                flush()
            if ctx.cancelled:
                break
        flush()
        with keep.conn:
            keep.videos_made += writing.store(keep.conn)


class _Writing:
    """The videos a run writes of one video's frames: a file per save_video_to_babasu (written in
    MEDIA_DIR, at the rate it's read at), stored as a new video by store(); a file not stored is
    removed on leaving."""

    def __init__(self, video: sqlite3.Row, saves: list):
        self.video = video
        self.steps = [step for step in saves if step.spec.name == "save_video_to_babasu"]
        self.open: dict[str, tuple[Path, pf.VideoWriter]] = {}  # step id -> its file and writer, from its first frame

    def write(self, outputs: dict) -> None:
        for step in self.steps:
            if step.id not in self.open:
                file = config.MEDIA_DIR / f".{secrets.token_hex(8)}.mp4"
                self.open[step.id] = file, pf.VideoWriter(str(file), fps=self.video["sample_fps"])
            self.open[step.id][1].write(outputs[step.id])

    def store(self, conn: sqlite3.Connection) -> int:
        """Store what was written as new videos (the caller commits): how many are new."""
        made = 0
        for step in self.steps:
            if step.id not in self.open:
                continue  # no frame reached it
            file, writer = self.open.pop(step.id)
            writer.close()
            label = step.parameters.get("label") or step.id
            target = step.parameters.get("project") or self.video["project_id"]
            made += media.store_video(conn, target, file, f"{Path(self.video['name']).stem}-{label}.mp4", move=True)[1]
        return made

    def __enter__(self) -> "_Writing":
        return self

    def __exit__(self, *exc) -> None:
        for file, writer in self.open.values():
            writer.close()
            file.unlink(missing_ok=True)


class _Keeper:
    """What a run keeps of an image's or a video's results (run_job): the annotations reaching
    save_to_babasu, the images reaching save_image_to_babasu (a video's frames: _Writing), the classes it looked for; and what it
    did, for the job's result. Each call that writes is in the caller's transaction."""

    def __init__(self, conn: sqlite3.Connection, saves: list, classes: list[str], min_score: float):
        self.conn, self.saves, self.classes, self.min_score = conn, saves, classes, min_score
        self.annotates = any(step.spec.name == "save_to_babasu" for step in saves)
        self.replaced = {c.casefold() for c in classes}
        self.objects, self.tracks, self.known = Counter(), set(), {}  # known: each project's classes, read once
        self.frames = self.made = self.videos_made = 0  # a video's frames done; new images and videos saved

    def found(self, row: sqlite3.Row, outputs: dict, index: int | None = None) -> list[dict]:
        """One image's (or a video frame's: item `index`) results: its annotations as stored; the
        images it made stored as new images."""
        new = []
        self.frames += index is not None
        for step in self.saves:
            if step.spec.name == "save_video_to_babasu":
                continue  # _Writing's, on a video; nothing on an image
            if step.spec.name == "save_to_babasu":
                found = annotations(outputs[step.id], self.classes, self.min_score)
                self.objects.update(a.class_ for a in found if a.track is None)
                self.tracks.update((row["id"], a.class_, a.track) for a in found if a.track is not None)
                new += [a.stored() for a in found]
            else:
                label = step.parameters.get("label") or step.id
                name = f"{Path(row['name']).stem}-{label}{'' if index is None else f'-{index}'}.png"
                target = step.parameters.get("project") or row["project_id"]
                self.made += media.store(self.conn, target, _png(outputs[step.id]), name)[1]
        return new

    def save(self, row: sqlite3.Row, found: list[dict], replace: bool) -> None:
        """Keep `found` as the image's (or video's) new annotations, next to those it has (read now,
        so edits made meanwhile stay); `replace`: without what a model found before of the classes."""
        if not self.annotates:
            return
        current = json.loads(self.conn.execute("SELECT annotations FROM media WHERE id = ?", (row["id"],)).fetchone()[0])
        kept = [a for a in current if not (replace and a.get("score") is not None and a["class"].casefold() in self.replaced)]
        stored = self.known.setdefault(row["project_id"], projects.stored_classes(self.conn, row["project_id"]))
        projects.save(self.conn, row["id"], row["project_id"], kept + found, stored)

    def analysed(self, row: sqlite3.Row, workflow_id: str | None) -> None:
        """Mark the image or video looked at for the classes by this workflow."""
        if not (self.classes and workflow_id):
            return
        analysed = json.loads(self.conn.execute("SELECT analysed FROM media WHERE id = ?", (row["id"],)).fetchone()[0])
        done = analysed.setdefault(workflow_id, [])
        done += [c for c in self.classes if c.casefold() not in {d.casefold() for d in done}]
        self.conn.execute("UPDATE media SET analysed = ? WHERE id = ?", (json.dumps(analysed), row["id"]))

    def summary(self, images: int, failed: int) -> dict:
        """The job's result: images (videos too) done, failed (images and frames), objects found by
        class (a tracked one once), a video's frames done, images and videos saved."""
        found = self.objects + Counter(name for _, name, _ in self.tracks)
        return ({"images": images, "failed": failed, "found": dict(found)}
                | ({"frames": self.frames} if self.frames else {}) | ({"images_saved": self.made} if self.made else {})
                | ({"videos_saved": self.videos_made} if self.videos_made else {}))
