"""Workflows and their runs: a mozo workflow over stored images, as a job, with its results kept
in babasu.

Workflows are mozo's JSON files: the built-ins ship with the app (app/workflows), the user's are in
the workspace (config.WORKFLOWS_DIR). A workflow's id is its file name; a built-in's can't be
taken by a saved one. A workflow says what to keep with save_to_babasu nodes: one run of a
workflow with two of them (say SAM 3, and Grounding DINO → EdgeTAM) keeps two labelled results
per image.
"""

from pathlib import Path

from mozo.workflow import Detections, Workflow, node

from app import config

BUILTIN_DIR = config.BASE_DIR / "workflows"

# The built-in workflows: id -> name. mozo's workflow file has no name, so a saved one is named
# by its id.
BUILTIN = {
    "sam3": "SAM 3",
    "grounding-dino": "Grounding DINO",
    "grounding-dino-edgetam": "Grounding DINO → EdgeTAM",
}


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


def _entry(id: str) -> dict:
    return {"id": id, "name": BUILTIN.get(id, id), "builtin": id in BUILTIN}


def workflows() -> list[dict]:
    """Every workflow: id, name, whether it's built in, and the models it uses (as detail() lists
    them), or `error`: why mozo doesn't load it."""
    out = []
    for id, path in _files().items():
        try:
            models = Workflow.load(path).check(device=config.DEVICE).models
        except (KeyError, ValueError) as e:  # mozo's refusal of the file
            out.append(_entry(id) | {"error": str(e)})
            continue
        out.append(_entry(id) | {"models": [use._asdict() for use in models]})
    return out


def load(id: str) -> Workflow | None:
    """The workflow read from its file, or None if there's no such workflow. A file that doesn't
    load raises mozo's error: KeyError for a node this mozo doesn't have, ValueError otherwise."""
    path = _files().get(id)
    return Workflow.load(path) if path else None


def detail(id: str, workflow: Workflow) -> dict:
    """A loaded workflow with its graph (mozo's {nodes, edges}), the models it uses (each with
    its licence and the bytes still to download for config.DEVICE, 0 once downloaded) and the
    files it names that are missing."""
    found = workflow.check(device=config.DEVICE)
    return _entry(id) | {
        "graph": workflow.to_dict(),
        "models": [use._asdict() for use in found.models],
        "missing_files": [{"node": node, "parameter": name, "path": path} for node, name, path in found.missing],
    }
