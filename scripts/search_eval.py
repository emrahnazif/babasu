"""How well search_docs finds what the agent asks for: real queries the agent sent to its docs and
API lookups, each with the parts that answer it (written by hand), scored by whether one of them is
in the 3 results, by words, by meaning (mozo's Granite, from its download cache) and both combined.
Run it after changing the docs or the search: uv run python scripts/search_eval.py
"""

import os
import sys
import tempfile
from pathlib import Path

MODELS = Path.home() / ".cache" / "mozo"  # mozo's own download cache, as any script using it has
os.environ["HOME"] = tempfile.mkdtemp()  # a workspace of its own, never the real one
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mozo import weights  # noqa: E402

import app.server  # noqa: E402,F401  (the API, for the operations)
from app import agent_tools as t, docs_index  # noqa: E402

N, CH, RN, CK = "GET /api/workflows/nodes", "POST /api/projects/{project_id}/changes", "POST /api/projects/{project_id}/classes/rename", "POST /api/workflows/check"
WB, GREY = "workflows ## Building one", "code ## Grey copies of a project's images and videos"
CHG = [CH, "projects ## Changing many images at once", "api ## Changing them"]
DELO = CHG + ["code ## Delete one class's objects"]
RENAME = [RN, "projects ## Classes"]
ANN = ["POST /api/annotate", "auto-annotate ## A run", "GET /api/projects/{project_id}/annotate", "api ## Finding things with a model"]
CPY = ["POST /api/projects/{project_id}/images", "projects ## Copying and moving images", CH]
SAVE = ["POST /api/workflows", "PUT /api/workflows/{workflow_id}", WB]
DL = ["GET /api/images/{media_id}", "code ## babasu"]
CROP = [N, WB, "code ## Crop every person into a new project"]
QUERIES = {
    "/api/projects/59/changes": CHG, "annotate images in a project": ANN, "annotate run": ANN,
    "change class name project": RENAME, "changes": CHG,
    "check a workflow document before saving": [CK, WB, "api ## Building a workflow"],
    "check workflow": [CK, WB, "api ## Building a workflow"], "check workflow post /api/workflows/check": [CK],
    "convert image greyscale": [GREY, "pixelflow/utilities/transforms"],
    "copy images into a project": CPY, "copy images to another project": CPY,
    "create project": ["POST /api/projects", "projects ## A project"],
    "create workflow": ["POST /api/workflows", WB, "api ## Building a workflow"],
    "crop around detections save image output": CROP, "crop_around_detections": CROP,
    "data lake unlabelled images": ["app ## Finding what's in images nobody annotated", "GET /api/images", "filters # Filters"],
    "delete /api/projects/119/changes": CHG,
    "delete annotations": DELO + ["PUT /api/images/{media_id}/annotations", "projects ## Deleting"],
    "delete class": ["PATCH /api/projects/{project_id}/classes", "projects ## Classes", "code ## Delete one class's objects",
                     "POST /api/projects/{project_id}/classes/clear"],
    "delete objects": DELO, "delete objects in project": DELO,
    "delete project": ["DELETE /api/projects/{project_id}", "projects ## Deleting", "GET /api/projects/{project_id}/deletion"],
    "delete workflow": ["DELETE /api/workflows/{workflow_id}"],
    "detections filter score keep": [N, "pixelflow/core-concepts/filters"],
    "download media file": DL, "media file download": DL,
    "draw_boxes draw_labels crop_around_detections pad_detections track clip": [N, WB],
    "draw_masks draw_skeleton draw_keypoints blur_regions read_media": [N, WB],
    "draw_polygons, draw_labels, save_to_babasu": [N, WB],
    "easyocr": [N, "models ## Vision models", "GET /api/vision/models"], "enhance_clahe": [N],
    "find_segment_sam3 sam3": [N, "models ## Vision models", "GET /api/vision/models/{family}"],
    "grounding_dino node parameters": [N],
    "image objects count": ["GET /api/projects/{project_id}/stats", "GET /api/images/{media_id}", "filters ## Measuring"],
    "import grey images": ["POST /api/projects/{project_id}/import", GREY],
    "import images from files": ["POST /api/projects/{project_id}/import"],
    "list images": ["GET /api/projects/{project_id}", "GET /api/images", "api ## Reading a project and its images", "code ## A project's images"],
    "nodes": [N], "object detection node types": [N, WB],
    "one image and its objects": ["GET /api/images/{media_id}", "api ## Reading a project and its images"],
    "patch /api/projects/174/changes": [CH], "patch /api/projects/64/classes/rename": [RN],
    "patch /api/projects/{id}/classes hide": ["PATCH /api/projects/{project_id}/classes", "projects ## Classes"],
    "patch /api/projects/{project_id}/changes delete objects": [CH],
    "patch /api/workflows/{id}": ["PUT /api/workflows/{workflow_id}"],
    "post /api/projects/222/changes": [CH], "post /api/projects/60/changes": [CH],
    "post /api/projects/226/classes/rename": [RN], "post /api/projects/{id}/classes/clear": ["POST /api/projects/{project_id}/classes/clear"],
    "post /api/projects/{id}/images": ["POST /api/projects/{project_id}/images"],
    "post /api/projects/{id}/import": ["POST /api/projects/{project_id}/import"],
    "post /api/workflows body": ["POST /api/workflows"],
    "post /api/workflows check and post /api/workflows save": [CK, "POST /api/workflows"],
    "project classes rename": RENAME, "projects annotate auto-annotate": ANN,
    "projects/{project_id}/changes filter object dimensions": [CH, "filters ## Measuring", "filters # Filters"],
    "rename class": RENAME, "rename class project id": RENAME, "rename": RENAME, "rename classes": RENAME,
    "save workflow": SAVE, "save workflow by name": SAVE,
    "sam3 node": [N], "sam3_boxes segment node inputs outputs": [N],
    "video frames": ["GET /api/images/{media_id}/frames", "GET /api/images/{media_id}/frames/{frame}", "annotating ## Videos"],
    "workflow check": [CK], "workflow creation": ["POST /api/workflows", WB],
    "workflow node parameters details": [N],
    "workflow structure": ["workflows ## Workflows", WB, "api ## Building a workflow"],
    "workflows nodes": [N], "workflows nodes list": [N], "workflows/nodes": [N], "workflows/nodes full": [N],
    "annotate": ANN + ["auto-annotate # Auto-annotate", "annotating # Annotating"],
    "annotations": ["annotating # Annotating", "PUT /api/images/{media_id}/annotations", "code ## A project's images", "projects ## Images"],
    "change": CHG, "changes api reference": CHG,
    "classes": ["projects ## Classes", "GET /api/projects/{project_id}/classes", "PUT /api/projects/{project_id}/classes",
                "PATCH /api/projects/{project_id}/classes"],
    "find": ["api ## Finding things with a model", "POST /api/annotate", "api ## Finding something by its name", "GET /api/search",
             "app ## Finding what's in images nobody annotated"],
    "object filtering": ["filters # Filters", "filters ## Measuring"], "objects:range": ["filters # Filters", "filters ## Measuring"],
    "workflow definition": ["workflows ## Workflows", WB, "api ## Building a workflow", "workflows # Workflows and running them"],
}


def name(found: dict) -> str:
    return found.get("operation") or f"{found['doc']} {found['text'].partition('\n')[0]}"


def answers(found: dict, wanted: list[str]) -> bool:
    return any(name(found) == w or (w.startswith("pixelflow/") and found.get("doc") == w) for w in wanted)


class _Quiet:
    """What the index job reports progress to, here: nowhere."""
    cancelled = False

    def total(self, n):
        pass

    def advance(self, n=1, message=None):
        pass


def main() -> None:
    weights.configure(cache=MODELS)
    docs_index.build(_Quiet(), **docs_index.MODEL)  # as the app's job does, in this run's own workspace
    parts = [found for found, _ in t._index().parts]
    ways = {"words": lambda q: [parts[at] for at in t._by_words(q)],
            "meaning": lambda q: [parts[at] for at in docs_index.ranking(q)],
            "both": t._ranked}  # what search_docs answers
    for way, ranked in ways.items():
        hits = sum(any(answers(f, wanted) for f in (t._at_path(q) or ranked(q))[:3]) for q, wanted in QUERIES.items())
        print(f"{way:8} {hits}/{len(QUERIES)} ({hits / len(QUERIES):.0%}) with the right part in the top 3")


if __name__ == "__main__":
    main()
