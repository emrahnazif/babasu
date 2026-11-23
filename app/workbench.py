"""`babasu` in the agent's code (code mode): its projects, their images and each image's file, and
the app's jobs, in the terms of vision work, so code reads pixels with pixelflow (or any library)
straight from the files and puts what it makes back as new images.

Reading is open: an image's file is its original, read where it's stored, never written. Every
change goes through `api` (adding files is an import, finding is a job), so the app checks it, shows
it and History can undo it, as the pages' changes.

In the code's child process, built on its `api` (app/code.py), which connect() gives it; the code
has this module as `babasu`, and `import babasu` is it too. Names are the API's (media_id,
filename), so what the code knows from an answer holds here. Kept light: imports nothing of the
app's but code mode.
"""

import time
from pathlib import Path
from urllib.parse import urlencode

from app.code import Refused

POLL = 0.5  # seconds between looks at a job being waited for

_api = None  # the code's api (app.code.Api), and its progress(done, total): connect()
_progress = None


def connect(api, progress) -> None:
    global _api, _progress
    _api, _progress = api, progress


class Image:
    """An image or a video in a project, as the API names it: `media_id`, `filename`, `width`,
    `height`, `class_counts`, `annotations` (its objects: class, bbox [x, y, w, h], segments,
    keypoints, score; a video's, frame and track), and `file`, the path of its file to read
    (pf.read_image(image.file), pf.VideoReader(image.file))."""

    def __init__(self, data: dict) -> None:
        self.media_id = data["media_id"]
        self.filename = data["filename"]
        self.width = data["width"]
        self.height = data["height"]
        self.class_counts = data["class_counts"]
        self.annotations = data["annotations"]
        self._file = None

    def _located(self) -> dict:
        if self._file is None:
            self._file = _api._file(self.media_id)
        return self._file

    @property
    def file(self) -> Path:
        """Its file, the original as it came (a video's too): read it, never write it."""
        return Path(self._located()["path"])

    @property
    def is_video(self) -> bool:
        return self._located()["video"]

    def set_annotations(self, annotations: list[dict]) -> dict:
        """Replaces its objects with these (the whole list: leave out what goes); answers what it did."""
        return _api.put(f"/api/images/{self.media_id}/annotations", annotations)

    def __repr__(self) -> str:
        return f"Image(media_id={self.media_id}, filename={self.filename!r})"


class Project:
    """A project: `id`, `name`, `classes` (their names), and its images."""

    def __init__(self, data: dict) -> None:
        self.id = data["id"]
        self.name = data["name"]
        self.classes = [c["name"] for c in data["classes"]]

    def images(self, where: str = "") -> list[Image]:
        """Its images (those matching the filter `where`, read_docs filters), each with its objects:
        read whole first, so changing them while going through the list is safe."""
        found, page, pages = [], 1, 1
        while page <= pages:
            data = _api.get(f"/api/projects/{self.id}?"  # 100 a page, the most the API gives
                            + urlencode({"page": page, "where": where, "images": 100, "annotations": 1}))
            pages = data["pages"]
            found += [Image(image) for image in data["images"]]
            page += 1
        return found

    def add(self, *paths: str | Path) -> dict:
        """Adds image and video files (and folders of them) as new images, waiting until they're in:
        {added, already_there, skipped}. Relative paths are in your folder."""
        started = _api.post(f"/api/projects/{self.id}/import", {"paths": [str(Path(p).resolve()) for p in paths]})
        return wait(started["job_id"])["result"]

    def find(self, classes: list[str], where: str = "") -> dict:
        """Finds and outlines these classes on its images (those matching `where`) with the default
        workflow, as the find tool does, waiting until it's done; what it found becomes their
        annotations. Answers the job's result."""
        return self._annotate({"classes": classes}, where)

    def run(self, workflow_id: str, where: str = "") -> dict:
        """Runs a saved workflow (GET /api/workflows) on its images (those matching `where`),
        waiting until it's done: what reaches its save nodes is kept. Answers the job's result."""
        return self._annotate({"workflow_id": workflow_id}, where)

    def _annotate(self, asked: dict, where: str) -> dict:
        images = {"project_id": self.id} | ({"where": where} if where else {})
        started = _api.post("/api/annotate", {"images": images} | asked)
        return wait(started["job_id"])["result"]

    def __repr__(self) -> str:
        return f"Project(id={self.id}, name={self.name!r})"


def projects() -> list[Project]:
    return [Project(data) for data in _api.get("/api/projects")]


def project(project_id: int) -> Project:
    return Project(_api.get(f"/api/projects/{project_id}?" + urlencode({"images": 0})))


def new_project(name: str, description: str = "") -> Project:
    return project(_api.post("/api/projects", {"name": name, "description": description})["id"])


def wait(job_id: int) -> dict:
    """Waits for a job to end, showing how far it is: the job (status, result). One that didn't
    succeed raises Refused with why."""
    while True:
        job = _api.get(f"/api/jobs/{job_id}")
        if job["total"]:
            _progress(job["done"], job["total"])
        if not job["active"]:
            break
        time.sleep(POLL)
    if job["status"] != "succeeded":
        raise Refused(f"Job {job_id} {job['status']}: {job['error'] or job['summary']}")
    return job
