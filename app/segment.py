"""Click to segment: a click model (EdgeTAM, SAM 2 or SAM 3, from mozo) on one image, from points
and a box.

The model is the one chosen (a setting, EdgeTAM by default). One that isn't downloaded is
downloaded when chosen; meanwhile the first downloaded one in MODELS' order answers.

The image is prepared once (its encoding, the expensive half, kept by the model); each prompt then
runs only the cheap half, fast enough for every mouse move. That is why it's called directly
rather than through a workflow (product.md). One call runs at a time, and a call that's no longer
the latest when its turn comes is skipped: the pointer has moved on.
"""

import itertools
import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pixelflow as pf
from mozo import weights
from mozo.image import load_image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app import annotators, config, projects, settings, vision

# The click models, in the menu's order: "family/variant" -> name. One SAM 2, the large one: every
# model takes about the same time a click (40-50 ms), so a bigger one costs only its prepare,
# done ahead, and memory.
MODELS = {"edgetam/edgetam": "EdgeTAM", "sam2/large": "SAM 2", "sam3/sam3": "SAM 3"}
Model = Literal[tuple(MODELS)]  # one of MODELS' keys


class Change(BaseModel):
    """A change to the click tool: on or off, and/or its model."""

    model_config = ConfigDict(extra="forbid")
    on: bool | None = Field(None, description="Turn the click tool on (true) or off (false); omit to leave it.")
    model: Model | None = Field(
        None, description="Its model: edgetam/edgetam (EdgeTAM, the default), sam2/large (SAM 2) or sam3/sam3 (SAM 3); omit to leave it.")

Point = tuple[float, float]


class Prompt(BaseModel):
    """What to segment, in the image's own pixels: points to include or leave out, and/or a box."""

    model_config = ConfigDict(extra="forbid")
    points: list[Point] = Field(default=[], description="x, y clicks.")
    labels: list[int] = Field(default=[], description="One per point: 1 includes it, 0 leaves it out.")
    box: tuple[float, float, float, float] | None = Field(default=None, description="x1, y1, x2, y2.")
    class_id: int = Field(description="The class the object is for; the answer is an annotation of it.")

    @model_validator(mode="after")
    def _a_prompt(self):
        if not self.points and self.box is None:
            raise ValueError("give points, a box, or both")
        if len(self.labels) != len(self.points) or set(self.labels) - {0, 1}:
            raise ValueError("labels: one per point, 1 to include it or 0 to leave it out")
        return self


class Superseded(Exception):
    """A newer call came while this one waited: its answer would be for where the pointer was."""


_turn = threading.Lock()  # one call at a time
_numbering = threading.Lock()  # each call gets its number and becomes the latest, together
_calls = itertools.count(1)
_latest = 0


def _parts(model: str) -> tuple[str, str]:
    family, variant = model.split("/")
    return family, variant


def in_use(downloaded: set[str] | None = None) -> str | None:
    """The model that answers: the chosen one, or while it isn't downloaded the first downloaded
    one in MODELS' order; None if none is. `downloaded`: the downloaded models, if known."""
    chosen = settings.get().click_model
    have = (lambda m: m in downloaded) if downloaded is not None else (lambda m: vision.downloaded(*_parts(m)))
    return next((m for m in (chosen, *MODELS) if m in MODELS and have(m)), None)


def tool(conn) -> dict:
    """The editor's click tool, as its menu shows it: on or off, the chosen model, the one in use,
    and each model with its name, licence, and whether it's downloaded, downloading or how many
    bytes it would download."""
    s, busy = settings.get(), vision.busy(conn)
    sizes = {m: weights.download_size(*_parts(m), device=config.DEVICE) for m in MODELS}  # 0: downloaded
    return {
        "on": s.segment_tool,
        "model": s.click_model,
        "in_use": in_use({m for m, size in sizes.items() if size == 0}),
        "models": [{"id": m, "name": name, "license": weights.license_of(*_parts(m)), "downloaded": sizes[m] == 0,
                    "downloading": busy.get(_parts(m)) == "downloading", "to_download": sizes[m]}
                   for m, name in MODELS.items()],
    }


def change(conn, change: Change) -> dict:
    """Turn the tool on or off and/or choose its model, downloading it if it isn't (a job); returns
    the tool as tool() does."""
    fields = {"segment_tool": change.on, "click_model": change.model}
    settings.update(**{k: v for k, v in fields.items() if v is not None})
    if change.model:
        vision.ensure(conn, *_parts(change.model))
    return tool(conn)


def _model():
    model = in_use()
    if model is None:
        raise LookupError("No click model is downloaded yet")
    return vision.load(*_parts(model))


@lru_cache(maxsize=3)
def _pixels(path: Path) -> np.ndarray:
    """The image decoded, kept for the image being worked on, the one before and the one prepared
    next (the model keeps their encodings too)."""
    return load_image(path)


def prepare(path: Path) -> None:
    """Encode the image now, so its prompts run only the cheap half."""
    _model().prepare(_pixels(path))


def segment(path: Path, prompt: Prompt, task: projects.Task) -> projects.Annotation | None:
    """The object the prompt points at, as an annotation of `prompt.class_id`: its box, and for a
    segmentation (or keypoints) project its polygon too; None if nothing was found. Raises
    Superseded if a newer call came while this one waited; LookupError if no click model is
    downloaded."""
    global _latest
    with _numbering:
        call = _latest = next(_calls)
    with _turn:
        if call != _latest:
            raise Superseded
        # One click is ambiguous (the handle, the door, the car): three candidates, the best kept.
        # More points or a box say which, and one mask is the model's answer to them.
        found = _model().predict(
            _pixels(path),
            points=np.array(prompt.points, dtype=np.float32) if prompt.points else None,
            labels=np.array(prompt.labels, dtype=np.int64) if prompt.points else None,
            boxes=np.array(prompt.box, dtype=np.float32) if prompt.box else None,
            multimask_output=len(prompt.points) == 1 and prompt.box is None,
        )
    if not len(found):
        return None
    best = pf.detections.Detections()
    best.add_detection(found[0])  # ranked by the model's own score, best first
    annotation = next(iter(annotators.to_annotations(best, [prompt.class_id])), None)
    if annotation and task == "detection":
        return annotation.model_copy(update={"segments": None})
    return annotation
