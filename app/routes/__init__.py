"""Route modules (one per area), plus what they share."""

import random
import sqlite3
from collections.abc import Iterator

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app import media
from app.filters import WHERE, Filter

# Not `from app import projects` (nor runs): in this package those names are route modules.
from app.projects import PROMPT, Annotation
from app.runs import found_on, on_project

# The tag of API operations only the app's own pages use (the chat's, uploads): the agent's list
# of operations leaves them out (agent_tools.operations).
UI_ONLY = "ui-only"


def _query_parameters(dependant) -> Iterator[str]:
    """The query parameters a route takes, its dependencies' too."""
    yield from (p.alias for p in dependant.query_params)
    for inner in dependant.dependencies:
        yield from _query_parameters(inner)


def known_query(request: Request) -> None:
    """An /api operation refuses a query parameter it doesn't take (422, naming those it does), as
    a tool refuses an argument it doesn't take: one misspelt or made up is a mistake to say, not to
    ignore, or the answer reads as if it had been taken (all the projects as if filtered). The
    app's dependency for every route; pages keep theirs."""
    route = request.scope.get("route")
    if route is None or not request.url.path.startswith("/api/"):
        return
    taken = set(_query_parameters(route.dependant))
    if unknown := sorted(set(request.query_params) - taken):
        raise HTTPException(422, f"Not a parameter of {route.path}: {', '.join(unknown)}; it takes "
                                 + (", ".join(sorted(taken)) or "none"))


class Answer(BaseModel):
    """What an API operation answers (its response_model): find_api shows its fields to the agent.
    A field the code adds without declaring it is an error, so the description can't drift from
    the answer. A field with a default may be left out of an answer: its route uses
    response_model_exclude_unset, so it stays out. Plain types only (no validators): every answer
    passes through its model."""

    model_config = ConfigDict(extra="forbid")


def _said(field: str) -> str:
    return Annotation.model_fields[field].description


class KeypointOut(Answer):
    name: str = Field(description="Which point, e.g. left_eye.")
    point: tuple[float, float] = Field(description="x, y in pixels of the original image.")


class AnnotationOut(Answer):
    """An Annotation as an answer has it: checked when it was written, not again on each read."""

    class_id: int = Field(description=_said("class_id"))
    bbox: tuple[float, float, float, float] | None = Field(None, description=_said("bbox"))
    segments: list[tuple[float, float]] | None = Field(None, description=_said("segments"))
    keypoints: list[KeypointOut] | None = Field(None, description=_said("keypoints"))
    score: float | None = Field(None, description=_said("score"))


IMAGES = ("Which images: {\"media_ids\": [...]} (e.g. those attached to a message), {\"project_id\": 3, \"where\": "
          "\"night\", \"count\": 10} (a project's, matching a filter, some at random), or {\"run_id\": 31, \"found\": "
          "\"person\"} (those a run found that on); each with its name.")


class Images(BaseModel):
    """Which images, one way: by id; a project's (those matching a filter); or a run's (those it
    found a class on). Any of them, some at random (count)."""

    model_config = ConfigDict(extra="forbid")
    media_ids: list[int] | None = Field(None, description="By id, e.g. those attached to a message.")
    project_id: int | None = Field(None, description="A project's images (with where).")
    where: str = Field("", max_length=2000, description=WHERE)
    run_id: int | None = Field(None, description="A run's images (with found: those it found that on).")
    found: str | None = Field(None, description="With run_id: a class name; only the images the run found it on.")
    count: int | None = Field(None, ge=1, description="That many of them, picked at random; empty for all.")

    @model_validator(mode="after")
    def _one_way(self):
        if sum(given is not None for given in (self.media_ids, self.project_id, self.run_id)) != 1:
            raise ValueError("Name the images one way: media_ids, project_id (with where) or run_id (with found)")
        if self.where and self.project_id is None:
            raise ValueError("where filters a project's images: give project_id with it")
        if self.found and self.run_id is None:
            raise ValueError("found picks a run's images: give run_id with it")
        return self

    def resolve(self, conn: sqlite3.Connection) -> list[dict]:
        """The images, as media.listed has them (named as their project names them, if they're a
        project's or a run of one's). ValueError saying why there are none."""
        project_id = self.project_id
        if self.media_ids is not None:
            ids = list(dict.fromkeys(self.media_ids))
        elif self.project_id is not None:
            ids = on_project(conn, self.project_id, Filter.parse(self.where))
        else:
            project_id, ids = found_on(conn, self.run_id, self.found)
        if self.count is not None and self.count < len(ids):
            ids = sorted(random.sample(ids, self.count))
        images = media.listed(conn, ids, project_id)
        if missing := sorted(set(ids) - {i["media_id"] for i in images}):
            raise ValueError(f"No stored images {missing}")
        return images


class ClassOut(Answer):
    id: int = Field(description="The class's id in its project (what annotations' class_id refers to).")
    name: str = Field(description="Its name, e.g. car.")
    color: str = Field(description="The colour it's drawn in, #rrggbb.")
    prompt: str = Field(description=PROMPT)


class ToDownload(Answer):
    model: str = Field(description="The model: its family and variant, e.g. \"sam3 base\".")
    size: int = Field(description="Bytes still to download.")


def safe_next(next: str | None) -> str:
    """A return target from a form or query: only local paths are allowed, anything else is "/"."""
    if next and next.startswith("/") and not next.startswith("//"):
        return next
    return "/"


def message(exc: ValueError) -> str:
    """A failed change explained in one line: a pydantic error's first message, or the error's text."""
    if isinstance(exc, ValidationError):
        error = exc.errors(include_url=False)[0]
        field = ".".join(str(part) for part in error["loc"])
        text = error["msg"].removeprefix("Value error, ")
        return f"{field}: {text}" if field else text
    return str(exc)
