"""Route modules (one per area), plus what they share."""

import json
import random
import sqlite3
from collections.abc import Iterator
from typing import Annotated, get_args

from fastapi import HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

from app import media
from app.filters import WHERE, Filter
from app.formatting import filesize

# Not `from app import projects` (nor runs, vision): in this package those names are route modules.
from app.projects import PROMPT, Annotation, sample
from app.vision import downloads_first

# The tag of API operations only the app's own pages use (the chat's, uploads): the agent's list
# of operations leaves them out (agent_tools.operations).
UI_ONLY = "ui-only"
# The tag of POST operations that only read (a check, a test, a suggestion): they answer their
# result, not what they did (Done; check_writes).
READS = "reads"


def _query_parameters(dependant) -> Iterator[str]:
    """The query parameters a route takes, its dependencies' too; a query model's (Images) each."""
    for param in dependant.query_params:
        model = param.field_info.annotation
        if isinstance(model, type) and issubclass(model, BaseModel):
            yield from (field.alias or name for name, field in model.model_fields.items())
        else:
            yield param.alias
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


def fields_at(annotation, loc: tuple) -> list[str]:
    """The fields of the object at `loc` (field names and list indexes) in a model: those an
    unknown name there could have been. KeyError or StopIteration if `loc` isn't in it."""
    return [f.alias or name for name, f in _model_at(annotation, loc).model_fields.items()]  # as JSON writes them: class, not class_


def field_at(annotation, loc: tuple):
    """The field at `loc` (its last part a field name) in a model, its FieldInfo: its description
    says what it takes. KeyError or StopIteration if `loc` isn't in it."""
    return _field(_model_at(annotation, loc[:-1]), loc[-1])


def _model_at(annotation, loc: tuple) -> type[BaseModel]:
    """The model of the object at `loc` in a model (list indexes skipped)."""
    model = _model(annotation)
    for part in loc:
        if isinstance(part, str):
            model = _model(_field(model, part).annotation)
    return model


def _field(model: type[BaseModel], name: str):
    return next(f for key, f in model.model_fields.items() if name in (key, f.alias))


def _model(annotation) -> type[BaseModel]:
    """The model an annotation holds: list[ClassEdit] | None -> ClassEdit."""
    def types(a) -> list:
        return [a, *(t for arg in get_args(a) for t in types(arg))]
    return next(t for t in types(annotation) if isinstance(t, type) and issubclass(t, BaseModel))


async def refused(request: Request, exc: RequestValidationError) -> JSONResponse:
    """A request whose body or parameters don't fit (422), as FastAPI answers it, except that the body
    says how to correct it: a field it doesn't take says which it does, as a parameter it doesn't
    take says (known_query); a field missing, or not the object it should be, says what it takes
    (its description, its fields). A made-up or misshapen one is a mistake to correct, and that's how."""
    body = getattr(request.scope.get("route"), "body_field", None)
    errors = []
    for error in exc.errors():
        loc, said = error["loc"], None
        try:
            if body is None or loc[:1] != ("body",):
                pass
            elif error["type"] == "missing" and loc == ("body",):
                said = f"a JSON body is needed; it takes {', '.join(fields_at(body.field_info.annotation, ()))}"
            elif error["type"] == "extra_forbidden":
                said = f"not a field here; it takes {', '.join(fields_at(body.field_info.annotation, loc[1:-1]))}"
            elif error["type"] in ("missing", "model_attributes_type", "model_type", "dict_type"):
                field = field_at(body.field_info.annotation, loc[1:])
                wrong = ("Field required" if error["type"] == "missing"
                         else f"Should be an object of {', '.join(fields_at(body.field_info.annotation, loc[1:]))}")
                said = f"{wrong}. {field.description}" if field.description else wrong
        except (KeyError, StopIteration):
            pass
        errors.append(error | {"msg": said} if said else error)
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


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

    class_: str = Field(alias="class", description=_said("class_"))
    bbox: tuple[float, float, float, float] | None = Field(None, description=_said("bbox"))
    segments: list[tuple[float, float]] | None = Field(None, description=_said("segments"))
    keypoints: list[KeypointOut] | None = Field(None, description=_said("keypoints"))
    score: float | None = Field(None, description=_said("score"))
    frame: int | None = Field(None, description=_said("frame"))
    track: int | None = Field(None, description=_said("track"))


class Done(Answer):
    """What a write answers: what it did, in a sentence (`did`), so whoever made it (the agent too)
    knows what happened; with what a next step needs (an id, the classes). Not `done`: a job's
    answer has that (how much of it is done)."""

    did: str = Field(description="What it did, e.g. “Added 8 images to Sandwiches (2 there already)”.")


class ToDownload(Answer):
    model: str = Field(description="The model: its family and variant, e.g. \"sam3 base\".")
    size: int = Field(description="Bytes still to download.")


class Started(Done):
    """A write that starts a job: what it does is the job's (follow it)."""

    job_id: int = Field(description="Follow it at /api/jobs/{job_id}.")
    images: int | None = Field(None, description="How many images it runs on.")
    downloads_first: list[ToDownload] | None = Field(None, description="The models it downloads before it starts.")


def started(conn: sqlite3.Connection, job_id: int, did: str) -> dict:
    """A Started answer's end: the models the job downloads before it starts, and what was done,
    saying so."""
    first = downloads_first(conn, job_id)
    size = sum(m["size"] for m in first)
    return {"downloads_first": first, "did": did + (f"; it downloads {filesize(size)} of models first" if size else "")}


def check_writes(openapi: dict) -> None:
    """Every write of the API says what it did (its answer is a Done), checked when the app starts,
    as an Answer's undeclared field is: a write that doesn't would leave whoever made it guessing."""
    said = {name for name, schema in openapi["components"]["schemas"].items() if "did" in schema.get("properties", {})}
    missing = [f"{method.upper()} {path}" for path, operations in openapi["paths"].items() if path.startswith("/api/")
               for method, operation in operations.items()
               if method != "get" and not {UI_ONLY, READS} & set(operation.get("tags", []))
               and next(iter(operation["responses"].values())).get("content", {}).get("application/json", {})
                   .get("schema", {}).get("$ref", "").split("/")[-1] not in said]
    if missing:
        raise RuntimeError(f"Writes that don't answer what they did (a Done response_model): {', '.join(missing)}")


IMAGES = ("Which images: {\"project_id\": 3, \"where\": \"night\", \"count\": 10} (a project's, matching a filter, "
          "some at random), {\"where\": \"tomato\"} (every project's matching it) or {\"media_ids\": [...]}.")


def _comma_list(value):
    """A list in a query, written either way: ?ids=1,2 or ?ids=1&ids=2 (a JSON list passes as it is)."""
    if isinstance(value, list):
        return [part for item in value for part in ([p.strip() for p in item.split(",")] if isinstance(item, str) else [item])
                if part != ""]
    return value


CommaList = BeforeValidator(_comma_list)


class NoImages(ValueError):
    """No images match: to read, that's the answer (0); to work on, there's nothing to do."""


class Images(BaseModel):
    """Which images, one way: by id, or a project's (or every project's) matching a filter; any of
    them, some at random (count)."""

    model_config = ConfigDict(extra="forbid")
    media_ids: Annotated[list[int] | None, CommaList] = Field(None, description="By id (in a query: ?media_ids=1,2).")
    project_id: int | None = Field(None, description="A project's images (with where); without one, where looks in every project.")
    where: str | None = Field(None, max_length=2000, description=WHERE)
    count: int | None = Field(None, ge=1, description="That many of them, picked at random; empty for all.")

    @model_validator(mode="after")
    def _one_way(self):
        if (self.media_ids is None) == (self.project_id is None and self.where is None):
            raise ValueError("Name the images one way: media_ids, or project_id and/or where (every project's)")
        return self

    def ids(self, conn: sqlite3.Connection) -> list[int]:
        """The images' ids. ValueError saying why there are none, or which ids aren't stored images."""
        if self.media_ids is not None:
            ids = list(dict.fromkeys(self.media_ids))
            stored = {r[0] for r in conn.execute("SELECT id FROM media WHERE id IN (SELECT value FROM json_each(?))",
                                                 (json.dumps(ids),))}
            if missing := sorted(set(ids) - stored):
                raise ValueError(f"No stored images {missing}")
        else:
            if self.project_id is not None and conn.execute("SELECT 1 FROM project WHERE id = ?", (self.project_id,)).fetchone() is None:
                raise ValueError(f"There is no project {self.project_id}")
            if not (ids := sample(conn, Filter.parse(self.where or ""), self.project_id)):
                raise NoImages("No images match that filter")
        if self.count is not None and self.count < len(ids):
            ids = sorted(random.sample(ids, self.count))
        return ids

    def resolve(self, conn: sqlite3.Connection) -> list[dict]:
        """The images, as media.listed has them. ValueError as ids()."""
        return media.listed(conn, self.ids(conn))


class ClassOut(Answer):
    name: str = Field(description="Its name, e.g. car (what annotations' class says).")
    color: str = Field(description="The colour it's drawn in, #rrggbb.")
    prompt: str = Field(description=PROMPT)
    hidden: bool = Field(description="Hidden on the project's pages (they don't draw it); its annotations are still here.")


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
