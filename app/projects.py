"""Projects and their media (product.md "Projects, media and annotations"). Reads return plain dicts
(JSON columns parsed), ready for templates and the API.

An image is a row of `media` with its own id, in one project; its annotations are a list on the
row. A class is a name on an annotation: a project keeps its classes (name, colour, prompt, in
order), and an annotation of a name it hasn't had adds one. Names are matched ignoring case, and an
annotation written as "Tomato" in a project that has "tomato" is saved as "tomato", so one class
never becomes two.
"""

import json
import math
import sqlite3
from collections import Counter
from typing import Annotated, Literal

from PIL import ImageColor
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app import db, jobs, media
from app.filters import Filter, FilterError, ObjectCondition

# What a project is for. It suggests (an annotator: annotators.SUGGESTED; later what to train), it
# doesn't restrict: every project may hold every shape. Optional; classification comes later.
Task = Literal["detection", "segmentation", "keypoints"]
TASKS: dict[Task, str] = {"detection": "Object detection", "segmentation": "Segmentation", "keypoints": "Keypoint detection"}

# Pixel coordinate on the original image, rounded to 1 decimal.
Coord = Annotated[float, AfterValidator(lambda v: round(v, 1))]

# What people type for a project and its classes, checked the same way wherever it comes in.
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Description = Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)]
ClassName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
Prompt = Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)]
PROMPT = ("What annotators look for (not its name), e.g. 'hole in the road surface' for a class named pothole; "
          "empty: its name.")


class Keypoint(BaseModel):
    """One named point of an object."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="Which point, e.g. left_eye.")
    point: tuple[Coord, Coord] = Field(description="x, y in pixels of the original image.")


class Annotation(BaseModel):
    """One thing in an image, with whichever shapes it has (like COCO and pixelflow's Detection)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)
    class_: ClassName = Field(alias="class", description="Its class's name, e.g. car; a new name is a new class.")
    bbox: tuple[Coord, Coord, Coord, Coord] | None = Field(None, description="Its box: x, y, width, height in pixels.")
    segments: list[tuple[Coord, Coord]] | None = Field(None, description="Its outline: polygon points [x, y] in pixels.")
    keypoints: list[Keypoint] | None = Field(None, description="Its named points.")
    score: Annotated[float, Field(ge=0, le=1), AfterValidator(lambda v: round(v, 3))] | None = Field(
        None, description="A model's confidence (0..1), on an object a model made and nobody has reviewed yet.")

    @model_validator(mode="after")
    def _has_a_shape(self):
        if self.bbox is None and not self.segments and not self.keypoints:
            raise ValueError("an annotation needs a bbox, segments or keypoints")
        return self

    def stored(self) -> dict:
        """As it's kept in media.annotations."""
        return self.model_dump(exclude_none=True)


def _color(value: str) -> str:
    """A colour as #rrggbb, from #rrggbb or as people name it (yellow, #ff0, rgb(…))."""
    try:
        return "#{:02x}{:02x}{:02x}".format(*ImageColor.getrgb(value.strip())[:3])
    except ValueError:
        raise ValueError(f"{value!r} isn't a colour: give a name (yellow) or #rrggbb") from None


Color = Annotated[str, AfterValidator(_color)]


class NewClass(BaseModel):
    """A class to add. No colour: it gets the first palette colour not in use; a blank prompt is the name."""

    model_config = ConfigDict(extra="forbid")
    name: ClassName = Field(description="What the objects are, e.g. car.")
    color: Color | None = Field(None, description="A colour name or #rrggbb; empty: one not in use.")
    prompt: Prompt = Field("", description=PROMPT)


HIDDEN = "Hidden on the project's pages (its images and editor don't draw it); the API still has its annotations."


class ClassChange(NewClass):
    """One class of a project's class list, as it's set (set_classes): an existing one (by its
    name) or a new one."""

    hidden: bool = Field(False, description=HIDDEN)


class ClassEdit(BaseModel):
    """A change to one class (edit_classes): which one, by its name, and what changes. None of it
    touches images (renaming does: POST …/classes/rename)."""

    model_config = ConfigDict(extra="forbid")
    name: ClassName = Field(description="The class's name.")
    prompt: Prompt | None = Field(None, description=PROMPT)
    color: Color | None = Field(None, description="A colour name or #rrggbb.")
    hidden: bool | None = Field(None, description=HIDDEN)


# Class colours, in the order classes get them; the first is also an object's with no class yet.
PALETTE = ["#c82e2d", "#2563eb", "#16a34a", "#d97706", "#9333ea", "#0891b2", "#db2777", "#65a30d"]

PAGE_SIZE = 100  # images per page
COVER_SIZE = 4  # images shown on a project's card


def _free_color(listed: list[dict]) -> str:
    """The first palette colour none of these classes has (round the palette when all are taken)."""
    taken = {c.get("color") for c in listed}
    return next((p for p in PALETTE if p not in taken), PALETTE[len(listed) % len(PALETTE)])


def _classes(stored: list[dict]) -> list[dict]:
    """A project's classes as kept (project.classes), each with its prompt (its name unless set) and
    whether it's hidden on the project's pages."""
    return [{"name": c["name"], "prompt": c.get("prompt") or c["name"], "color": c["color"],
             "hidden": c.get("hidden", False)} for c in stored]


def stored_classes(conn: sqlite3.Connection, project_id: int) -> list[dict]:
    row = conn.execute("SELECT classes FROM project WHERE id = ?", (project_id,)).fetchone()
    return json.loads(row["classes"]) if row else []


def classes(conn: sqlite3.Connection, project_id: int) -> list[dict]:
    """The project's classes (name, prompt, colour), in order."""
    return _classes(stored_classes(conn, project_id))


def _project(row: sqlite3.Row) -> dict:
    return dict(row) | {"classes": _classes(json.loads(row["classes"]))}


def _row(conn: sqlite3.Connection, project_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM project WHERE id = ?", (project_id,)).fetchone()


def get(conn: sqlite3.Connection, project_id: int) -> dict | None:
    row = _row(conn, project_id)
    return _project(row) if row else None


def list_all(conn: sqlite3.Connection) -> list[dict]:
    """Every project, most recently changed first, with its image count, how many of its images
    are labelled (have an annotation), how many classes it has and the thumbnails of its first few
    images for its card."""
    rows = conn.execute(
        """SELECT project.*, count(media.id) AS image_count,
                  count(CASE WHEN media.annotations != '[]' THEN 1 END) AS labelled_count,
                  (SELECT json_group_array(sha256) FROM (
                      SELECT sha256 FROM media AS first WHERE first.project_id = project.id ORDER BY first.id LIMIT ?)) AS covers
           FROM project LEFT JOIN media ON media.project_id = project.id
           GROUP BY project.id ORDER BY project.updated_at DESC""",
        (COVER_SIZE,),
    ).fetchall()
    listed = []
    for row in rows:
        project = _project(row) | {"covers": [media.thumb_url(sha) for sha in json.loads(row["covers"])]}
        project["class_count"] = len(project.pop("classes"))
        listed.append(project)
    return listed


def image_count(conn: sqlite3.Connection, project_id: int) -> int:
    return conn.execute("SELECT count(*) FROM media WHERE project_id = ?", (project_id,)).fetchone()[0]


def detail(conn: sqlite3.Connection, project_id: int, page: int = 1, filter: Filter = Filter(),
           page_size: int = PAGE_SIZE) -> dict | None:
    """A project with one page of the images matching `filter` (by file name, `page_size` a page;
    0: none) and their annotations (marking the objects the filter asks for), how many images it
    has and how many are labelled, and how many annotations each class has. Raises FilterError for
    a filter that names a class the project doesn't have."""
    row = _row(conn, project_id)
    if row is None:
        return None
    project = _project(row)
    use = class_use(conn, project_id)
    project["classes"] = [c | {"annotation_count": use.get(c["name"].casefold(), {}).get("annotations", 0)}
                          for c in project["classes"]]
    return project | _page(conn, filter, project_id, page, page_size)


def _page(conn: sqlite3.Connection, filter: Filter, project_id: int | None, page: int, page_size: int) -> dict:
    """One page of the images matching `filter`, a project's or every project's (by file name,
    `page_size` a page; 0: none), marking the objects the filter asks for; how many images there
    are, how many are labelled, and how many match. FilterError as _where."""
    scope, scope_params = ("media.project_id = ?", [project_id]) if project_id is not None else ("true", [])
    total, labelled = conn.execute(f"SELECT count(*), count(CASE WHEN annotations != '[]' THEN 1 END) FROM media WHERE {scope}",
                                   scope_params).fetchone()
    where, params = _where(conn, filter, project_id)
    match_count = total if filter == Filter() else _count(conn, where, params)
    pages = max(1, math.ceil(match_count / page_size)) if page_size else 1
    page = min(max(page, 1), pages)
    rows = conn.execute(f"{_IMAGE_SELECT} WHERE {where} ORDER BY media.name, media.id LIMIT ? OFFSET ?",
                        (*params, page_size, (page - 1) * page_size)).fetchall()
    matches = _matches(conn, filter, [r["id"] for r in rows])
    return {"image_count": total, "labelled_count": labelled, "filter": filter.text(), "match_count": match_count,
            "page": page, "pages": pages, "images": [_image(r, matches and matches[r["id"]]) for r in rows]}


def browse(conn: sqlite3.Connection, filter: Filter = Filter(), page: int = 1, project_id: int | None = None,
           page_size: int = PAGE_SIZE) -> dict:
    """Every project's images matching `filter` (one project's, given one), a page at a time, as
    detail() has a project's, each captioned with its project's name; and the classes to filter and
    draw them with: the project's, or every project's (all_classes). FilterError as detail()."""
    found = _page(conn, filter, project_id, page, page_size)
    names = dict(conn.execute("SELECT id, name FROM project"))
    if project_id is None:
        listed = all_classes(conn)
    else:
        use = class_use(conn, project_id)
        listed = [c | {"annotation_count": use.get(c["name"].casefold(), {}).get("annotations", 0)}
                  for c in classes(conn, project_id)]
    return found | {"classes": listed,
                    "images": [i | {"caption": f"{i['filename']} · {names[i['project_id']]}"} for i in found["images"]]}


def all_classes(conn: sqlite3.Connection) -> list[dict]:
    """The classes of every project, one per name (ignoring case: the first project's spelling and
    colour), each with how many annotations of it there are in all."""
    counts = {name.casefold(): n for name, n in conn.execute("SELECT class, count(*) FROM annotation GROUP BY class")}
    merged: dict[str, dict] = {}
    for (stored,) in conn.execute("SELECT classes FROM project ORDER BY id"):
        for c in _classes(json.loads(stored)):
            merged.setdefault(c["name"].casefold(), c | {"annotation_count": counts.get(c["name"].casefold(), 0)})
    return list(merged.values())


def _count(conn: sqlite3.Connection, where: str, params: list) -> int:
    return conn.execute(f"SELECT count(*) FROM media WHERE {where}", params).fetchone()[0]


def named(conn: sqlite3.Connection, text: str, limit: int = 20) -> dict:
    """Projects and images whose name contains `text` (ignoring case): projects (id, name) and the
    first `limit` images (media id, file name, project id and name), with how many images match."""
    found = conn.execute("SELECT id, name FROM project WHERE instr(lower(name), lower(?)) > 0 ORDER BY name", (text,))
    images = "FROM media JOIN project ON project.id = media.project_id WHERE instr(lower(media.name), lower(?)) > 0"
    rows = conn.execute(f"""SELECT media.id AS media_id, media.name AS filename, project.id AS project_id,
                                   project.name AS project {images} ORDER BY media.name, project.name LIMIT ?""", (text, limit))
    return {"projects": [dict(r) for r in found], "images": [dict(r) for r in rows],
            "image_count": conn.execute(f"SELECT count(*) {images}", (text,)).fetchone()[0]}


def sample(conn: sqlite3.Connection, filter: Filter = Filter(), project_id: int | None = None,
           count: int | None = None, first: int | None = None) -> list[int]:
    """The ids of the images matching `filter`, a project's or across all: all of them (by file
    name), the `first` few, or `count` picked at random. FilterError as detail()."""
    where, params = _where(conn, filter, project_id)
    order = "ORDER BY random()" if count else "ORDER BY media.name, media.id"
    return [r[0] for r in conn.execute(f"SELECT media.id FROM media WHERE {where} {order} LIMIT ?",
                                       (*params, count or first or -1))]


def image(conn: sqlite3.Connection, media_id: int, filter: Filter = Filter()) -> dict | None:
    """One image with its annotations, its project, and where the image is among the project's
    images matching `filter` (by file name): its position, how many match, and the previous and
    next ones."""
    found = conn.execute(f"{_IMAGE_SELECT} WHERE media.id = ?", (media_id,)).fetchone()
    if not found:
        return None
    project_id = found["project_id"]
    where, params = _where(conn, filter, project_id)
    key = (found["name"], media_id)  # file names can repeat, so the id breaks ties

    def neighbour(before: bool) -> int | None:
        op, order = ("<", "DESC") if before else (">", "ASC")
        hit = conn.execute(f"""SELECT id FROM media WHERE {where} AND (name, id) {op} (?, ?)
                               ORDER BY name {order}, id {order} LIMIT 1""", (*params, *key)).fetchone()
        return hit[0] if hit else None

    match_count, position = conn.execute(
        f"SELECT count(*), total((name, id) <= (?, ?)) FROM media WHERE {where}", (*key, *params)).fetchone()
    matches = _matches(conn, filter, [media_id])
    return _image(found, matches and matches[media_id]) | {
        "project": get(conn, project_id),
        "filter": filter.text(),
        "position": int(position),
        "match_count": match_count,
        "previous_id": neighbour(before=True),
        "next_id": neighbour(before=False),
    }


_IMAGE_SELECT = """SELECT media.id, media.project_id, media.name, media.split, media.annotations, media.sha256,
                          media.ext, media.width, media.height FROM media"""


def _image(row: sqlite3.Row, matched: set[int] | None = None) -> dict:
    """An image row for templates and the API: its objects per class, its annotations, and, when
    the filter asks for objects, `matches`: the positions in that list of the objects it found."""
    annotations = json.loads(row["annotations"])
    image = {"media_id": row["id"], "project_id": row["project_id"], "filename": row["name"], "split": row["split"],
             "width": row["width"], "height": row["height"],
             "objects": dict(Counter(a["class"] for a in annotations)),
             "annotations": annotations,
             "url": media.url(row["sha256"], row["ext"]),
             "thumb_url": media.thumb_url(row["sha256"])}
    if matched is not None:
        image["matches"] = sorted(matched)
    return image


def drawn(image: dict, project: dict) -> dict:
    """An image as the project's pages draw it: without the annotations of the project's hidden
    classes (the positions the filter matched follow), those kept aside in `hidden` (for the editor
    to save back as they are). The API answers every annotation."""
    hides = {c["name"].casefold() for c in project["classes"] if c["hidden"]}
    kept = [(i, a) for i, a in enumerate(image["annotations"]) if a["class"].casefold() not in hides]
    shown = image | {"annotations": [a for _, a in kept],
                     "hidden": [a for a in image["annotations"] if a["class"].casefold() in hides]}
    if image.get("matches") is not None:
        place = {i: n for n, (i, _) in enumerate(kept)}
        shown["matches"] = [place[i] for i in image["matches"] if i in place]
    return shown


def images(conn: sqlite3.Connection, media_ids: list[int]) -> dict:
    """Some images, in the order given (those that exist), each as image pages get it, with the
    classes to draw them in: their projects', the first project's spelling and colour for a name
    two of them have."""
    rows = {r["id"]: r for r in conn.execute(f"{_IMAGE_SELECT} WHERE media.id IN (SELECT value FROM json_each(?))",
                                             (json.dumps(media_ids),))}
    found = [_image(rows[m]) for m in media_ids if m in rows]
    drawn: dict[str, dict] = {}
    for project_id in dict.fromkeys(i["project_id"] for i in found):
        for c in classes(conn, project_id):
            drawn.setdefault(c["name"].casefold(), c)
    return {"classes": list(drawn.values()), "images": found}


def image_file(conn: sqlite3.Connection, media_id: int) -> dict | None:
    """One image as a model works on it, read cheaply (on every mouse move of the click tool): its
    file (path) and its project's task. None if there's none."""
    row = conn.execute("""SELECT media.sha256, media.ext, project.task FROM media JOIN project ON project.id = media.project_id
                          WHERE media.id = ?""", (media_id,)).fetchone()
    return row and {"path": media.path(row["sha256"], row["ext"]), "task": row["task"]}


# An annotation `a` of the media row being filtered.
_OF_IMAGE = "a.media_id = media.id"


def _where(conn: sqlite3.Connection, filter: Filter, project_id: int | None = None) -> tuple[str, list]:
    """SQL condition (and parameters) on `media` for the images matching `filter`: a project's, or
    every project's. Raises FilterError if it names a class there isn't."""
    parts, params = (["media.project_id = ?"], [project_id]) if project_id is not None else (["true"], [])
    _known(conn, filter, project_id)
    if filter.show != "all":
        parts.append("media.annotations != '[]'" if filter.show == "labelled" else "media.annotations = '[]'")
    if filter.name:
        parts.append("instr(lower(media.name), lower(?)) > 0")
        params.append(filter.name)
    if filter.objects:
        condition, more = filter.objects.sql(f"(SELECT count(*) FROM annotation a WHERE {_OF_IMAGE})")
        parts.append(condition)
        params += more
    for c in filter.conditions:
        condition, more = _object_sql(c)
        parts.append(f"{'NOT ' if c.negate else ''}EXISTS (SELECT 1 FROM annotation a WHERE {_OF_IMAGE} AND {condition})")
        params += more
    return " AND ".join(parts), params


def _known(conn: sqlite3.Connection, filter: Filter, project_id: int | None) -> None:
    """FilterError for a class the filter names that the project (or no project) has: a typo would
    otherwise find nothing and look like an answer."""
    names = {n for c in filter.conditions for n in c.classes}
    if not names:
        return
    if project_id is not None:
        have = [c["name"] for c in classes(conn, project_id)]
    else:
        have = [r[0] for r in conn.execute("SELECT DISTINCT class FROM annotation")]
    known = {n.casefold() for n in have}
    if missing := sorted(n for n in names if n.casefold() not in known):
        where = "This project has no" if project_id is not None else "No project has a"
        raise FilterError(f"{where} class “{missing[0]}”; the classes: {', '.join(have) or 'none'}")


def _object_sql(c: ObjectCondition) -> tuple[str, list]:
    """SQL condition (and parameters) on an annotation `a`: it meets all of `c` (its sign aside)."""
    parts, params = [], []
    if c.classes:
        parts.append("a.class IN (SELECT value FROM json_each(?))")  # the column's NOCASE decides
        params.append(json.dumps(c.classes))
    for column, r in (("a.w", c.w), ("a.h", c.h), ("a.area", c.area), ("a.aspect", c.aspect)):
        if r:
            condition, more = r.sql(column)
            parts.append(condition)
            params += more
    if c.edge is not None:
        parts.append("a.edge = ?")
        params.append(int(c.edge))
    return " AND ".join(parts) or "true", params


def _wanted(filter: Filter) -> tuple[str, list] | None:
    """SQL condition (and parameters) on an annotation `a`: it's one of the objects the filter
    asks for (meets one of its conditions that aren't negated), or None if it asks for none."""
    wanted = [_object_sql(c) for c in filter.conditions if not c.negate]
    if not wanted:
        return None
    return " OR ".join(f"({condition})" for condition, _ in wanted), [p for _, more in wanted for p in more]


def _matches(conn: sqlite3.Connection, filter: Filter, media_ids: list[int]) -> dict[int, set[int]] | None:
    """For these images, the objects the filter asks for (positions in each image's annotations
    list), or None if it doesn't ask for any."""
    wanted = _wanted(filter)
    if wanted is None:
        return None
    condition, params = wanted
    found: dict[int, set[int]] = {m: set() for m in media_ids}
    rows = conn.execute(f"SELECT a.media_id, a.idx FROM annotation a WHERE a.media_id IN (SELECT value FROM json_each(?)) "
                        f"AND ({condition})", (json.dumps(media_ids), *params))
    for media_id, idx in rows:
        found[media_id].add(idx)
    return found


STAT_FIELDS = ("class", "w", "h", "area", "aspect", "objects")


def stats(conn: sqlite3.Connection, project_id: int, field: str, filter: Filter = Filter(), bins: int = 20) -> dict | None:
    """How a measure is spread over the images matching `filter`: "class" counts objects per
    class; "w", "h", "area" and "aspect" give a histogram of the objects (those the filter asks
    for, or all of them); "objects" gives a histogram of objects per image."""
    if _row(conn, project_id) is None:
        return None
    where, params = _where(conn, filter, project_id)
    if field == "objects":
        rows = conn.execute(f"""SELECT n, count(*) FROM (SELECT (SELECT count(*) FROM annotation a WHERE {_OF_IMAGE}) AS n
                                FROM media WHERE {where}) GROUP BY n ORDER BY n""", params).fetchall()
        return {"field": field, "counts": [{"objects": n, "images": k} for n, k in rows]}
    condition, more = _wanted(filter) or ("true", [])
    objects = f"FROM annotation a JOIN media ON {_OF_IMAGE} WHERE {where} AND ({condition})"
    params += more
    if field == "class":
        rows = conn.execute(f"SELECT a.class, count(*) {objects} GROUP BY a.class ORDER BY 2 DESC", params)
        return {"field": field, "counts": [{"class": name, "objects": n} for name, n in rows]}
    if field not in STAT_FIELDS:
        raise FilterError(f"There are no statistics for “{field}”; try one of {', '.join(STAT_FIELDS)}")
    column = f"a.{field}"
    low, high, total = conn.execute(f"SELECT min({column}), max({column}), count({column}) {objects}", params).fetchone()
    if not total:
        return {"field": field, "objects": 0, "bins": []}
    width = (high - low) / bins or 1
    rows = dict(conn.execute(
        f"SELECT min(CAST(({column} - ?) / ? AS INTEGER), ?) AS b, count(*) {objects} AND {column} IS NOT NULL GROUP BY b",
        (low, width, bins - 1, *params),
    ).fetchall())
    return {
        "field": field,
        "objects": total,
        "bins": [{"from": low + b * width, "to": low + (b + 1) * width, "objects": rows.get(b, 0)} for b in range(bins)],
    }


def create(conn: sqlite3.Connection, name: str, description: str = "", task: Task | None = None,
           annotator: str | None = None, classes: list[ClassChange] = ()) -> int:
    """Create a project, with an annotator (an id in annotators.FINDERS; None: the suggested one)
    and these classes (as set_classes sets them). Caller commits. Raises ValueError as set_classes."""
    project_id = conn.execute("INSERT INTO project (name, description, task, annotator) VALUES (?, ?, ?, ?)",
                              (name, description, task, annotator)).lastrowid
    if classes:
        set_classes(conn, project_id, list(classes))
    return project_id


def update(conn: sqlite3.Connection, project_id: int, name: str, description: str, task: Task | None,
           annotator: str | None) -> None:
    """Change a project's name, description, type or annotator. Its annotations stay as they are.
    Caller commits."""
    conn.execute("UPDATE project SET name = ?, description = ?, task = ?, annotator = ? WHERE id = ?",
                 (name, description, task, annotator, project_id))
    _touch(conn, project_id)


def _active_jobs(conn: sqlite3.Connection, project_id: int) -> int:
    return conn.execute(f"SELECT count(*) FROM job WHERE project_id = ? AND status IN {jobs.ACTIVE}", (project_id,)).fetchone()[0]


def deletion(conn: sqlite3.Connection, project_id: int) -> dict:
    """What deleting the project would remove: its `images` and their `annotations`.
    `jobs`: how many of its jobs are queued or running; while there are, it can't be deleted."""
    count = lambda sql: conn.execute(sql, (project_id,)).fetchone()[0]
    return {
        "images": image_count(conn, project_id),
        "annotations": count("SELECT count(*) FROM annotation a JOIN media m ON m.id = a.media_id WHERE m.project_id = ?"),
        "jobs": _active_jobs(conn, project_id),
    }


def remove(conn: sqlite3.Connection, project_id: int) -> None:
    """Delete a project with its images, and commit. The files stay while History can
    bring the images back (history.purge). Raises ValueError while one of its jobs is queued or running."""
    if _active_jobs(conn, project_id):
        raise ValueError("The project has jobs queued or running; cancel them first")
    with conn:
        conn.execute("DELETE FROM project WHERE id = ?", (project_id,))


def add_images(conn: sqlite3.Connection, project_id: int, media_ids: list[int]) -> list[int]:
    """Copy images (of any project) into this one, with their annotations, names and splits: new
    rows on the same files. One whose file it has already stays as it is. The new rows' ids.
    Caller commits."""
    added = sorted(r[0] for r in conn.execute(
        """INSERT INTO media (project_id, sha256, ext, width, height, name, split, annotations)
           SELECT ?, sha256, ext, width, height, name, split, annotations FROM media
           WHERE id IN (SELECT value FROM json_each(?)) ON CONFLICT DO NOTHING RETURNING id""",
        (project_id, json.dumps(media_ids))))
    _arrived(conn, project_id, added)
    return added


def _arrived(conn: sqlite3.Connection, project_id: int, media_ids: list[int]) -> None:
    """Images that came into the project (copied, moved) with their annotations: the classes it
    hasn't had are added, and those it has are spelt its way, on the images where that changes."""
    stored = stored_classes(conn, project_id)
    count = len(stored)
    for row in conn.execute("SELECT id, annotations FROM media WHERE id IN (SELECT value FROM json_each(?))",
                            (json.dumps(media_ids),)).fetchall():
        objects = json.loads(row["annotations"])
        if (spelt := _spelt(stored, [dict(o) for o in objects])) != objects:
            conn.execute("UPDATE media SET annotations = ? WHERE id = ?", (json.dumps(spelt), row["id"]))
    if len(stored) > count:
        conn.execute("UPDATE project SET classes = ? WHERE id = ?", (json.dumps(stored), project_id))
    if media_ids:
        _touch(conn, project_id)


def _media_row(conn: sqlite3.Connection, media_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT project_id, name, annotations FROM media WHERE id = ?", (media_id,)).fetchone()


def set_annotations(conn: sqlite3.Connection, media_id: int, annotations: list[Annotation]) -> str | None:
    """Replace an image's annotations with these. Returns the image's file name; None if there's no
    such image. Caller commits."""
    row = _media_row(conn, media_id)
    if row is None:
        return None
    save(conn, media_id, row["project_id"], [a.stored() for a in annotations])
    return row["name"]


Change = Literal["delete_objects", "set_class", "remove_images", "copy_images", "move_images"]
EXAMPLES = 4  # images a change names as examples


def change(conn: sqlite3.Connection, project_id: int, filter: Filter, action: Change, class_name: str | None = None,
           to_project: int | None = None) -> dict | None:
    """Change the project's images matching `filter` all at once, as one action (History undoes it
    whole): delete the objects the filter names in brackets, e.g. (car w:<10), or give them the
    class `class_name`; remove the images from the project; or copy or move them, with their
    objects, to `to_project` (images whose file it has already stay as they are, in both). Returns
    how many images (and objects) it changed and the first few images; a preview is this, rolled
    back. None if there's no such project. Caller commits. Raises ValueError saying why it can't."""
    if _row(conn, project_id) is None:
        return None
    media_ids = sample(conn, filter, project_id)
    if action in ("delete_objects", "set_class"):
        if action == "set_class" and not class_name:
            raise ValueError("Give class: the class the objects become")
        return {"action": action} | _rewrite(conn, project_id, filter, media_ids, class_name if action == "set_class" else None)
    if action == "remove_images":
        conn.execute("DELETE FROM media WHERE id IN (SELECT value FROM json_each(?))", (json.dumps(media_ids),))
        _touch(conn, project_id)
        return {"action": action, "images": len(media_ids), "examples": media_ids[:EXAMPLES]}
    if to_project is None or to_project == project_id or _row(conn, to_project) is None:
        raise ValueError("Give to_project: another project's id")
    if action == "move_images":
        moved = [r[0] for r in conn.execute(
            f"""UPDATE OR IGNORE media SET project_id = ?, updated_at = {db.NOW}
               WHERE id IN (SELECT value FROM json_each(?)) RETURNING id""", (to_project, json.dumps(media_ids)))]
        _arrived(conn, to_project, moved)
        _touch(conn, project_id)
        done = moved
    else:
        done = add_images(conn, to_project, media_ids)
    objects = conn.execute("SELECT count(*) FROM annotation WHERE media_id IN (SELECT value FROM json_each(?))",
                           (json.dumps(done),)).fetchone()[0]
    return {"action": action, "images": len(done), "objects": objects, "examples": done[:EXAMPLES],
            "already_in_target": len(media_ids) - len(done)}


def _rewrite(conn: sqlite3.Connection, project_id: int, filter: Filter, media_ids: list[int], class_name: str | None) -> dict:
    """Delete the objects the filter names (class_name None), or give them class_name: those images'
    and objects' counts, and the first images."""
    matches = _matches(conn, filter, media_ids)
    if matches is None:
        raise ValueError("Say which objects, in brackets: e.g. (car w:<10)")
    changed, objects, known = [], 0, stored_classes(conn, project_id)
    for media_id, found in matches.items():
        stored = json.loads(_media_row(conn, media_id)["annotations"])
        if class_name is not None:  # already that class: as it is
            found = {i for i in found if stored[i]["class"].casefold() != class_name.casefold()}
        if not found:
            continue
        save(conn, media_id, project_id, [o for i, o in enumerate(stored) if i not in found] if class_name is None
              else [o | {"class": class_name} if i in found else o for i, o in enumerate(stored)], known)
        changed.append(media_id)
        objects += len(found)
    return {"images": len(changed), "objects": objects, "examples": changed[:EXAMPLES]}


def _spelt(stored: list[dict], objects: list[dict]) -> list[dict]:
    """These annotations with each class spelt as the project (its classes as kept, `stored`) spells
    it; a name it hasn't had is added to `stored`, with a colour of its own."""
    spellings = {c["name"].casefold(): c["name"] for c in stored}
    for o in objects:
        if o["class"].casefold() not in spellings:
            stored.append({"name": o["class"], "color": _free_color(stored)})
            spellings[o["class"].casefold()] = o["class"]
        o["class"] = spellings[o["class"].casefold()]
    return objects


def save(conn: sqlite3.Connection, media_id: int, project_id: int, objects: list[dict],
          stored: list[dict] | None = None) -> None:
    """Write an image's annotations (all of them, as stored), spelt the project's way (_spelt;
    `stored`: its classes, read once for many images), keeping a class it hasn't had, and mark the
    image and its project changed."""
    stored = stored_classes(conn, project_id) if stored is None else stored
    count = len(stored)
    _spelt(stored, objects)
    if len(stored) > count:
        conn.execute("UPDATE project SET classes = ? WHERE id = ?", (json.dumps(stored), project_id))
    conn.execute(f"UPDATE media SET annotations = ?, updated_at = {db.NOW} WHERE id = ?", (json.dumps(objects), media_id))
    _touch(conn, project_id)


def corners(a: dict) -> tuple[float, float, float, float] | None:
    """An annotation's box as corners: its bbox, or its polygon's bounds; None without either."""
    if a.get("bbox"):
        x, y, w, h = a["bbox"]
        return x, y, x + w, y + h
    if a.get("segments"):
        xs, ys = [p[0] for p in a["segments"]], [p[1] for p in a["segments"]]
        return min(xs), min(ys), max(xs), max(ys)
    return None


def with_prompts(listed: list[dict], prompts: dict[str, str]) -> list[dict]:
    """The classes with some prompts changed (class name -> prompt, ignoring case); an empty one is
    the class's name."""
    by_name = {k.casefold(): v for k, v in prompts.items()}
    return [c | {"prompt": by_name[c["name"].casefold()].strip() or c["name"]} if c["name"].casefold() in by_name else c
            for c in listed]


def including(listed: list[dict], names) -> list[dict]:
    """The classes and, after them, those of `names` they haven't (ignoring case): new classes,
    looked for by their names, with no colour yet."""
    have = {c["name"].casefold() for c in listed}
    new = {n.casefold(): n for n in names if n.casefold() not in have}
    return listed + [{"name": n, "prompt": n, "color": None} for n in new.values()]


def set_prompts(conn: sqlite3.Connection, project_id: int, prompts: dict[str, str]) -> None:
    """Change the prompts of some classes (class name -> prompt), the text sent to annotators; a
    name the project hasn't yet becomes a class. An empty prompt goes back to the class's name.
    Caller commits."""
    _keep(conn, project_id, with_prompts(including(classes(conn, project_id), prompts), prompts))


def _keep(conn: sqlite3.Connection, project_id: int, listed: list[dict]) -> list[dict]:
    """Keep these as the project's classes, in order, each with its colour (a free one if it has
    none), its prompt when it isn't the name, and hidden when it is; returns them as classes() does."""
    kept = []
    for c in listed:
        kept.append({"name": c["name"], "color": (c.get("color") or _free_color(kept)).lower()}
                    | ({"prompt": c["prompt"]} if c.get("prompt") and c["prompt"] != c["name"] else {})
                    | ({"hidden": True} if c.get("hidden") else {}))
    conn.execute("UPDATE project SET classes = ? WHERE id = ?", (json.dumps(kept), project_id))
    _touch(conn, project_id)
    return _classes(kept)


def _class(listed: list[dict], name: str) -> dict:
    """The class of these by its name, ignoring case; ValueError naming those there are."""
    found = next((c for c in listed if c["name"].casefold() == name.casefold()), None)
    if found is None:
        raise ValueError(f"There is no class {name!r}; the classes: {', '.join(c['name'] for c in listed) or 'none'}")
    return found


def class_use(conn: sqlite3.Connection, project_id: int) -> dict[str, dict]:
    """Per class (its name, case-folded): how many annotations of it the project's images have, and
    on how many images."""
    rows = conn.execute("""SELECT a.class, count(*), count(DISTINCT a.media_id) FROM annotation a
                           JOIN media m ON m.id = a.media_id WHERE m.project_id = ? GROUP BY a.class""", (project_id,))
    return {name.casefold(): {"annotations": n, "images": k} for name, n, k in rows}


def set_classes(conn: sqlite3.Connection, project_id: int, listed: list[ClassChange]) -> list[dict]:
    """Set the project's class list to these, in this order: classes it has (by name, ignoring case)
    keep their colour unless given another; new ones are added; one left out is removed. Images are
    never changed: a class images have can't be removed (clear its annotations first, or hide it).
    Returns the classes as set. Caller commits.

    Raises ValueError for a name given twice, or a class left out that images have."""
    current = {c["name"].casefold(): c for c in classes(conn, project_id)}
    keep, seen = [], set()
    for c in listed:
        if c.name.casefold() in seen:
            raise ValueError(f"Two classes are named {c.name}")
        seen.add(c.name.casefold())
        old = current.get(c.name.casefold())
        keep.append({"name": old["name"] if old else c.name, "prompt": c.prompt or c.name, "hidden": c.hidden,
                     "color": c.color or (old["color"] if old else None)})
    use = class_use(conn, project_id)
    if used := [current[n]["name"] for n in current if n not in seen and n in use]:
        raise ValueError(f"Images have {used[0]} ({use[used[0].casefold()]['annotations']} annotations): clear them "
                         "first to remove it, or hide it")
    return _keep(conn, project_id, keep)


def edit_classes(conn: sqlite3.Connection, project_id: int, add: list[NewClass] = (),
                 change: list[ClassEdit] = (), remove: list[ClassName] = ()) -> list[dict]:
    """Add, change (colour, prompt, hidden) and remove some of the project's classes; the others stay
    as they are. Added ones go last. Images are never changed (set_classes). Returns the classes as
    set_classes does. Caller commits. Raises ValueError for a class it doesn't have (saying which it
    has) or as set_classes."""
    current = classes(conn, project_id)
    rows = {c["name"].casefold(): {"name": c["name"], "color": c["color"], "hidden": c["hidden"],
                                   "prompt": "" if c["prompt"] == c["name"] else c["prompt"]} for c in current}
    for e in change:
        rows[_class(current, e.name)["name"].casefold()].update(e.model_dump(include={"prompt", "color", "hidden"}, exclude_none=True))
    for name in remove:
        del rows[_class(current, name)["name"].casefold()]
    return set_classes(conn, project_id, [ClassChange(**r) for r in rows.values()]
                       + [ClassChange(**c.model_dump()) for c in add])


def _relabel(conn: sqlite3.Connection, project_id: int, name: str, to: str | None) -> dict:
    """Give a class's annotations on the project's images another class (`to`), or delete them
    (None): how many annotations on how many images."""
    rows = conn.execute("""SELECT DISTINCT m.id, m.annotations FROM media m JOIN annotation a ON a.media_id = m.id
                           WHERE m.project_id = ? AND a.class = ?""", (project_id, name)).fetchall()
    stored, objects = stored_classes(conn, project_id), 0
    for row in rows:
        after = []
        for o in json.loads(row["annotations"]):
            if o["class"].casefold() != name.casefold():
                after.append(o)
            else:
                objects += 1
                if to is not None:
                    after.append(o | {"class": to})
        save(conn, row["id"], project_id, after, stored)
    return {"images": len(rows), "annotations": objects}


def rename_class(conn: sqlite3.Connection, project_id: int, name: str, new_name: str) -> dict:
    """Rename a class, on the project's images too (into another class's name: merged into it, which
    keeps its own colour and prompt): how many annotations on how many images. Caller commits.
    Raises ValueError for a class it doesn't have."""
    current = classes(conn, project_id)
    old = _class(current, name)
    into = next((c for c in current if c["name"].casefold() == new_name.casefold() and c is not old), None)
    listed = [c for c in current if c is not old] if into else [c | {"name": new_name} if c is old else c for c in current]
    _keep(conn, project_id, listed)
    return _relabel(conn, project_id, old["name"], into["name"] if into else new_name)


def clear_class(conn: sqlite3.Connection, project_id: int, name: str) -> dict:
    """Delete a class's annotations from the project's images; the class stays, now on none (so it
    can be removed): how many annotations on how many images. Caller commits. Raises ValueError for
    a class it doesn't have."""
    return _relabel(conn, project_id, _class(classes(conn, project_id), name)["name"], None)


def _touch(conn: sqlite3.Connection, project_id: int) -> None:
    conn.execute(f"UPDATE project SET updated_at = {db.NOW} WHERE id = ?", (project_id,))
