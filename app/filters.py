"""Which of a project's images to show: the filter behind the project page, its API and the agent.

A filter is a set of conditions that must all hold:
  - on the image: labelled or unlabelled, its file name contains some text, how many objects it has;
  - on its objects: the image has (or, negated, has no) object that meets all of a condition's
    parts at once: one of some classes, a width, height, area or aspect range, touching the edge.

Its text form is what URLs carry and what the agent writes; people see it as chips:

    labelled name:night objects:>20 (car w:8..16 h:8..16) -(person)

  labelled, unlabelled      which images (the default is all of them)
  name:TEXT                 the file name contains TEXT (quote it if it has spaces)
  objects:RANGE             how many objects the image has
  (…)  -(…)                 has / has no object with all of these:
      car  car|truck          one of these classes (quote names with spaces: "traffic light")
      w: h: size:RANGE        width, height, or both, in pixels of the original image
      area:RANGE              in square pixels
      aspect:RANGE            width / height
      edge  edge:no           touches the image's edge (within 1 px), or doesn't
  car  -person              short for (car) and -(person)

A RANGE is 8..16, 8.., ..16, >8, <16, >=8, <=16 or 8 (ends included).
"""

import re
import shlex
from typing import Literal

from pydantic import BaseModel

# The text form in short, for the API's ?where= and the agent.
WHERE = """Which images, as a filter; empty for all. Terms, all of which must hold:
labelled | unlabelled; name:TEXT (file name contains); objects:RANGE (how many objects the image has);
(…) the image has an object with ALL of these at once, -(…) it has none: class names (car|truck for
either; quote names with spaces), w:RANGE h:RANGE size:RANGE (pixels; size is both), area:RANGE,
aspect:RANGE (width/height), edge or edge:no (touches the image edge). `car` is short for (car),
`-person` for -(person). RANGE: 8..16, 8.., ..16, >8, <16, >=8, <=16 or 8.
Example: labelled (car size:8..16) -person. Use the project's own class names."""


class FilterError(ValueError):
    """The filter's text can't be read, or names something the project doesn't have."""


class Range(BaseModel):
    """Numbers from min to max; an end that's None is open, and > or < leave the end itself out."""

    min: float | None = None
    max: float | None = None
    min_excluded: bool = False
    max_excluded: bool = False

    @classmethod
    def parse(cls, text: str) -> "Range":
        number = r"-?\d+(?:\.\d+)?"
        if m := re.fullmatch(rf"({number})?\.\.({number})?", text):
            low, high = m.groups()
            if low is None and high is None:
                raise FilterError("A range needs at least one end, like 8..16 or 8..")
            return cls(min=_number(low), max=_number(high))
        if m := re.fullmatch(rf"(>=|<=|>|<)?({number})", text):
            op, value = m.group(1), float(m.group(2))
            return {
                None: cls(min=value, max=value),
                ">=": cls(min=value),
                "<=": cls(max=value),
                ">": cls(min=value, min_excluded=True),
                "<": cls(max=value, max_excluded=True),
            }[op]
        raise FilterError(f"“{text}” isn't a range; write it like 8..16, >8 or <=16")

    def text(self) -> str:
        if self.min_excluded and self.max is None:
            return f">{_show(self.min)}"
        if self.max_excluded and self.min is None:
            return f"<{_show(self.max)}"
        if self.min == self.max:
            return _show(self.min)
        return f"{_show(self.min)}..{_show(self.max)}"

    def label(self, unit: str = "") -> str:
        """In words for a chip: "8–16 px", "≥ 8 px", "< 16 px", "8 px"."""
        unit = f" {unit}" if unit else ""
        if self.min == self.max:
            return f"{_show(self.min)}{unit}"
        if self.min is None:
            return f"{'<' if self.max_excluded else '≤'} {_show(self.max)}{unit}"
        if self.max is None:
            return f"{'>' if self.min_excluded else '≥'} {_show(self.min)}{unit}"
        return f"{_show(self.min)}–{_show(self.max)}{unit}"

    def sql(self, column: str, column_params: tuple = ()) -> tuple[str, list]:
        """SQL condition on a column, or on an expression with its own parameters, and the
        condition's parameters."""
        parts, params = [], []
        if self.min is not None:
            parts.append(f"{column} {'>' if self.min_excluded else '>='} ?")
            params += [*column_params, self.min]
        if self.max is not None:
            parts.append(f"{column} {'<' if self.max_excluded else '<='} ?")
            params += [*column_params, self.max]
        return " AND ".join(parts), params


class ObjectCondition(BaseModel):
    """The image has an object meeting all of these at once (or, negated, has none)."""

    negate: bool = False
    classes: list[str] = []  # class names; any of them (none: any class)
    w: Range | None = None
    h: Range | None = None
    area: Range | None = None
    aspect: Range | None = None
    edge: bool | None = None

    def text(self) -> str:
        sign = "-" if self.negate else ""
        names = "|".join(_quote(c) for c in self.classes)
        parts = [names] if names else []
        if self.w and self.w == self.h:
            parts.append(f"size:{self.w.text()}")
        else:
            parts += [f"{key}:{r.text()}" for key, r in (("w", self.w), ("h", self.h)) if r]
        parts += [f"{key}:{r.text()}" for key, r in (("area", self.area), ("aspect", self.aspect)) if r]
        if self.edge is not None:
            parts.append("edge" if self.edge else "edge:no")
        if parts == [names] and len(self.classes) == 1:
            return sign + names  # the short form: car, -person
        return f"{sign}({' '.join(parts)})"

    def part(self) -> str:
        """The part that says most about it, for the panel that changes it: size, aspect, edge or class."""
        if self.w or self.h:
            return "size"
        if self.aspect:
            return "aspect"
        return "edge" if self.edge is not None else "class"

    def label(self) -> str:
        """In words for a chip: "car or truck · 8–16 px", "no person", "no object · on the edge"."""
        what = " or ".join(self.classes) or ("object" if self.negate else "any object")
        parts = [f"no {what}" if self.negate else what]
        if self.w and self.w == self.h:
            parts.append(self.w.label("px"))
        else:
            parts += [f"{r.label('px')} {word}" for word, r in (("wide", self.w), ("high", self.h)) if r]
        if self.area:
            parts.append(self.area.label("px²"))
        if self.aspect:
            parts.append(f"aspect {self.aspect.label()}")
        if self.edge is not None:
            parts.append("on the edge" if self.edge else "not on the edge")
        return " · ".join(parts)


class Filter(BaseModel):
    show: Literal["all", "labelled", "unlabelled"] = "all"
    name: str = ""
    objects: Range | None = None  # how many objects the image has
    conditions: list[ObjectCondition] = []

    @classmethod
    def parse(cls, text: str) -> "Filter":
        """Read the text form. Raises FilterError with a message for people."""
        try:
            tokens = shlex.split(re.sub(r"(-?\()|(\))", r" \1\2 ", text), posix=True)
        except ValueError:
            raise FilterError("A quote isn't closed") from None
        f, i = cls(), 0
        while i < len(tokens):
            token = tokens[i]
            if token in ("(", "-("):
                end = next((j for j in range(i + 1, len(tokens)) if tokens[j] == ")"), None)
                if end is None:
                    raise FilterError("A bracket isn't closed")
                f.conditions.append(_object_condition(tokens[i + 1 : end], negate=token == "-("))
                i = end + 1
                continue
            if token == ")":
                raise FilterError("A bracket is closed that wasn't opened")
            key, _, value = token.partition(":")
            if token in ("all", "labelled", "unlabelled"):
                f.show = token
            elif key == "name" and value:
                f.name = value
            elif key == "objects" and value:
                f.objects = Range.parse(value)
            elif ":" in token:
                raise FilterError(f"“{key}:” isn't something to filter by here; object sizes go in brackets: (w:8..16)")
            else:  # a class name, as short for (name) or -(name)
                negate = token.startswith("-")
                f.conditions.append(_object_condition([token.removeprefix("-")], negate=negate))
            i += 1
        return f

    def text(self) -> str:
        parts = [] if self.show == "all" else [self.show]
        if self.name:
            parts.append(f"name:{_quote(self.name)}")
        if self.objects:
            parts.append(f"objects:{self.objects.text()}")
        parts += [c.text() for c in self.conditions]
        return " ".join(parts)

    def set(self, **changes) -> "Filter":
        """A copy with these fields changed (e.g. for a link that shows labelled images)."""
        return self.model_copy(update=changes)

    def with_condition(self, condition: ObjectCondition, index: int | None = None) -> "Filter":
        """A copy with this object condition put at `index` (replacing it), or added."""
        conditions = list(self.conditions)
        if index is not None and 0 <= index < len(conditions):
            conditions[index] = condition
        else:
            conditions.append(condition)
        return self.set(conditions=conditions)

    def cleared(self) -> "Filter":
        """A copy with only the choice of labelled / unlabelled kept."""
        return Filter(show=self.show)

    def chips(self, classes: list[dict]) -> list[dict]:
        """Each condition in words: its kind ("name", "objects" or "condition", with its index),
        the filter's text without it (for its ×) and, when it's about one of these classes, that
        class's colour."""
        colors = {c["name"].casefold(): c["color"] for c in classes}
        chips = []
        if self.name:
            chips.append({"kind": "name", "label": f"name contains “{self.name}”", "without": self.set(name="").text(), "color": None})
        if self.objects:
            chips.append({"kind": "objects", "label": self.objects.label("objects"), "without": self.set(objects=None).text(), "color": None})
        for i, c in enumerate(self.conditions):
            rest = self.set(conditions=self.conditions[:i] + self.conditions[i + 1 :])
            color = colors.get(c.classes[0].casefold()) if len(c.classes) == 1 else None
            chips.append({"kind": "condition", "index": i, "label": c.label(), "without": rest.text(), "color": color})
        return chips


def class_ids(names: list[str], classes: list[dict]) -> list[int]:
    """The ids of these class names among a project's (active) classes; FilterError for a name it doesn't have."""
    by_name = {c["name"].casefold(): c["id"] for c in classes}
    missing = [n for n in names if n.casefold() not in by_name]
    if missing:
        raise FilterError(f"This project has no class “{missing[0]}”")
    return [by_name[n.casefold()] for n in names]


_RANGE_KEYS = {"w", "h", "size", "area", "aspect"}


def _object_condition(tokens: list[str], negate: bool) -> ObjectCondition:
    c = ObjectCondition(negate=negate)
    for token in tokens:
        key, colon, value = token.partition(":")
        if token == "edge":
            c.edge = True
        elif key == "edge" and value in ("yes", "no"):
            c.edge = value == "yes"
        elif colon and key in _RANGE_KEYS:
            r = Range.parse(value)
            if key == "size":
                c.w = c.h = r
            else:
                setattr(c, key, r)
        elif colon:
            raise FilterError(f"“{key}:” isn't something an object has; use w:, h:, size:, area:, aspect: or edge")
        else:
            c.classes += [n for n in token.split("|") if n]
    return c


def _number(text: str | None) -> float | None:
    return None if text is None else float(text)


def _show(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


def _quote(text: str) -> str:
    return text if re.fullmatch(r"[^\s()\"|:]+", text) else '"' + text.replace('"', "") + '"'
