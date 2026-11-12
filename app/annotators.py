"""Annotators: mozo workflows that find a project's objects from its class prompts (product.md
"Annotation"). A project names the one it uses.
"""

from typing import Literal, get_args

from app import runs

# The built-in workflows that find objects from class prompts: id -> name.
Finder = Literal["sam3", "grounding-dino", "grounding-dino-edgetam"]
FINDERS: dict[Finder, str] = {id: runs.BUILTIN[id] for id in get_args(Finder)}
