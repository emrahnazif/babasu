"""Annotators: mozo workflows that find a project's objects from its class prompts (product.md
"Annotation"). A project names the one it uses.
"""

from typing import Literal

# Built-in annotators that find objects: id -> name. Their workflows ship with the app.
Finder = Literal["sam3", "grounding-dino-edgetam"]
FINDERS: dict[Finder, str] = {
    "sam3": "SAM 3",
    "grounding-dino-edgetam": "Grounding DINO → EdgeTAM",
}
