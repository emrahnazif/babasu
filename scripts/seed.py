"""Placeholder data for development: a project of generated shape images with matching annotations.

Squares have a box, circles a box and a center keypoint, triangles a box and a polygon.

    uv run python -m scripts.seed

Writes to the real workspace (DATA_DIR). Runs once: does nothing if the project exists.
The images are the same every run (fixed random seed), so media is never stored twice.
"""

import io
import random
from contextlib import closing
from functools import cache

from PIL import Image, ImageDraw

from app import db, media, projects

NAME = "Shapes (sample)"
CLASSES = [("square", "#e11d48"), ("circle", "#2563eb"), ("triangle", "#16a34a")]
IMAGE_COUNT = 24
WIDTH, HEIGHT = 640, 480


def make_image(rng: random.Random) -> tuple[bytes, list[dict]]:
    """Draw 1-4 shapes on a plain background; return JPEG bytes and their annotations."""
    image = Image.new("RGB", (WIDTH, HEIGHT), tuple(rng.randint(200, 245) for _ in range(3)))
    draw = ImageDraw.Draw(image)
    annotations = []
    for _ in range(rng.randint(1, 4)):
        shape = CLASSES[rng.randrange(len(CLASSES))][0]
        size = rng.randint(60, 160)
        x, y = rng.randint(0, WIDTH - size), rng.randint(0, HEIGHT - size)
        fill = tuple(rng.randint(40, 160) for _ in range(3))
        # One object per shape, with every form it has (like COCO and pixelflow's Detection).
        obj = {"class": shape, "bbox": [x, y, size, size]}
        box = (x, y, x + size, y + size)
        if shape == "square":
            draw.rectangle(box, fill=fill)
        elif shape == "circle":
            draw.ellipse(box, fill=fill)
            obj["keypoints"] = [{"name": "center", "point": [x + size / 2, y + size / 2]}]
        else:
            obj["segments"] = [[x + size / 2, y], [x + size, y + size], [x, y + size]]
            draw.polygon([tuple(p) for p in obj["segments"]], fill=fill)
        annotations.append(obj)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue(), annotations


@cache
def shapes() -> list[tuple[str, bytes, list[dict]]]:
    """The images, the same every time: file name, JPEG, annotations."""
    rng = random.Random(0)
    return [(f"shapes_{i:03}.jpg", *make_image(rng)) for i in range(IMAGE_COUNT)]


def make_project(conn, name: str, description: str) -> int:
    """A project of the images, with their annotations and splits (18 train, 4 val, 2 test)."""
    project_id = projects.create(conn, name, description, "detection", None,
                                 [projects.ClassChange(name=n, color=c) for n, c in CLASSES])
    for i, (filename, data, annotations) in enumerate(shapes()):
        split = "train" if i < 18 else "val" if i < 22 else "test"
        media_id, _ = media.store(conn, project_id, data, filename)
        conn.execute("UPDATE media SET split = ? WHERE id = ?", (split, media_id))
        projects.set_annotations(conn, media_id, [projects.Annotation(**a) for a in annotations])
    return project_id


def main() -> None:
    db.migrate()
    with closing(db.connect()) as conn, conn:  # `conn` as a context manager commits at the end
        if conn.execute("SELECT 1 FROM project WHERE name = ?", (NAME,)).fetchone():
            print(f"'{NAME}' already exists; nothing to do.")
            return
        make_project(conn, NAME, "Generated squares, circles and triangles.")
    print(f"Created '{NAME}' with {IMAGE_COUNT} images.")


if __name__ == "__main__":
    main()
